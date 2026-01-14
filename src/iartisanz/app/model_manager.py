from __future__ import annotations

import contextlib
import contextvars
import threading
from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch

from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError


ModelComponent = Literal[
    "tokenizer",
    "text_encoder",
    "transformer",
    "vae",
]


@dataclass(frozen=True)
class ModelHandle:
    component: ModelComponent


def _is_torch_module(obj: Any) -> bool:
    return isinstance(obj, torch.nn.Module)


def _module_device(module: torch.nn.Module) -> Optional[torch.device]:
    try:
        for p in module.parameters(recurse=True):
            return p.device
        for b in module.buffers(recurse=True):
            return b.device
    except Exception:
        return None
    return None


class ModelManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._components: dict[ModelComponent, Any] = {}
        self._model_id: str | None = None

        # Per-thread defaults for where models should live while the graph runs.
        # Nodes should not be responsible for model placement; they run under a scope.
        self._scope_local = threading.local()

        # Device placement/offload policy:
        # - Prefer keeping models on GPU if VRAM allows.
        # - Only offload to CPU when needed (CUDA OOM while moving a component).
        # - Prioritize keeping transformer + VAE on GPU during generation.
        self.offload_on_cuda_oom: bool = True

        # Optional per-component default device override.
        # If set, this wins over the current device_scope default.
        self._default_component_device: dict[ModelComponent, torch.device] = {}

        # Track LoRA adapter origin so we can reuse or force-reload safely.
        # Key is adapter_name as used by the transformer.
        self._lora_sources: dict[str, str] = {}

    def active_model_id(self) -> str | None:
        with self._lock:
            return self._model_id

    def is_active_model(self, model_id: str | None) -> bool:
        with self._lock:
            return model_id is not None and self._model_id == model_id

    def get_raw(self, component: ModelComponent) -> Any:
        """Return the stored component without moving devices/dtypes."""
        with self._lock:
            if component not in self._components or self._components[component] is None:
                raise IArtisanZNodeError(
                    f"Model component '{component}' is not available (active model not loaded?).",
                    self.__class__.__name__,
                )
            return self._components[component]

    def clear(self):
        with self._lock:
            self._components.clear()
            self._model_id = None
            self._lora_sources.clear()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

    def get_lora_source(self, adapter_name: str) -> str | None:
        with self._lock:
            return self._lora_sources.get(adapter_name)

    def set_lora_source(self, adapter_name: str, path: str) -> None:
        with self._lock:
            if adapter_name:
                self._lora_sources[adapter_name] = path

    def clear_lora_source(self, adapter_name: str) -> None:
        with self._lock:
            self._lora_sources.pop(adapter_name, None)

    def _get_scope_stack(self) -> list[tuple[torch.device | None, torch.dtype | None]]:
        stack = getattr(self._scope_local, "stack", None)
        if stack is None:
            stack = []
            self._scope_local.stack = stack
        return stack

    def _scoped_defaults(self) -> tuple[torch.device | None, torch.dtype | None]:
        stack = self._get_scope_stack()
        if not stack:
            return None, None
        return stack[-1]

    @contextlib.contextmanager
    def device_scope(
        self,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype | None = None,
    ):
        stack = self._get_scope_stack()

        scoped_device = torch.device(device) if device is not None else None
        stack.append((scoped_device, dtype))
        try:
            yield self
        finally:
            # Always unwind even if a node errors.
            if stack:
                stack.pop()

    def register_active_model(
        self,
        *,
        model_id: str | None = None,
        tokenizer: Any = None,
        text_encoder: Any = None,
        transformer: Any = None,
        vae: Any = None,
    ):
        with self._lock:
            self._model_id = model_id
            if tokenizer is not None:
                self._components["tokenizer"] = tokenizer
            if text_encoder is not None:
                self._components["text_encoder"] = text_encoder
            if transformer is not None:
                self._components["transformer"] = transformer
            if vae is not None:
                self._components["vae"] = vae

    def has(self, component: ModelComponent) -> bool:
        with self._lock:
            return component in self._components and self._components[component] is not None

    def set_default_device(self, component: ModelComponent, device: torch.device | str | None):
        """Set (or clear) a per-component default device.

        If set, `get(component)` will use this device when the caller does not
        explicitly pass a device and regardless of the active device_scope.
        """
        with self._lock:
            if device is None:
                self._default_component_device.pop(component, None)
                return
            self._default_component_device[component] = torch.device(device)

    def _is_cuda_oom(self, exc: BaseException) -> bool:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        msg = str(exc).lower()
        return "out of memory" in msg and "cuda" in msg

    def _cuda_offload_order(self, requesting: ModelComponent) -> tuple[ModelComponent, ...]:
        # Per-request offload priority.
        # - When we need the text encoder on GPU, we are willing to temporarily offload
        #   VAE/transformer to make it fit.
        # - When we need transformer or VAE on GPU, we offload text encoder first.
        if requesting == "text_encoder":
            return ("vae", "transformer")
        if requesting == "transformer":
            return ("text_encoder", "vae")
        if requesting == "vae":
            return ("text_encoder", "transformer")
        return ()

    def _offload_for_cuda_unlocked(self, requesting: ModelComponent):
        for candidate in self._cuda_offload_order(requesting):
            self._offload_to_cpu_unlocked(candidate)

            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            # Offload one at a time; caller will retry allocation/move.
            yield candidate

    def get(
        self,
        component: ModelComponent,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        with self._lock:
            if component not in self._components or self._components[component] is None:
                raise IArtisanZNodeError(
                    f"Model component '{component}' is not available (active model not loaded?).",
                    self.__class__.__name__,
                )

            obj = self._components[component]

            scoped_device, scoped_dtype = self._scoped_defaults()
            if device is None:
                device = self._default_component_device.get(component, scoped_device)
            if dtype is None:
                dtype = scoped_dtype

            # Tokenizers are not torch modules; leave them as-is.
            if device is None or not _is_torch_module(obj):
                return obj

            target_device = torch.device(device)

            current_device = _module_device(obj)
            if current_device is None or current_device != target_device:
                try:
                    obj = obj.to(device=target_device, dtype=dtype) if dtype is not None else obj.to(target_device)
                except Exception as e:
                    # Only offload when needed (OOM), and retry with prioritized offloading.
                    if self.offload_on_cuda_oom and target_device.type == "cuda" and self._is_cuda_oom(e):
                        moved = False
                        for _offloaded in self._offload_for_cuda_unlocked(component):
                            try:
                                obj = (
                                    obj.to(device=target_device, dtype=dtype)
                                    if dtype is not None
                                    else obj.to(target_device)
                                )
                                moved = True
                                break
                            except Exception as retry_exc:
                                if not self._is_cuda_oom(retry_exc):
                                    raise IArtisanZNodeError(
                                        f"Failed moving '{component}' to device {target_device}: {retry_exc}",
                                        self.__class__.__name__,
                                    ) from retry_exc
                                continue

                        if not moved:
                            raise IArtisanZNodeError(
                                f"Failed moving '{component}' to device {target_device} (CUDA OOM)",
                                self.__class__.__name__,
                            ) from e
                    else:
                        raise IArtisanZNodeError(
                            f"Failed moving '{component}' to device {target_device}: {e}",
                            self.__class__.__name__,
                        ) from e
                self._components[component] = obj

            return obj

    def resolve(
        self,
        value: Any,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        if isinstance(value, ModelHandle):
            return self.get(value.component, device=device, dtype=dtype)
        return value

    def offload_to_cpu(self, component: ModelComponent):
        with self._lock:
            self._offload_to_cpu_unlocked(component)

    def _offload_to_cpu_unlocked(self, component: ModelComponent):
        if component not in self._components:
            return
        obj = self._components.get(component)
        if obj is None or not _is_torch_module(obj):
            return
        try:
            self._components[component] = obj.to("cpu")
        except Exception:
            # If the module can't be moved (e.g., sharded device_map), keep as-is for now.
            return


_MODEL_MANAGER_SINGLETON: ModelManager | None = None
_MODEL_MANAGER_SINGLETON_LOCK = threading.Lock()
_CURRENT_MODEL_MANAGER: contextvars.ContextVar[ModelManager | None] = contextvars.ContextVar(
    "iartisanz_current_model_manager",
    default=None,
)


def set_global_model_manager(manager: ModelManager | None) -> None:
    """Set the process-wide default ModelManager.

    This is the simplest way to make a single manager available application-wide
    (including across threads). If set to None, a new singleton will be created
    lazily on next `get_model_manager()` call.
    """

    global _MODEL_MANAGER_SINGLETON
    with _MODEL_MANAGER_SINGLETON_LOCK:
        _MODEL_MANAGER_SINGLETON = manager


@contextlib.contextmanager
def use_model_manager(manager: ModelManager):
    """Temporarily bind `get_model_manager()` to a specific manager.

    This is context-local (uses ContextVar). It is useful when you want to
    override the manager for a limited scope without changing the global
    singleton.
    """

    token = _CURRENT_MODEL_MANAGER.set(manager)
    try:
        yield manager
    finally:
        _CURRENT_MODEL_MANAGER.reset(token)


@contextlib.contextmanager
def model_scope(*, device: torch.device | str | None, dtype: torch.dtype | None = None):
    """Convenience scope for the current application model manager."""

    mm = get_model_manager()
    with mm.device_scope(device=device, dtype=dtype):
        yield mm


def get_model_manager() -> ModelManager:
    global _MODEL_MANAGER_SINGLETON

    current = _CURRENT_MODEL_MANAGER.get()
    if current is not None:
        return current

    if _MODEL_MANAGER_SINGLETON is None:
        with _MODEL_MANAGER_SINGLETON_LOCK:
            if _MODEL_MANAGER_SINGLETON is None:
                _MODEL_MANAGER_SINGLETON = ModelManager()
    return _MODEL_MANAGER_SINGLETON
