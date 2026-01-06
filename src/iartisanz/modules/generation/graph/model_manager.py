from __future__ import annotations

import contextlib
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
    """Lightweight reference passed through the node graph.

    The heavy model objects live in ModelManager; nodes exchange handles.
    """

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
    """Very small first-pass model manager.

    Goals (v1):
    - accept/register models when loaded
    - let nodes request models by component name
    - keep GPU memory bounded via simple offload policy
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._components: dict[ModelComponent, Any] = {}
        self._model_id: str | None = None

        # Per-thread defaults for where models should live while the graph runs.
        # Nodes should not be responsible for model placement; they run under a scope.
        self._scope_local = threading.local()

        # Simple policy: when a component is moved to CUDA, offload the other
        # heavy components (text_encoder/transformer/vae) back to CPU.
        self.offload_others_on_cuda_acquire: bool = True
        self._gpu_managed: set[ModelComponent] = {"text_encoder", "transformer", "vae"}

    def clear(self):
        with self._lock:
            self._components.clear()
            self._model_id = None
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

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
        """Temporarily set default device/dtype for model components.

        Within this scope, calls to `get()` / `resolve()` without an explicit
        `device` / `dtype` will use these defaults.
        """

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
                device = scoped_device
            if dtype is None:
                dtype = scoped_dtype

            # Tokenizers are not torch modules; leave them as-is.
            if device is None or not _is_torch_module(obj):
                return obj

            target_device = torch.device(device)

            # Optional offload policy: if we're moving something to CUDA, push the others to CPU.
            if self.offload_others_on_cuda_acquire and target_device.type == "cuda" and component in self._gpu_managed:
                for other in sorted(self._gpu_managed):
                    if other != component:
                        self._offload_to_cpu_unlocked(other)

            current_device = _module_device(obj)
            if current_device is None or current_device != target_device:
                try:
                    obj = obj.to(device=target_device, dtype=dtype) if dtype is not None else obj.to(target_device)
                except Exception as e:
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


def get_model_manager() -> ModelManager:
    global _MODEL_MANAGER_SINGLETON
    if _MODEL_MANAGER_SINGLETON is None:
        with _MODEL_MANAGER_SINGLETON_LOCK:
            if _MODEL_MANAGER_SINGLETON is None:
                _MODEL_MANAGER_SINGLETON = ModelManager()
    return _MODEL_MANAGER_SINGLETON
