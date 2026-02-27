from __future__ import annotations

import contextlib
import contextvars
import logging
import threading
from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch

from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError


logger = logging.getLogger(__name__)


ModelComponent = Literal[
    "tokenizer",
    "text_encoder",
    "transformer",
    "vae",
    "controlnet",
    "preprocessor",
]

# Attention backend options. The display names are for UI presentation.
# Backend IDs must match what diffusers' set_attention_backend() expects.
# Note: get_available_attention_backends() dynamically detects which are actually usable.
ATTENTION_BACKEND_OPTIONS: list[tuple[str, str]] = [
    ("native", "Auto (PyTorch SDPA)"),
    ("flash", "Flash Attention 2"),
    ("flash_hub", "Flash Attention 2 (Hub)"),
    ("_flash_3_hub", "Flash Attention 3 (Hub)"),
    ("sage", "Sage Attention"),
    ("sage_hub", "Sage Attention (Hub)"),
    ("xformers", "xFormers"),
]

# Mapping from user-facing backends to varlen variants for models that require attention masks.
# Some models (e.g., Z-Image) use variable-length sequences with padding and pass attention
# masks to the attention dispatcher. These need varlen backends that convert masks to cu_seqlens.
VARLEN_BACKEND_MAPPING: dict[str, str] = {
    "flash": "flash_varlen",
    "flash_hub": "flash_varlen_hub",
    "_flash_3_hub": "_flash_3_varlen_hub",
    "sage": "sage_varlen",
    "sage_hub": "sage_varlen",  # No hub variant for sage_varlen
}

# Backends that are NOT compatible with torch.compile due to graph breaks.
# sage_varlen calls torch.cuda.set_device() which is marked as non-traceable by dynamo.
COMPILE_INCOMPATIBLE_BACKENDS: frozenset[str] = frozenset(
    {
        "sage",
        "sage_hub",
        "sage_varlen",
    }
)


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

        # Content hashes for loaded components, keyed by component type.
        # Used by smart model switching to skip reloading unchanged components.
        self._component_hashes: dict[ModelComponent, str] = {}

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

        # Optional compiled module wrappers (torch.compile). Stored separately so
        # toggling compile on/off does not mutate the raw components.
        # Key includes model_id + component + device + dtype + compile options.
        self._compiled_components: dict[tuple[str | None, ModelComponent, str, str, str], Any] = {}

        # Runtime setting for torch.compile. This is NOT saved in the graph JSON,
        # allowing users to toggle compilation without affecting shareable outputs.
        self._use_torch_compile: bool = False

        # Runtime setting for attention backend. This is NOT saved in the graph JSON,
        # allowing users to select the backend without affecting shareable outputs.
        # None means use default ("native" / PyTorch SDPA).
        self._attention_backend: str | None = None

        # Offload strategy settings (runtime config, not saved in graph JSON).
        self._offload_strategy: str = "auto"
        self._group_offload_use_stream: bool = False
        self._group_offload_low_cpu_mem: bool = False

        # Named nn.Module components managed by offload lifecycle.
        self._managed_components: dict[str, Any] = {}

        # Currently applied offload strategy (None = not yet applied).
        self._applied_strategy: str | None = None

    @property
    def attention_backend(self) -> str:
        """Return the current attention backend. Defaults to 'native' (PyTorch SDPA)."""
        with self._lock:
            return self._attention_backend or "native"

    @attention_backend.setter
    def attention_backend(self, value: str | None) -> None:
        """Set the attention backend for transformer inference."""
        with self._lock:
            self._attention_backend = value if value and value != "native" else None

    @property
    def offload_strategy(self) -> str:
        with self._lock:
            return self._offload_strategy

    @offload_strategy.setter
    def offload_strategy(self, value: str) -> None:
        with self._lock:
            self._offload_strategy = value if value else "auto"

    @property
    def group_offload_use_stream(self) -> bool:
        with self._lock:
            return self._group_offload_use_stream

    @group_offload_use_stream.setter
    def group_offload_use_stream(self, value: bool) -> None:
        with self._lock:
            self._group_offload_use_stream = bool(value)

    @property
    def group_offload_low_cpu_mem(self) -> bool:
        with self._lock:
            return self._group_offload_low_cpu_mem

    @group_offload_low_cpu_mem.setter
    def group_offload_low_cpu_mem(self, value: bool) -> None:
        with self._lock:
            self._group_offload_low_cpu_mem = bool(value)

    @property
    def applied_strategy(self) -> str | None:
        with self._lock:
            return self._applied_strategy

    def get_available_attention_backends(self) -> list[tuple[str, str]]:
        """Return list of (backend_id, display_name) tuples for available backends.

        Only returns backends that are actually usable on this system.
        Detects both locally installed packages and HuggingFace Hub kernels.
        """
        available: list[tuple[str, str]] = [("native", "Auto (PyTorch SDPA)")]

        if not torch.cuda.is_available():
            return available

        # Check for HuggingFace kernels package (provides hub variants)
        has_kernels = False
        try:
            import kernels  # noqa: F401

            has_kernels = True
        except ImportError:
            pass

        # Check GPU compute capability for Hopper detection
        is_hopper_gpu = False
        try:
            is_hopper_gpu = torch.cuda.get_device_capability()[0] >= 9
        except Exception:
            pass

        # Check for Flash Attention 2 (local)
        has_flash_local = False
        try:
            import flash_attn  # noqa: F401

            has_flash_local = True
            available.append(("flash", "Flash Attention 2"))
        except ImportError:
            pass

        # Flash Attention 2 from Hub (if kernels installed)
        if has_kernels:
            available.append(("flash_hub", "Flash Attention 2 (Hub)"))

        # Flash Attention 3 - only for Hopper GPUs (SM 9.0+)
        if is_hopper_gpu:
            # Local FA3 requires building from source, so we only check for hub variant
            if has_kernels:
                available.append(("_flash_3_hub", "Flash Attention 3 (Hub)"))

        # Check for Sage Attention (local)
        has_sage_local = False
        try:
            import sageattention  # noqa: F401

            has_sage_local = True
            available.append(("sage", "Sage Attention"))
        except ImportError:
            pass

        # Sage Attention from Hub (if kernels installed)
        if has_kernels:
            available.append(("sage_hub", "Sage Attention (Hub)"))

        # Check for xFormers (local only, no hub variant)
        try:
            import xformers  # noqa: F401

            available.append(("xformers", "xFormers"))
        except ImportError:
            pass

        return available

    def _requires_varlen_backend(self, transformer: Any) -> bool:
        """Check if a transformer model requires varlen attention backends.

        Some models (like Z-Image) use variable-length sequences with padding and
        pass attention masks to the attention dispatcher. These models need varlen
        variants of attention backends that support masks.

        Args:
            transformer: The transformer model to check.

        Returns:
            True if the model requires varlen backends, False otherwise.
        """
        # Z-Image transformer uses variable-length sequences with padding
        # and always passes attn_mask to dispatch_attention_fn
        transformer_class_name = type(transformer).__name__
        return transformer_class_name == "ZImageTransformer2DModel"

    def is_attention_backend_compile_compatible(self, transformer: Any = None) -> bool:
        """Check if the current attention backend is compatible with torch.compile.

        Some backends (e.g., sage_varlen) use operations like torch.cuda.set_device()
        that are not traceable by torch.compile/dynamo, causing graph breaks.

        Args:
            transformer: Optional transformer to check for varlen mapping.
                        If provided and model requires varlen, checks the mapped backend.

        Returns:
            True if the backend is compatible with torch.compile, False otherwise.
        """
        backend = self.attention_backend

        # Check if varlen mapping applies
        if transformer is not None and backend != "native":
            if self._requires_varlen_backend(transformer):
                varlen_backend = VARLEN_BACKEND_MAPPING.get(backend)
                if varlen_backend:
                    backend = varlen_backend

        return backend not in COMPILE_INCOMPATIBLE_BACKENDS

    def apply_attention_backend(self, transformer: Any) -> bool:
        """Apply the current attention backend to a transformer model.

        Args:
            transformer: The transformer model to configure.

        Returns:
            True if backend was applied successfully, False otherwise.
        """
        backend = self.attention_backend

        # Map to varlen variant if the model requires it (e.g., Z-Image uses attention masks)
        # This mapping applies even for "native" check below, but native doesn't need mapping
        effective_backend = backend
        if backend != "native" and self._requires_varlen_backend(transformer):
            varlen_backend = VARLEN_BACKEND_MAPPING.get(backend)
            if varlen_backend:
                effective_backend = varlen_backend
                logger.debug(f"Model requires varlen backend: mapping '{backend}' -> '{effective_backend}'")

        # Try to apply the selected backend using the new dispatcher API
        # Always call set_attention_backend explicitly, including for "native",
        # to ensure the backend is properly set/reset on the transformer processors.
        if hasattr(transformer, "set_attention_backend"):
            try:
                transformer.set_attention_backend(effective_backend)
                logger.debug(f"Applied attention backend '{effective_backend}' to transformer")
                return True
            except Exception as e:
                logger.warning(f"Failed to set attention backend '{effective_backend}': {e}. Falling back to native.")
                # Try to reset to native as fallback
                if effective_backend != "native":
                    try:
                        transformer.set_attention_backend("native")
                    except Exception:
                        pass
                return False

        # Transformer doesn't support the new API - this is fine, native will be used
        logger.debug("Transformer does not support set_attention_backend; using native attention")
        return True

    @property
    def use_torch_compile(self) -> bool:
        """Whether to use torch.compile for the transformer during inference."""
        with self._lock:
            return self._use_torch_compile

    @use_torch_compile.setter
    def use_torch_compile(self, value: bool) -> None:
        """Set whether to use torch.compile for the transformer during inference."""
        with self._lock:
            self._use_torch_compile = bool(value)

    def get_component_hash(self, component: ModelComponent) -> str | None:
        with self._lock:
            return self._component_hashes.get(component)

    def set_component_hash(self, component: ModelComponent, hash_value: str) -> None:
        with self._lock:
            self._component_hashes[component] = hash_value

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

    # ------------------------------------------------------------------
    # Offload strategy lifecycle
    # ------------------------------------------------------------------

    def resolve_offload_strategy(self, device: torch.device | str | None = None) -> str:
        """Resolve the effective offload strategy.

        If strategy is "auto", use VRAM-based heuristics:
          >=20 GB → "auto" (reactive OOM is fine)
          >=12 GB → "model_offload"
          >= 8 GB → "sequential_group_offload"
          <  8 GB → "group_offload"
        """
        strategy = self.offload_strategy
        if strategy != "auto":
            return strategy

        if device is None or not torch.cuda.is_available():
            return "auto"

        try:
            dev = torch.device(device)
            if dev.type != "cuda":
                return "auto"
            idx = dev.index if dev.index is not None else torch.cuda.current_device()
            total_vram = torch.cuda.get_device_properties(idx).total_mem
            total_gb = total_vram / (1024 ** 3)
        except Exception:
            return "auto"

        if total_gb >= 20:
            return "auto"
        if total_gb >= 12:
            return "model_offload"
        if total_gb >= 8:
            return "sequential_group_offload"
        return "group_offload"

    def register_managed_component(self, name: str, module: Any) -> None:
        """Register a named nn.Module for offload lifecycle management.

        Resets ``_applied_strategy`` when a component changes so that
        ``apply_offload_strategy`` will re-apply on next call.
        """
        with self._lock:
            old = self._managed_components.get(name)
            if old is not module:
                self._managed_components[name] = module
                self._applied_strategy = None

    def get_managed_component(self, name: str) -> Any:
        with self._lock:
            return self._managed_components.get(name)

    @staticmethod
    def remove_offload_hooks(module: torch.nn.Module) -> None:
        """Remove diffusers group-offloading hooks from a module and all submodules."""
        try:
            from diffusers.hooks import HookRegistry

            for submod in module.modules():
                if hasattr(submod, "_diffusers_hook"):
                    registry = submod._diffusers_hook
                    if isinstance(registry, HookRegistry):
                        try:
                            registry.remove_hook("group_offloading", recurse=True)
                        except Exception:
                            pass
        except Exception:
            pass

    def prepare_strategy_transition(
        self,
        new_strategy: str,
        device: torch.device | str,
    ) -> None:
        """Clean up old offload strategy before applying a new one."""
        with self._lock:
            old = self._applied_strategy

        if old is None and new_strategy == "auto":
            return

        cpu = torch.device("cpu")
        with self._lock:
            for name, module in self._managed_components.items():
                if not _is_torch_module(module):
                    continue
                # Remove any diffusers hooks from previous group offload
                self.remove_offload_hooks(module)
                # Move to CPU as neutral starting point
                try:
                    module.to(cpu)
                except Exception:
                    pass

    def apply_offload_strategy(self, device: torch.device | str) -> str:
        """Resolve and apply the offload strategy to all managed components.

        Returns the resolved strategy name.
        """
        resolved = self.resolve_offload_strategy(device)

        with self._lock:
            if self._applied_strategy == resolved:
                return resolved

        self.prepare_strategy_transition(resolved, device)

        if resolved == "auto":
            # No-op: reactive OOM via mm.get() handles placement.
            pass
        elif resolved == "model_offload":
            # Keep everything on CPU; use_components() does bulk transfers.
            pass
        elif resolved == "group_offload":
            self._apply_group_offload_hooks(device)
        elif resolved == "sequential_group_offload":
            # Keep on CPU; hooks applied per-node in use_components().
            pass

        with self._lock:
            self._applied_strategy = resolved

        logger.debug("Applied offload strategy: %s", resolved)
        return resolved

    def _apply_group_offload_hooks(self, device: torch.device | str) -> None:
        """Apply diffusers group_offloading hooks to all managed components."""
        try:
            from diffusers.hooks.group_offloading import apply_group_offloading
        except ImportError:
            logger.warning("diffusers.hooks.group_offloading not available; falling back to auto")
            return

        target_device = torch.device(device)
        use_stream = self.group_offload_use_stream
        low_cpu_mem = self.group_offload_low_cpu_mem

        with self._lock:
            components = dict(self._managed_components)

        for name, module in components.items():
            if not _is_torch_module(module):
                continue
            try:
                apply_group_offloading(
                    module,
                    onload_device=target_device,
                    offload_device=torch.device("cpu"),
                    offload_type="leaf_level",
                    use_stream=use_stream,
                    low_cpu_mem_usage=low_cpu_mem,
                )
                logger.debug("Group offload hooks applied to '%s'", name)
            except Exception:
                logger.exception("Failed to apply group offload hooks to '%s'", name)

    @contextlib.contextmanager
    def use_components(
        self,
        *names: str,
        device: torch.device | str,
        strategy_override: str | None = None,
    ):
        """Context manager for component lifecycle during node execution.

        - ``auto``: no-op yield (reactive OOM via ``mm.get()``).
        - ``model_offload``: bulk move to GPU on enter, back to CPU on exit.
        - ``group_offload``: no-op yield (hooks already applied at load time).
        - ``sequential_group_offload``: apply hooks on enter, remove + CPU on exit.
        """
        strategy = strategy_override or self.resolve_offload_strategy(device)

        if strategy == "auto" or strategy == "group_offload":
            yield
            return

        target_device = torch.device(device)
        cpu = torch.device("cpu")

        with self._lock:
            modules = [(n, self._managed_components.get(n)) for n in names]
            modules = [(n, m) for n, m in modules if m is not None and _is_torch_module(m)]

        if strategy == "model_offload":
            # Move requested components to GPU
            for name, module in modules:
                try:
                    module.to(target_device)
                except Exception:
                    logger.warning("Failed to move '%s' to %s", name, target_device)
            try:
                yield
            finally:
                for name, module in modules:
                    try:
                        module.to(cpu)
                    except Exception:
                        pass

        elif strategy == "sequential_group_offload":
            # Apply hooks temporarily
            try:
                from diffusers.hooks.group_offloading import apply_group_offloading
            except ImportError:
                yield
                return

            use_stream = self.group_offload_use_stream
            low_cpu_mem = self.group_offload_low_cpu_mem

            for name, module in modules:
                try:
                    apply_group_offloading(
                        module,
                        onload_device=target_device,
                        offload_device=cpu,
                        offload_type="leaf_level",
                        use_stream=use_stream,
                        low_cpu_mem_usage=low_cpu_mem,
                    )
                except Exception:
                    logger.warning("Failed to apply sequential group offload to '%s'", name)
            try:
                yield
            finally:
                for name, module in modules:
                    try:
                        self.remove_offload_hooks(module)
                        module.to(cpu)
                    except Exception:
                        pass
        else:
            yield

    def clear(self):
        with self._lock:
            self._components.clear()
            self._model_id = None
            self._component_hashes.clear()
            self._lora_sources.clear()
            self._compiled_components.clear()
            self._managed_components.clear()
            self._applied_strategy = None
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

    def clear_compiled(self, component: ModelComponent | None = None) -> None:
        with self._lock:
            if component is None:
                self._compiled_components.clear()
                return
            for key in [k for k in self._compiled_components.keys() if k[1] == component]:
                self._compiled_components.pop(key, None)

    def disable_compiled(self, component: ModelComponent) -> None:
        """Best-effort disable of compilation for a component.

        For Diffusers models, `.compile()` typically sets `_compiled_call_impl`.
        Removing that attribute restores eager execution.
        """

        try:
            module = self.get_raw(component)
        except Exception:
            self.clear_compiled(component)
            return

        if _is_torch_module(module):
            for submod in module.modules():
                if hasattr(submod, "_compiled_call_impl"):
                    try:
                        delattr(submod, "_compiled_call_impl")
                    except Exception:
                        try:
                            setattr(submod, "_compiled_call_impl", None)
                        except Exception:
                            pass

        self.clear_compiled(component)

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
        controlnet: Any = None,
        preprocessor: Any = None,
    ):
        with self._lock:
            self._model_id = model_id
            # Model changed (or re-registered); compiled wrappers are no longer trustworthy.
            self._compiled_components.clear()
            if tokenizer is not None:
                self._components["tokenizer"] = tokenizer
            if text_encoder is not None:
                self._components["text_encoder"] = text_encoder
            if transformer is not None:
                self._components["transformer"] = transformer
            if vae is not None:
                self._components["vae"] = vae
            if controlnet is not None:
                self._components["controlnet"] = controlnet
            if preprocessor is not None:
                self._components["preprocessor"] = preprocessor

    def register_component(self, component: ModelComponent, value: Any) -> None:
        with self._lock:
            if value is None:
                self._components.pop(component, None)
            else:
                self._components[component] = value
            for key in [k for k in self._compiled_components.keys() if k[1] == component]:
                self._compiled_components.pop(key, None)

    def clear_component(self, component: ModelComponent) -> None:
        with self._lock:
            self._components.pop(component, None)
            self._component_hashes.pop(component, None)
            for key in [k for k in self._compiled_components.keys() if k[1] == component]:
                self._compiled_components.pop(key, None)
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

    def get_compiled(
        self,
        component: ModelComponent,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        compile_kwargs: dict[str, Any] | None = None,
    ):
        """Return a torch.compile-wrapped module for `component`.

        This is cached and does NOT overwrite the raw stored component, so callers can
        toggle compiled/eager execution without changing results.
        """

        compile_kwargs = compile_kwargs or {}

        # torch.compile is only available on PyTorch 2.x
        if not hasattr(torch, "compile"):
            return self.get(component, device=device, dtype=dtype)

        module = self.get(component, device=device, dtype=dtype)
        if not _is_torch_module(module):
            return module

        target_device = torch.device(device) if device is not None else (_module_device(module) or torch.device("cpu"))
        dtype_key = str(dtype) if dtype is not None else str(getattr(module, "dtype", None))
        compile_opts_key = repr(sorted(compile_kwargs.items(), key=lambda kv: kv[0]))

        key = (self._model_id, component, str(target_device), dtype_key, compile_opts_key)

        with self._lock:
            cached = self._compiled_components.get(key)
            if cached is not None:
                return cached

        # Diffusers provides regional compilation for some transformer models.
        # Prefer it when available (it is substantially faster to compile and more compatible).
        if component == "transformer" and hasattr(module, "compile_repeated_blocks"):
            compile_kwargs2 = dict(compile_kwargs)
            compile_kwargs2.setdefault("fullgraph", True)

            # When running interactive apps, prompts/shapes can vary between runs.
            # With fullgraph=True, this can trigger multiple recompilations; increase the
            # Dynamo cache limit to avoid hitting the default recompile limit.
            try:
                import os

                import torch._dynamo.config as dynamo_config

                desired_limit = int(os.environ.get("IARTISANZ_TORCH_DYNAMO_CACHE_LIMIT", "64"))
                dynamo_config.cache_size_limit = max(int(dynamo_config.cache_size_limit), desired_limit)
            except Exception:
                pass

            # TorchInductor's CUDA graphs can reuse/overwrite outputs between repeated-block
            # invocations, which can crash with models that call compiled blocks in a loop.
            # Disable CUDA graphs for this regional compilation path.
            if target_device.type == "cuda":
                opts = dict(compile_kwargs2.get("options") or {})
                opts.setdefault("triton.cudagraphs", False)
                compile_kwargs2["options"] = opts
            try:
                module.compile_repeated_blocks(**compile_kwargs2)
            except Exception:
                return module

            with self._lock:
                # Regional compilation mutates submodules in-place; cache the module marker.
                self._compiled_components[key] = module
            return module

        try:
            compiled = torch.compile(module, **compile_kwargs)
        except Exception:
            return module

        with self._lock:
            self._compiled_components[key] = compiled
        return compiled

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
            return ("preprocessor", "vae", "transformer")
        if requesting == "transformer":
            return ("preprocessor", "text_encoder", "vae", "controlnet")
        if requesting == "vae":
            return ("preprocessor", "text_encoder", "transformer", "controlnet")
        if requesting == "controlnet":
            return ("preprocessor", "text_encoder", "vae", "transformer")
        if requesting == "preprocessor":
            return ("text_encoder", "vae", "transformer", "controlnet")
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

    @contextlib.contextmanager
    def default_device_scope(self, device: torch.device | str | None):
        if device is None:
            yield
            return

        if hasattr(torch, "get_default_device") and hasattr(torch, "set_default_device"):
            try:
                previous = torch.get_default_device()
            except Exception:
                previous = None
            try:
                if previous is None or torch.device(previous) != torch.device(device):
                    torch.set_default_device(device)
                    yield
                else:
                    yield
            finally:
                if previous is not None:
                    try:
                        torch.set_default_device(previous)
                    except Exception:
                        pass
        else:
            yield

    def ensure_module_device(
        self,
        module: Any,
        *,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> None:
        if not _is_torch_module(module):
            return

        target_device = torch.device(device)
        needs_move = False
        try:
            for p in module.parameters(recurse=True):
                if p.device != target_device:
                    needs_move = True
                    break
            if not needs_move:
                for b in module.buffers(recurse=True):
                    if b.device != target_device:
                        needs_move = True
                        break
        except Exception:
            needs_move = True

        if needs_move:
            try:
                module.to(device=target_device, dtype=dtype)
            except Exception:
                try:
                    module.to(device=target_device)
                except Exception:
                    pass

        embedders = getattr(module, "control_all_x_embedder", None)
        if isinstance(embedders, torch.nn.ModuleDict):
            for submodule in embedders.values():
                if _is_torch_module(submodule):
                    try:
                        submodule.to(device=target_device, dtype=dtype)
                    except Exception:
                        try:
                            submodule.to(device=target_device)
                        except Exception:
                            pass
        elif isinstance(embedders, dict):
            for submodule in embedders.values():
                if _is_torch_module(submodule):
                    try:
                        submodule.to(device=target_device, dtype=dtype)
                    except Exception:
                        try:
                            submodule.to(device=target_device)
                        except Exception:
                            pass

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

    def is_cuda_oom(self, exc: BaseException) -> bool:
        """Check if an exception is a CUDA out-of-memory error.

        This is the public version of `_is_cuda_oom` for use by nodes and other
        code that needs to detect OOM conditions.

        Args:
            exc: The exception to check.

        Returns:
            True if the exception indicates CUDA OOM, False otherwise.
        """
        return self._is_cuda_oom(exc)

    def free_vram_for_forward_pass(
        self,
        *,
        preserve: tuple[ModelComponent, ...] = (),
    ) -> int:
        """Best-effort VRAM cleanup for retrying a forward pass after OOM.

        Offloads components to CPU (respecting `preserve`) and clears CUDA cache.
        This method is intended for reactive OOM recovery during model forward
        passes (not model loading, which is handled by `get()`).

        Args:
            preserve: Components that should NOT be offloaded (e.g., the ones
                      currently needed for the forward pass).

        Returns:
            Number of components offloaded.
        """
        if not self.offload_on_cuda_oom:
            return 0

        offloaded = 0
        # Offload order: preprocessor, text_encoder, then larger models if needed
        offload_candidates: tuple[ModelComponent, ...] = (
            "preprocessor",
            "text_encoder",
            "controlnet",
            "vae",
            "transformer",
        )

        with self._lock:
            for component in offload_candidates:
                if component in preserve:
                    continue
                if component not in self._components:
                    continue
                obj = self._components.get(component)
                if obj is None or not _is_torch_module(obj):
                    continue

                # Check if already on CPU
                current_device = _module_device(obj)
                if current_device is not None and current_device.type == "cpu":
                    continue

                self._offload_to_cpu_unlocked(component)
                offloaded += 1
                logger.debug(f"[OOM recovery] Offloaded '{component}' to CPU")

        # Clear CUDA cache after offloading
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

        return offloaded


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
