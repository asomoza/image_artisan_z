import os

from diffusers import ZImageControlNetModel

from iartisanz.app.model_manager import ModelHandle, get_model_manager
from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.nodes.node import Node


class ControlNetModelNode(Node):
    """Loads an optional Z-Image ControlNet.

    This node is not part of the default graph. When present and enabled, it
    registers a `controlnet` component with the global ModelManager and outputs
    a `ModelHandle("controlnet")` for downstream nodes.
    """

    PRIORITY = 2
    REQUIRED_INPUTS = ["transformer"]
    OUTPUTS = ["controlnet"]

    def __init__(self, path: str | None = None):
        super().__init__()
        self.path = path

    def update_value(self, path: str | None):
        self.path = path
        self.set_updated()

    def __call__(self):
        if not self.path:
            raise IArtisanZNodeError("No ControlNet path provided", self.__class__.__name__)

        if not os.path.exists(self.path):
            raise IArtisanZNodeError(f"ControlNet file not found: {self.path}", self.__class__.__name__)

        mm = get_model_manager()

        # If already loaded from this exact path, reuse it.
        if mm.has("controlnet"):
            try:
                existing = mm.get_raw("controlnet")
                if getattr(existing, "_iartisanz_source_path", None) == self.path:
                    self.values["controlnet"] = ModelHandle("controlnet")
                    return self.values
            except Exception:
                pass

        transformer = mm.resolve(self.transformer)

        try:
            controlnet = ZImageControlNetModel.from_single_file(self.path, torch_dtype=self.dtype)
            controlnet = ZImageControlNetModel.from_transformer(controlnet, transformer)
        except Exception as e:
            raise IArtisanZNodeError(f"Failed to load ControlNet: {e}", self.__class__.__name__) from e

        # Track provenance for reuse across runs.
        try:
            setattr(controlnet, "_iartisanz_source_path", self.path)
        except Exception:
            pass

        mm.register_active_model(model_id=mm.active_model_id(), controlnet=controlnet)
        self.values["controlnet"] = ModelHandle("controlnet")
        return self.values
