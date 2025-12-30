from PIL import Image

from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.nodes.node import Node


class ImageLoadNode(Node):
    PRIORITY = 2
    OUTPUTS = ["image"]

    SERIALIZE_EXCLUDE = {"image"}

    def __init__(self, path: str = None, image: Image = None):
        super().__init__()
        self.path = path
        self.image = image

    def update_value(self, path: str):
        self.path = path
        self.set_updated()

    def update_image(self, image: Image):
        self.image = image
        self.set_updated()

    def update_path(self, path: str):
        self.path = path

        try:
            self.image = Image.open(self.path).convert("RGB")
        except FileNotFoundError as e:
            raise IArtisanZNodeError(e, self.__class__.__name__)

        self.set_updated()

    def __call__(self):
        if self.image is None:
            try:
                pil_image = Image.open(self.path).convert("RGB")
            except FileNotFoundError as e:
                raise IArtisanZNodeError(e, self.__class__.__name__)
            except AttributeError as e:
                raise IArtisanZNodeError(e, self.__class__.__name__)
        else:
            pil_image = self.image

        self.values["image"] = pil_image

        return self.values
