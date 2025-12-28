from PIL import Image

from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.nodes.node import Node


class ImageLoadNode(Node):
    PRIORITY = 2
    OUTPUTS = ["image"]

    SERIALIZE_EXCLUDE = {"image"}

    def __init__(
        self, path: str = None, image: Image = None, weight: float = None, noise: float = None, noise_index: int = 0
    ):
        super().__init__()
        self.path = path
        self.image = image
        self.weight = weight
        self.noise = noise
        self.noise_index = noise_index

    def update_value(self, path: str):
        self.path = path
        self.set_updated()

    def update_image(self, image: Image):
        self.image = image
        self.set_updated()

    def update_path(self, path: str):
        self.path = path

        try:
            self.image = Image.open(self.path)
        except FileNotFoundError as e:
            raise IArtisanZNodeError(e, self.__class__.__name__)

        self.set_updated()

    def update_weight(self, weight: float):
        self.weight = weight
        self.set_updated()

    def update_noise(self, noise: float):
        self.noise = noise
        self.set_updated()

    def update_path_weight_noise(self, path: str, weight: float, noise: float, noise_index: int):
        self.path = path
        self.image = Image.open(self.path)
        self.weight = weight
        self.noise = noise
        self.noise_index = noise_index
        self.set_updated()

    def __call__(self):
        if self.image is None:
            try:
                pil_image = Image.open(self.path)
            except FileNotFoundError as e:
                raise IArtisanZNodeError(e, self.__class__.__name__)
            except AttributeError as e:
                raise IArtisanZNodeError(e, self.__class__.__name__)
        else:
            pil_image = self.image

        if self.weight is not None and self.noise is not None:
            self.values["image"] = {
                "image": pil_image,
                "weight": self.weight,
                "noise": self.noise,
                "noise_index": self.noise_index,
            }
        else:
            self.values["image"] = pil_image

        return self.values
