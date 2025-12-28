from iartisanz.modules.generation.graph.nodes.node import Node


class ImageSendNode(Node):
    REQUIRED_INPUTS = ["image"]
    OUTPUTS = []

    SERIALIZE_INCLUDE = {"image_callback"}
    SERIALIZE_CONVERTERS = {
        "image_callback": (
            # to_json
            lambda cb: cb.__name__ if cb else None,
            # from_json (value, callbacks)
            lambda name, callbacks: None if not name else (callbacks or {}).get(name),
        )
    }

    def __init__(self, image_callback: callable = None):
        super().__init__()
        self.image_callback = image_callback

    def __call__(self):
        if self.image_callback is None:
            return
        self.image_callback(self.image)
