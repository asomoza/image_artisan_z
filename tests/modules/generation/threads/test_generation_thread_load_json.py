import torch

from iartisanz.modules.generation.threads.generation_thread import NodeGraphThread


class FakeNode:
    def __init__(self):
        self.callback = None
        self.image_callback = None


class FakeGraph:
    def __init__(self):
        self.from_json_called = False
        self.from_json_args = None
        self._denoise = FakeNode()
        self._image_send = FakeNode()
        self.nodes = []

    def set_abort_function(self, fn):
        self._abort_fn = fn

    def from_json(self, json_graph, node_classes=None, callbacks=None):
        self.from_json_called = True
        self.from_json_args = (json_graph, node_classes, callbacks)

    def get_node_by_name(self, name):
        if name == "denoise":
            return self._denoise
        if name == "image_send":
            return self._image_send
        return None


def test_load_json_graph_wires_callbacks():
    fake_graph = FakeGraph()
    thread = NodeGraphThread(None, fake_graph, torch.float32, torch.device("cpu"))

    json_graph = '{"hello": "world"}'
    thread.load_json_graph(json_graph)

    assert fake_graph.from_json_called
    assert fake_graph.from_json_args[0] == json_graph

    assert fake_graph.get_node_by_name("denoise").callback == thread.step_progress_update
    assert fake_graph.get_node_by_name("image_send").image_callback == thread.preview_image
