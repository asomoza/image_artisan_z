"""Regression test: load_json_graph must clean up the persistent run graph.

Bug: When loading a LoRA example, load_json_graph() used to set
_persistent_run_graph = None without calling clean_up(). The run graph's
LoKr LoraNode had merged weights into the transformer, and without
before_delete() those weights stayed baked in permanently.

Fix: load_json_graph() now calls _persistent_run_graph.clean_up() before
dropping the reference, which invokes before_delete() on all run-graph nodes.
"""

import torch

from iartisanz.modules.generation.threads.generation_thread import NodeGraphThread


class FakeNode:
    def __init__(self):
        self.callback = None
        self.image_callback = None


class FakeModelNode:
    def __init__(self):
        self.model_name = "test"
        self.path = "/fake"
        self.version = "1"
        self.model_type = 1

    def clear_models(self):
        pass


class FakeGraph:
    """Staging graph stub."""

    def __init__(self):
        self._denoise = FakeNode()
        self._image_send = FakeNode()
        self._model = FakeModelNode()
        self.nodes = []

    def set_abort_function(self, fn):
        pass

    def from_json(self, json_graph, node_classes=None, callbacks=None):
        pass

    def to_json(self, **kwargs):
        return '{"nodes": [], "connections": []}'

    def get_node_by_name(self, name):
        if name == "denoise":
            return self._denoise
        if name == "image_send":
            return self._image_send
        if name == "model":
            return self._model
        return None


class FakeRunGraph:
    """Run graph stub that tracks whether clean_up was called."""

    def __init__(self):
        self.clean_up_called = False
        self.nodes = []

    def set_abort_function(self, fn):
        pass

    def from_json(self, json_graph, node_classes=None, callbacks=None):
        pass

    def update_from_json(self, json_graph, node_classes=None, callbacks=None):
        pass

    def get_node_by_name(self, name):
        return None

    def clean_up(self):
        self.clean_up_called = True


class TestLoadJsonGraphRunGraphCleanup:
    def test_load_json_graph_cleans_up_persistent_run_graph(self):
        """load_json_graph must call clean_up() on the existing run graph
        before dropping it, so LoKr weight merges get restored.
        """
        staging = FakeGraph()
        thread = NodeGraphThread(None, staging, torch.float32, torch.device("cpu"))

        # Simulate a prior generation having created a persistent run graph
        run_graph = FakeRunGraph()
        thread._persistent_run_graph = run_graph

        # Load a new example (rebuilds graph)
        thread.load_json_graph('{"nodes": [], "connections": []}')

        assert run_graph.clean_up_called, (
            "load_json_graph must call clean_up() on the persistent run graph "
            "before dropping it — otherwise LoKr weight merges are orphaned"
        )
        assert thread._persistent_run_graph is None

    def test_load_json_graph_no_run_graph_does_not_crash(self):
        """load_json_graph must handle _persistent_run_graph=None gracefully."""
        staging = FakeGraph()
        thread = NodeGraphThread(None, staging, torch.float32, torch.device("cpu"))
        assert thread._persistent_run_graph is None

        # Should not raise
        thread.load_json_graph('{"nodes": [], "connections": []}')
        assert thread._persistent_run_graph is None

    def test_successive_example_loads_clean_up_each_time(self):
        """Each call to load_json_graph must clean up the current run graph."""
        staging = FakeGraph()
        thread = NodeGraphThread(None, staging, torch.float32, torch.device("cpu"))

        run_graph_1 = FakeRunGraph()
        thread._persistent_run_graph = run_graph_1
        thread.load_json_graph('{"nodes": [], "connections": []}')
        assert run_graph_1.clean_up_called

        # Simulate another generation creating a new run graph
        run_graph_2 = FakeRunGraph()
        thread._persistent_run_graph = run_graph_2
        thread.load_json_graph('{"nodes": [], "connections": []}')
        assert run_graph_2.clean_up_called
