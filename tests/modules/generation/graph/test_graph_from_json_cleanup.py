"""Regression test: from_json must call before_delete on existing nodes.

Bug: loading a LoRA example rebuilds the graph via from_json(), which used to
call self.nodes.clear() without invoking before_delete() on existing nodes.
This left LoKr weight merges baked into the transformer permanently.
"""

from unittest.mock import MagicMock, patch

from iartisanz.modules.generation.graph.iartisanz_node_graph import ImageArtisanZNodeGraph
from iartisanz.modules.generation.graph.nodes.node import Node


class StubNode(Node):
    """Minimal node for testing lifecycle callbacks."""

    REQUIRED_INPUTS = []
    OUTPUTS = ["out"]

    def __init__(self):
        super().__init__()
        self.before_delete_called = False

    def before_delete(self):
        self.before_delete_called = True

    def __call__(self):
        self.values["out"] = 1
        return self.values


def _make_empty_graph_json():
    """Minimal valid graph JSON with no nodes."""
    return '{"nodes": [], "connections": []}'


class TestFromJsonCleanup:
    def test_from_json_calls_before_delete_on_existing_nodes(self):
        """from_json must call before_delete on all existing nodes before replacing."""
        graph = ImageArtisanZNodeGraph()

        # Add some nodes manually
        node_a = StubNode()
        node_a.id = 1
        node_a.name = "a"
        node_b = StubNode()
        node_b.id = 2
        node_b.name = "b"
        graph.nodes.append(node_a)
        graph.nodes.append(node_b)
        graph.node_counter = 3

        # Load a new graph (replaces all nodes)
        graph.from_json(_make_empty_graph_json(), node_classes={})

        assert node_a.before_delete_called, "before_delete not called on node_a during from_json"
        assert node_b.before_delete_called, "before_delete not called on node_b during from_json"

    def test_from_json_clears_nodes_after_cleanup(self):
        """After from_json, old nodes must be gone and new ones present."""
        graph = ImageArtisanZNodeGraph()

        old_node = StubNode()
        old_node.id = 1
        old_node.name = "old"
        graph.nodes.append(old_node)
        graph.node_counter = 2

        # Load new graph with a node
        new_graph_json = '{"nodes": [{"id": 10, "class": "StubNode", "name": "new"}], "connections": []}'
        graph.from_json(new_graph_json, node_classes={"StubNode": StubNode})

        assert old_node.before_delete_called
        node_names = [n.name for n in graph.nodes]
        assert "old" not in node_names
        assert "new" in node_names
