import json

import pytest

from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.iartisanz_node_graph import ImageArtisanZNodeGraph
from iartisanz.modules.generation.graph.nodes.node import Node


class SourceNode(Node):
    OUTPUTS = ["value"]

    def __init__(self, value=0, calls=None):
        super().__init__()
        self.value = value
        self._calls = calls if calls is not None else []

    def __call__(self):
        self.values["value"] = self.value
        self._calls.append(self.name)


class ManualSourceNode(Node):
    OUTPUTS = ["value"]

    def __init__(self, *, initial=None, calls=None):
        super().__init__()
        self._value = initial
        self._calls = calls if calls is not None else []

    def update_value(self, value):
        self._value = value
        self.set_updated()

    def __call__(self):
        self.values["value"] = self._value
        self._calls.append(self.name)


class AppendInputsNode(Node):
    REQUIRED_INPUTS = ["in_value"]
    OUTPUTS = ["values"]

    def __init__(self, calls=None):
        super().__init__()
        self._calls = calls if calls is not None else []

    def __call__(self):
        v = self.in_value
        incoming = v if isinstance(v, list) else [v]

        existing = self.values.get("values", [])
        existing = existing if isinstance(existing, list) else [existing]

        self.values["values"] = existing + incoming
        self._calls.append(self.name)


class PassthroughNode(Node):
    REQUIRED_INPUTS = ["in_value"]
    OUTPUTS = ["value"]

    def __init__(self, calls=None):
        super().__init__()
        self._calls = calls if calls is not None else []

    def __call__(self):
        self.values["value"] = self.in_value
        self._calls.append(self.name)


class MultiOutputNode(Node):
    OUTPUTS = ["a", "b"]

    def __init__(self, *, a=1, b=2, calls=None):
        super().__init__()
        self.a = a
        self.b = b
        self._calls = calls if calls is not None else []

    def update_values(self, a, b):
        self.a = a
        self.b = b
        self.set_updated()

    def __call__(self):
        self.values["a"] = self.a
        self.values["b"] = self.b
        self._calls.append(self.name)


class MultiInputNode(Node):
    REQUIRED_INPUTS = ["in_a", "in_b"]
    OUTPUTS = ["out_sum"]

    def __init__(self, calls=None):
        super().__init__()
        self._calls = calls if calls is not None else []

    def __call__(self):
        self.values["out_sum"] = self.in_a + self.in_b
        self._calls.append(self.name)


class MultiInputOptionalNode(Node):
    REQUIRED_INPUTS = ["in_a"]
    OUTPUTS = ["out_sum"]
    OPTIONAL_INPUTS = ["in_b"]

    def __init__(self, calls=None):
        super().__init__()
        self._calls = calls if calls is not None else []

    def __call__(self):
        self.values["out_sum"] = self.in_a + (self.in_b if self.in_b is not None else 0)
        self._calls.append(self.name)


class AbortNode(Node):
    OUTPUTS = ["value"]

    def __init__(self, graph: ImageArtisanZNodeGraph, calls):
        super().__init__()
        self._graph = graph
        self._calls = calls

    def __call__(self):
        self._calls.append(self.name)
        self.values["value"] = 123
        self._graph.abort_graph()


class ErrorNode(Node):
    OUTPUTS = ["value"]

    def __call__(self):
        raise IArtisanZNodeError("boom", self.__class__.__name__)


class ChangingSourceNode(Node):
    OUTPUTS = ["value"]

    def __init__(self, *, start=0, step=1, calls=None):
        super().__init__()
        self._current = start
        self._step = step
        self._calls = calls if calls is not None else []

    def __call__(self):
        self._current += self._step
        self.values["value"] = self._current
        self._calls.append(self.name)


class CollectInputsNode(Node):
    REQUIRED_INPUTS = ["in_value"]
    OUTPUTS = ["values"]

    def __init__(self, calls=None):
        super().__init__()
        self._calls = calls if calls is not None else []

    def __call__(self):
        v = self.in_value
        # Normalize to list so tests can assert consistently.
        if isinstance(v, list):
            values = v
        else:
            values = [v]
        self.values["values"] = values
        self._calls.append(self.name)


def test_add_node_sets_id_name_and_abort_callable():
    g = ImageArtisanZNodeGraph()
    n = SourceNode(value=5)

    called = {"abort": 0}

    def _abort():
        called["abort"] += 1

    g.set_abort_function(_abort)
    g.add_node(n, name="source")

    assert n.id == 0
    assert n.name == "source"
    assert n.abort_callable is _abort
    assert n.updated is True


def test_add_node_rejects_duplicate_name():
    g = ImageArtisanZNodeGraph()
    g.add_node(SourceNode(), name="dup")
    with pytest.raises(IArtisanZNodeError):
        g.add_node(SourceNode(), name="dup")


def test_get_node_helpers_return_expected_nodes():
    g = ImageArtisanZNodeGraph()
    a = SourceNode()
    b = SourceNode()
    g.add_node(a, name="a")
    g.add_node(b, name="b")

    assert g.get_node(a.id) is a
    assert g.get_node_by_name("b") is b
    assert g.get_node(99999) is None
    assert g.get_node_by_name("missing") is None


def test_delete_node_disconnects_related_nodes_and_removes_from_graph():
    g = ImageArtisanZNodeGraph()
    src = SourceNode(value=1)
    sink = PassthroughNode()

    g.add_node(src, name="src")
    sink.connect("in_value", src, "value")
    g.add_node(sink, name="sink")

    assert sink.dependencies == [src]
    assert src.dependents == [sink]
    assert "in_value" in sink.connections

    g.delete_node(src)

    assert g.get_node_by_name("src") is None
    assert sink.dependencies == []
    assert dict(sink.connections) == {}


def test_graph_call_executes_dependencies_before_dependents_even_with_priority():
    calls: list[str] = []

    class LowPrioritySource(SourceNode):
        PRIORITY = 0

    class HighPriorityDependent(PassthroughNode):
        PRIORITY = 100

    g = ImageArtisanZNodeGraph()
    src = LowPrioritySource(value=7, calls=calls)
    dep = HighPriorityDependent(calls=calls)

    g.add_node(src, name="src")
    dep.connect("in_value", src, "value")
    g.add_node(dep, name="dep")

    g()

    assert calls == ["src", "dep"]
    assert dep.values["value"] == 7
    assert g.updated is True
    assert src.updated is False
    assert dep.updated is False

    # Second run should be a no-op because nodes are no longer updated.
    calls.clear()
    g()
    assert calls == []
    assert g.updated is False


def test_graph_call_skips_disabled_nodes():
    calls: list[str] = []
    g = ImageArtisanZNodeGraph()

    a = SourceNode(value=1, calls=calls)
    b = SourceNode(value=2, calls=calls)
    g.add_node(a, name="a")
    g.add_node(b, name="b")

    a.enabled = False
    a.updated = True
    b.updated = True

    g()
    assert calls == ["b"]


def test_graph_call_raises_on_cycle():
    class CycleNode(Node):
        REQUIRED_INPUTS = ["in_value"]
        OUTPUTS = ["value"]

        def __call__(self):
            self.values["value"] = self.in_value

    g = ImageArtisanZNodeGraph()
    a = CycleNode()
    b = CycleNode()
    g.add_node(a, name="a")
    g.add_node(b, name="b")

    a.connect("in_value", b, "value")
    b.connect("in_value", a, "value")

    with pytest.raises(ValueError, match="cycle"):
        g()


def test_abort_graph_stops_execution_and_calls_abort_function():
    calls: list[str] = []
    g = ImageArtisanZNodeGraph()

    abort_called = {"n": 0}

    def _abort():
        abort_called["n"] += 1

    g.set_abort_function(_abort)

    first = AbortNode(g, calls)
    second = SourceNode(value=999, calls=calls)
    g.add_node(first, name="first")
    g.add_node(second, name="second")

    g()

    assert calls == ["first"]
    assert abort_called["n"] == 1
    assert first.abort is False
    assert g.executing_node is None


def test_graph_propagates_iartisanz_node_error():
    g = ImageArtisanZNodeGraph()
    g.add_node(ErrorNode(), name="err")
    with pytest.raises(IArtisanZNodeError, match="boom"):
        g()


def test_to_json_from_json_roundtrip_restores_nodes_and_connections():
    g = ImageArtisanZNodeGraph()
    g.additional_generation_data = {"foo": "bar"}

    src = MultiOutputNode(a=10, b=20)
    sink = PassthroughNode()

    g.add_node(src, name="src")
    sink.connect("in_value", src, "a")
    g.add_node(sink, name="sink")

    json_str = g.to_json()
    payload = json.loads(json_str)
    assert payload["format_version"] == 1
    assert payload["additional_generation_data"] == {"foo": "bar"}

    g2 = ImageArtisanZNodeGraph()
    g2.from_json(
        json_str,
        node_classes={"MultiOutputNode": MultiOutputNode, "PassthroughNode": PassthroughNode},
    )

    assert g2.additional_generation_data == {"foo": "bar"}
    assert g2.get_node_by_name("src").id == 0
    assert g2.get_node_by_name("sink").id == 1

    sink2 = g2.get_node_by_name("sink")
    src2 = g2.get_node_by_name("src")
    assert sink2.dependencies == [src2]
    assert sink2.connections["in_value"] == [(src2, "a")]
    assert g2.node_counter == 2


def test_update_from_json_rewires_connections_and_marks_updated():
    g = ImageArtisanZNodeGraph()

    src1 = SourceNode(value=1)
    src2 = SourceNode(value=2)
    sink = PassthroughNode()

    g.add_node(src1, name="src1")
    g.add_node(src2, name="src2")
    sink.connect("in_value", src1, "value")
    g.add_node(sink, name="sink")

    # Ensure baseline wiring is correct
    assert sink.dependencies == [src1]
    assert src1.dependents == [sink]
    assert src2.dependents == []

    # Create an updated JSON that rewires sink.in_value from src1 -> src2.
    payload = json.loads(g.to_json())
    assert payload["connections"], "Expected at least one connection"
    payload["connections"][0]["from_node_id"] = src2.id
    new_json = json.dumps(payload)

    g.update_from_json(
        new_json,
        node_classes={"SourceNode": SourceNode, "PassthroughNode": PassthroughNode},
    )

    assert sink.dependencies == [src2]
    assert dict(sink.connections) == {"in_value": [(src2, "value")]}
    assert sink in src2.dependents
    assert sink not in src1.dependents

    # Rewiring should invalidate sink at minimum.
    assert sink.updated is True


def test_delete_one_upstream_then_rerun_multiple_times_with_changing_output():
    calls: list[str] = []
    g = ImageArtisanZNodeGraph()

    a = SourceNode(value=100, calls=calls)
    b = ChangingSourceNode(start=0, step=1, calls=calls)
    c = CollectInputsNode(calls=calls)

    g.add_node(a, name="A")
    g.add_node(b, name="B")
    # Connect A then B so first run yields [A, B] deterministically.
    c.connect("in_value", a, "value")
    c.connect("in_value", b, "value")
    g.add_node(c, name="C")

    g()
    assert calls == ["A", "B", "C"]
    assert c.values["values"] == [100, 1]

    # Delete A (including its connections to C).
    g.delete_node_by_name("A")
    assert g.get_node_by_name("A") is None
    assert c.dependencies == [b]
    assert dict(c.connections) == {"in_value": [(b, "value")]}

    # Run twice; B should produce a new value each run, and C should see it.
    calls.clear()
    b.set_updated()  # marks B and dependent C as updated
    g()
    assert calls == ["B", "C"]
    assert c.values["values"] == [2]

    calls.clear()
    b.set_updated()
    g()
    assert calls == ["B", "C"]
    assert c.values["values"] == [3]


def test_rerun_multiple_times_with_changing_input_output():
    calls: list[str] = []
    g = ImageArtisanZNodeGraph()

    a = ManualSourceNode(calls=calls)
    b = AppendInputsNode(calls=calls)

    g.add_node(a, name="A")
    g.add_node(b, name="B")
    b.connect("in_value", a, "value")

    a.update_value(1)
    g()
    assert calls == ["A", "B"]
    assert b.values["values"] == [1]

    calls.clear()
    a.update_value(2)
    g()
    assert calls == ["A", "B"]
    assert b.values["values"] == [1, 2]


def test_multi_input_required_node():
    calls: list[str] = []
    g = ImageArtisanZNodeGraph()

    a = ManualSourceNode(calls=calls)
    b = ManualSourceNode(calls=calls)
    c = MultiInputNode(calls=calls)

    g.add_node(a, name="A")
    g.add_node(b, name="B")
    g.add_node(c, name="C")
    c.connect("in_a", a, "value")
    c.connect("in_b", b, "value")

    a.update_value(1)
    b.update_value(2)
    g()
    assert calls == ["A", "B", "C"]
    assert c.values["out_sum"] == 3
    assert c.dependencies == [a, b]

    g.delete_node_by_name("A")
    assert g.get_node_by_name("A") is None
    assert c.dependencies == [b]
    assert dict(c.connections) == {"in_b": [(b, "value")]}

    calls.clear()
    b.update_value(1)

    with pytest.raises(IArtisanZNodeError, match="The required input 'in_a' is not connected."):
        g()


def test_multi_input_optional_node():
    calls: list[str] = []
    g = ImageArtisanZNodeGraph()

    a = ManualSourceNode(calls=calls)
    b = ManualSourceNode(calls=calls)
    c = MultiInputOptionalNode(calls=calls)

    g.add_node(a, name="A")
    g.add_node(b, name="B")
    g.add_node(c, name="C")
    c.connect("in_a", a, "value")
    c.connect("in_b", b, "value")

    a.update_value(1)
    b.update_value(2)
    g()
    assert calls == ["A", "B", "C"]
    assert c.values["out_sum"] == 3
    assert c.dependencies == [a, b]

    g.delete_node_by_name("B")
    assert g.get_node_by_name("B") is None
    assert c.dependencies == [a]
    assert dict(c.connections) == {"in_a": [(a, "value")]}

    for value in [1, 5, 10, 15, 20]:
        calls.clear()
        a.update_value(value)
        g()
        assert calls == ["A", "C"]
        assert c.values["out_sum"] == value


def test_multi_input_delete_one_connection():
    calls: list[str] = []
    g = ImageArtisanZNodeGraph()

    a = MultiOutputNode(a=10, b=20, calls=calls)
    b = MultiInputOptionalNode(calls=calls)

    g.add_node(a, name="A")
    g.add_node(b, name="B")

    b.connect("in_a", a, "a")
    b.connect("in_b", a, "b")
    assert b.dependencies == [a]
    assert a.dependents == [b]

    g()
    assert calls == ["A", "B"]
    assert b.values["out_sum"] == 30

    calls.clear()
    b.disconnect("in_b", a, "b")
    assert b.dependencies == [a]
    assert a.dependents == [b]

    g()
    assert calls == ["B"]
    assert b.values["out_sum"] == 10

    calls.clear()
    a.update_values(a=5, b=15)
    g()
    assert calls == ["A", "B"]
    assert b.values["out_sum"] == 5
