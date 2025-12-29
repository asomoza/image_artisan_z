from __future__ import annotations

from typing import Union

from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.nodes.node import Node


Number = Union[int, float]


def _is_number(x: object) -> bool:
    # bool is a subclass of int; treat it as invalid here
    return isinstance(x, (int, float)) and not isinstance(x, bool)


class NumberRangeNode(Node):
    PRIORITY = 2
    OUTPUTS = ["value"]
    INPUTS: list[str] = []

    def __init__(self, value: list[Number] | None = None):
        super().__init__()
        if value is None:
            value = [0.0, 1.0]

        try:
            self.value: list[Number] = self._validate_pair(value)
        except Exception as e:
            raise IArtisanZNodeError(str(e), self.__class__.__name__)

    def update_value(self, value: object):
        try:
            self.value = self._validate_pair(value)
        except Exception as e:
            raise IArtisanZNodeError(str(e), self.__class__.__name__)

        self.set_updated()

    def __call__(self):
        self.values["value"] = self.value
        return self.values

    @staticmethod
    def _validate_pair(value: object) -> list[Number]:
        if not isinstance(value, list) or len(value) != 2:
            raise TypeError("NumberRangeNode expects a [start, end] list")

        start, end = value[0], value[1]

        if not _is_number(start) or not _is_number(end):
            raise TypeError("NumberRangeNode expects [int|float, int|float]")

        return [start, end]
