from __future__ import annotations

import logging
from collections.abc import Iterable

from iartisanz.modules.generation.graph.nodes.node import Node


logger = logging.getLogger(__name__)


class ChoiceNode(Node):
    """A small "value node" that outputs a string constrained to a set of choices."""

    PRIORITY = 2
    OUTPUTS = ["value"]
    INPUTS = []

    def __init__(self, *, value: str | None = None, choices: Iterable[str] | None = None, default: str | None = None):
        super().__init__()
        self.choices = list(choices or [])
        self.default = default
        self.value = value

    def _normalize(self, value: object) -> str | None:
        if value is None:
            return None
        try:
            return str(value).strip().lower()
        except Exception:
            return None

    def _validated(self, value: object) -> str | None:
        normalized = self._normalize(value)
        if normalized is None:
            return None

        if self.choices:
            allowed = {self._normalize(c) for c in self.choices}
            if normalized not in allowed:
                logger.warning(
                    "ChoiceNode: value '%s' not in %s; falling back to default.",
                    normalized,
                    sorted(allowed),
                )
                return self._normalize(self.default)
        return normalized

    def update_value(self, value: str):
        self.value = self._validated(value)
        self.set_updated()

    def __call__(self):
        out = self._validated(self.value)
        if out is None:
            out = self._normalize(self.default)
        self.values["value"] = out
        return self.values
