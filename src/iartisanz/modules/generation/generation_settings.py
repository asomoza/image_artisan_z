from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from PyQt6.QtCore import QSettings

from iartisanz.modules.generation.data_objects.model_data_object import ModelDataObject
from iartisanz.modules.generation.data_objects.scheduler_data_object import SchedulerDataObject
from iartisanz.utils.json_utils import cast_model, cast_number_range, cast_scheduler


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _coerce_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _coerce_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _coerce_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


@dataclass(slots=True)
class GenerationSettings:
    right_menu_expanded: bool = True

    image_width: int = 1024
    image_height: int = 1024
    num_inference_steps: int = 24
    guidance_scale: float = 4.0
    guidance_start_end: list[float] = field(default_factory=lambda: [0.0, 1.0])
    scheduler: SchedulerDataObject = field(default_factory=SchedulerDataObject)
    strength: float = 0.5
    model: ModelDataObject = field(default_factory=ModelDataObject)
    use_torch_compile: bool = False

    GROUP: str = "generation"

    # Keys that should be forwarded to the graph on initialization / change
    GRAPH_KEYS: tuple[str, ...] = (
        "image_width",
        "image_height",
        "num_inference_steps",
        "guidance_scale",
        "guidance_start_end",
        "scheduler",
        "strength",
        "use_torch_compile",
    )

    @classmethod
    def load(cls, qsettings: QSettings) -> "GenerationSettings":
        settings = cls()
        qsettings.beginGroup(settings.GROUP)

        try:
            settings.right_menu_expanded = _coerce_bool(
                qsettings.value("right_menu_expanded", settings.right_menu_expanded),
                settings.right_menu_expanded,
            )

            settings.image_width = _coerce_int(
                qsettings.value("image_width", settings.image_width), settings.image_width
            )
            settings.image_height = _coerce_int(
                qsettings.value("image_height", settings.image_height), settings.image_height
            )
            settings.num_inference_steps = _coerce_int(
                qsettings.value("num_inference_steps", settings.num_inference_steps),
                settings.num_inference_steps,
            )

            settings.guidance_scale = _coerce_float(
                qsettings.value("guidance_scale", settings.guidance_scale), settings.guidance_scale
            )

            raw_range = _coerce_json(qsettings.value("guidance_start_end", settings.guidance_start_end))
            try:
                settings.guidance_start_end = cast_number_range(raw_range)
            except Exception:
                settings.guidance_start_end = [0.0, 1.0]

            raw_sched = _coerce_json(qsettings.value("scheduler", SchedulerDataObject().to_dict()))
            settings.scheduler = cast_scheduler(raw_sched)

            settings.strength = _coerce_float(qsettings.value("strength", settings.strength), settings.strength)

            raw_model = _coerce_json(qsettings.value("model", ModelDataObject().to_dict()))
            settings.model = cast_model(raw_model)

            settings.use_torch_compile = _coerce_bool(
                qsettings.value("use_torch_compile", settings.use_torch_compile),
                settings.use_torch_compile,
            )

            return settings
        finally:
            qsettings.endGroup()

    def save(self, qsettings: QSettings) -> None:
        qsettings.beginGroup(self.GROUP)
        try:
            qsettings.setValue("right_menu_expanded", bool(self.right_menu_expanded))
            qsettings.setValue("image_width", int(self.image_width))
            qsettings.setValue("image_height", int(self.image_height))
            qsettings.setValue("num_inference_steps", int(self.num_inference_steps))
            qsettings.setValue("guidance_scale", float(self.guidance_scale))
            qsettings.setValue("guidance_start_end", list(self.guidance_start_end))
            qsettings.setValue("scheduler", self.scheduler.to_dict())
            qsettings.setValue("strength", float(self.strength))
            qsettings.setValue("model", self.model.to_dict())
            qsettings.setValue("use_torch_compile", bool(self.use_torch_compile))
        finally:
            qsettings.endGroup()

    def to_graph_nodes(self) -> dict[str, Any]:
        # What NodeGraphThread expects
        return {
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "num_inference_steps": int(self.num_inference_steps),
            "guidance_scale": float(self.guidance_scale),
            "guidance_start_end": cast_number_range(self.guidance_start_end),
            "scheduler": cast_scheduler(self.scheduler),
            "use_torch_compile": bool(self.use_torch_compile),
        }

    def apply_change(self, attr: str, value: Any) -> tuple[bool, Any | None]:
        """
        Applies a change to this settings object (with casting).
        Returns:
          (handled, graph_value)
        Where graph_value is the casted value that should be forwarded to the graph,
        or None if this attr is not a graph key.
        """
        if not hasattr(self, attr):
            return (False, None)

        if attr == "right_menu_expanded":
            self.right_menu_expanded = _coerce_bool(value, self.right_menu_expanded)
            return (True, None)

        if attr == "image_width":
            self.image_width = _coerce_int(value, self.image_width)
            return (True, int(self.image_width))

        if attr == "image_height":
            self.image_height = _coerce_int(value, self.image_height)
            return (True, int(self.image_height))

        if attr == "num_inference_steps":
            self.num_inference_steps = _coerce_int(value, self.num_inference_steps)
            return (True, int(self.num_inference_steps))

        if attr == "guidance_scale":
            self.guidance_scale = _coerce_float(value, self.guidance_scale)
            return (True, float(self.guidance_scale))

        if attr == "guidance_start_end":
            try:
                coerced = _coerce_json(value)
                self.guidance_start_end = cast_number_range(coerced)
            except Exception:
                return (True, None)  # handled, but ignore invalid
            return (True, list(self.guidance_start_end))

        if attr == "scheduler":
            self.scheduler = cast_scheduler(value)
            return (True, self.scheduler)

        if attr == "strength":
            self.strength = _coerce_float(value, self.strength)
            return (True, self.strength)

        if attr == "model":
            self.model = cast_model(value)
            return (True, None)

        if attr == "use_torch_compile":
            self.use_torch_compile = _coerce_bool(value, self.use_torch_compile)
            return (True, bool(self.use_torch_compile))

        # Fallback: handled but not forwarded
        setattr(self, attr, value)
        return (True, None)

    def reset_to_defaults(self, *, preserve_model: bool = True) -> None:
        """Reset settings to dataclass defaults.

        If preserve_model is True, keeps the currently selected model.
        """

        current_model = self.model
        current_right_menu_expanded = self.right_menu_expanded

        defaults = type(self)()
        self.right_menu_expanded = current_right_menu_expanded

        self.image_width = defaults.image_width
        self.image_height = defaults.image_height
        self.num_inference_steps = defaults.num_inference_steps
        self.guidance_scale = defaults.guidance_scale
        self.guidance_start_end = list(defaults.guidance_start_end)
        self.scheduler = defaults.scheduler
        self.strength = defaults.strength
        self.use_torch_compile = defaults.use_torch_compile

        if preserve_model:
            self.model = current_model
        else:
            self.model = defaults.model
