from typing import Any, Optional

import attr


@attr.s(slots=True, auto_attribs=True)
class ModelItemDataObject:
    root_filename: str
    filepath: str
    name: str
    version: str
    model_type: int
    hash: str
    tags: Optional[str] = None
    thumbnail: Optional[str] = None
    triggers: Optional[str] = None
    example: Optional[str] = None
    deleted: int = 0
    id: Optional[int] = None
    distilled: int = 1

    def to_dict(self) -> dict[str, Any]:
        return attr.asdict(self)

    @classmethod
    def get_column_names(cls) -> list[str]:
        return [field.name for field in attr.fields(cls)]

    @classmethod
    def from_tuple(cls, data_tuple: tuple[Any, ...]) -> "ModelItemDataObject":
        column_names = cls.get_column_names()
        if len(data_tuple) < len(column_names):
            # Pad with defaults for columns added in later migrations
            defaults = {f.name: f.default for f in attr.fields(cls)}
            data_dict = dict(zip(column_names, data_tuple))
            for name in column_names[len(data_tuple):]:
                data_dict[name] = defaults.get(name)
        else:
            data_dict = dict(zip(column_names, data_tuple))
        return cls(**data_dict)

    def __attrs_post_init__(self):
        for field in attr.fields(self.__class__):
            if isinstance(field.type, str) and getattr(self, field.name) == "":
                setattr(self, field.name, None)
