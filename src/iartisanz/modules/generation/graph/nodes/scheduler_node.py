from iartisanz.modules.generation.constants import SCHEDULER_CLASS_MAPPING
from iartisanz.modules.generation.data_objects.scheduler_data_object import SchedulerDataObject
from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.nodes.node import Node


class SchedulerNode(Node):
    OUTPUTS = ["scheduler"]

    SERIALIZE_INCLUDE = {"scheduler_data_object"}
    SERIALIZE_CONVERTERS = {
        "scheduler_data_object": (
            lambda obj: None if obj is None else obj.to_dict(),
            lambda data: None if data is None else SchedulerDataObject.from_dict(data),
        )
    }

    def __init__(self, scheduler_data_object: SchedulerDataObject | None = None):
        super().__init__()
        self.scheduler_data_object = scheduler_data_object

    def update_value(self, scheduler_data_object: SchedulerDataObject):
        self.scheduler_data_object = scheduler_data_object
        self.set_updated()

    def __call__(self):
        scheduler = self.load_scheduler(self.scheduler_data_object)
        self.values["scheduler"] = scheduler
        return self.values

    def load_scheduler(self, scheduler_data_object: SchedulerDataObject):
        if scheduler_data_object is None:
            raise IArtisanZNodeError("scheduler_data_object is None", self.__class__.__name__)

        scheduler_class_name = scheduler_data_object.scheduler_class
        scheduler_class = SCHEDULER_CLASS_MAPPING.get(scheduler_class_name)

        if scheduler_class is None:
            raise IArtisanZNodeError(f"Unknown scheduler class: {scheduler_class_name}", self.__class__.__name__)

        scheduler_additional_configs = {}
        scheduler_additional_configs["shift"] = self.scheduler_data_object.shift

        scheduler = scheduler_class.from_config(scheduler_data_object.to_config_dict(), **scheduler_additional_configs)
        return scheduler
