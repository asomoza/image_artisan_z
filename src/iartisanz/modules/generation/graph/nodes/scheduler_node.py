from diffusers import FlowMatchEulerDiscreteScheduler

from iartisanz.modules.generation.data_objects.scheduler_data_object import SchedulerDataObject
from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.nodes.node import Node


class SchedulerNode(Node):
    OUTPUTS = ["scheduler"]

    SCHEDULER_MAPPING = {
        "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
    }

    def __init__(self, scheduler_data_object: SchedulerDataObject, **kwargs):
        super().__init__(**kwargs)

        self.scheduler_data_object = scheduler_data_object

    def update_value(self, scheduler_data_object: SchedulerDataObject):
        self.scheduler_data_object = scheduler_data_object
        self.set_updated()

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["scheduler_data_object"] = self.scheduler_data_object.to_dict()
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(SchedulerNode, cls).from_dict(node_dict)
        node.scheduler_data_object = node_dict["scheduler_data_object"]
        return node

    def update_inputs(self, node_dict):
        self.scheduler_data_object = SchedulerDataObject.from_dict(node_dict["scheduler_data_object"])

    def __call__(self):
        scheduler = self.load_scheduler(self.scheduler_data_object)
        self.values["scheduler"] = scheduler
        return self.values

    def load_scheduler(self, scheduler_data_object: SchedulerDataObject):
        scheduler_class_name = scheduler_data_object.scheduler_class
        scheduler_class = self.SCHEDULER_MAPPING.get(scheduler_class_name)

        if scheduler_class is None:
            raise IArtisanZNodeError(f"Unknown scheduler class: {scheduler_class_name}", self.__class__.__name__)

        scheduler_additional_configs = {}
        if scheduler_data_object.scheduler_index == 0:  # FlowMatchEulerDiscreteScheduler
            scheduler_additional_configs["shift"] = 3.0

        scheduler = scheduler_class.from_config(scheduler_data_object.to_config_dict(), **scheduler_additional_configs)

        return scheduler
