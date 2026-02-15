from __future__ import annotations

import gc
import json
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING

import torch

from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError


if TYPE_CHECKING:
    from iartisanz.modules.generation.graph.nodes.node import Node


class ImageArtisanZNodeGraph:
    def __init__(self):
        self.node_counter = 0
        self.nodes = []
        self.updated = False
        self.abort_function = lambda: None
        self.executing_node = None

        self.device = None
        self.dtype = None

        self.additional_generation_data = {}

    def add_node(self, node: Node, name: str = None):
        if name is not None:
            for existing_node in self.nodes:
                if existing_node.name == name:
                    raise IArtisanZNodeError(
                        f"A node with the name {name} already exists in the graph.", self.__class__.__name__
                    )
        node.name = name
        node.id = self.node_counter
        node.abort_callable = self.abort_function
        node.updated = True
        self.nodes.append(node)
        self.node_counter += 1

    def get_node(self, node_id):
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_node_by_name(self, name: str):
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_all_nodes_class(self, node_class):
        return [node for node in self.nodes if isinstance(node, node_class)]

    def delete_node_by_id(self, node_id):
        node = self.get_node(node_id)
        if node is not None:
            self.delete_node(node)

    def delete_node_by_name(self, name: str):
        node = self.get_node_by_name(name)
        if node is not None:
            self.delete_node(node)

    def delete_node(self, node: Node):
        node.before_delete()

        # Make copies to avoid mutation during iteration
        related = list(set(node.dependencies + node.dependents))
        for other_node in related:
            other_node.disconnect_from_node(node)
            node.disconnect_from_node(other_node)

        node.delete()
        self.nodes.remove(node)
        del node

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    @torch.no_grad()
    def __call__(self):
        # Ensure model placement decisions are made centrally.
        from iartisanz.app.model_manager import get_model_manager

        self.updated = False
        sorted_nodes = deque()
        visited = set()
        visiting = set()

        def dfs(node):
            visiting.add(node)

            for dependency in sorted(node.dependencies, key=lambda x: x.PRIORITY, reverse=True):
                if dependency in visiting:
                    raise ValueError("Graph contains a cycle")
                if dependency not in visited:
                    dfs(dependency)

            visiting.remove(node)
            visited.add(node)
            sorted_nodes.append(node)

        for node in sorted(self.nodes, key=lambda x: x.PRIORITY, reverse=True):
            if node not in visited:
                dfs(node)

        mm = get_model_manager()
        with mm.device_scope(device=self.device, dtype=self.dtype):
            for node in sorted_nodes:
                if node.updated and node.enabled:
                    node.device = self.device
                    node.dtype = self.dtype
                    start_time = time.time()

                    try:
                        self.executing_node = node
                        node()
                    except IArtisanZNodeError:
                        raise

                    end_time = time.time()
                    node.elapsed_time = end_time - start_time
                    self.updated = True
                    self.executing_node = None

                    if node.abort:
                        node.abort = False
                        self.abort_function()
                        break

                    node.updated = False

    def to_json(self, additional_generation_data: dict | None = None):
        if additional_generation_data is None:
            additional_generation_data = self.additional_generation_data or {}

        graph_dict = {
            "format_version": 1,
            "nodes": [node.to_dict() for node in self.nodes],
            "connections": [],
            "additional_generation_data": additional_generation_data,
        }

        for node in self.nodes:
            for input_name, connections in node.connections.items():
                for connected_node, output_name in connections:
                    graph_dict["connections"].append(
                        {
                            "from_node_id": connected_node.id,
                            "from_output_name": output_name,
                            "to_node_id": node.id,
                            "to_input_name": input_name,
                        }
                    )

        return json.dumps(graph_dict)

    def from_json(self, json_str, node_classes, callbacks=None):
        graph_dict = json.loads(json_str)

        self.nodes.clear()
        self.node_counter = 0

        self.additional_generation_data = graph_dict.get("additional_generation_data", {}) or {}

        id_to_node = {}
        max_id = 0

        for node_dict in graph_dict["nodes"]:
            NodeClass = node_classes[node_dict["class"]]
            node = NodeClass.from_dict(node_dict, callbacks)
            id_to_node[node.id] = node
            self.nodes.append(node)
            max_id = max(max_id, node.id)

        for connection_dict in graph_dict["connections"]:
            from_node = id_to_node[connection_dict["from_node_id"]]
            to_node = id_to_node[connection_dict["to_node_id"]]
            to_node.connect(
                connection_dict["to_input_name"],
                from_node,
                connection_dict["from_output_name"],
            )

        self.node_counter = max_id + 1

    def save_to_json(self, filename, additional_generation_data: dict | None = None):
        json_str = self.to_json(additional_generation_data=additional_generation_data)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(json_str)

    def load_from_json(self, filename, node_classes, callbacks=None):
        with open(filename, "r", encoding="utf-8") as f:
            json_str = f.read()

        self.from_json(json_str, node_classes, callbacks)

    def update_from_json(self, json_str, node_classes, callbacks=None):
        graph_dict = json.loads(json_str)
        current_id_to_node = {node.id: node for node in self.nodes}
        updated_nodes = set()

        self.additional_generation_data = (
            graph_dict.get("additional_generation_data", self.additional_generation_data) or {}
        )

        new_id_to_node = {}
        max_id = 0

        for node_dict in graph_dict["nodes"]:
            node_class = node_classes[node_dict["class"]]

            if node_dict["id"] in current_id_to_node and isinstance(current_id_to_node[node_dict["id"]], node_class):
                node = current_id_to_node[node_dict["id"]]
                new_node = node_class.from_dict(node_dict, callbacks)

                if node.to_dict() != new_node.to_dict():
                    node.update_inputs(node_dict, callbacks=callbacks)
                    node.set_updated(updated_nodes)
            else:
                node = node_class.from_dict(node_dict, callbacks)
                node.set_updated(updated_nodes)
                if node_dict["id"] in current_id_to_node:
                    self.delete_node_by_id(node_dict["id"])
                self.nodes.append(node)

            new_id_to_node[node.id] = node
            max_id = max(max_id, node.id)

        for node_id in list(current_id_to_node.keys()):
            if node_id not in new_id_to_node:
                self.delete_node_by_id(node_id)

        # Node deletions disconnect affected nodes and mark them updated, but
        # those updates use a local set inside set_updated() and are NOT tracked
        # in our outer updated_nodes.  Capture them now so the final reset loop
        # doesn't discard them (fixes stale outputs when edit images are removed).
        for node in self.nodes:
            if node.updated and node.id not in updated_nodes:
                node.set_updated(updated_nodes)

        # Include input name so we detect re-wires between inputs
        new_connections = defaultdict(list)  # to_node_id -> [(to_input_name, from_node_id, from_output_name)]
        for connection_dict in graph_dict["connections"]:
            to_id = connection_dict["to_node_id"]
            new_connections[to_id].append(
                (
                    connection_dict["to_input_name"],
                    connection_dict["from_node_id"],
                    connection_dict["from_output_name"],
                )
            )

        for node in self.nodes:
            desired = new_connections[node.id]
            if node.connections_changed(desired):
                # Rewiring should invalidate this node (and its dependents) for the next execution.
                node.set_updated(updated_nodes)
                # Fully clear old wiring (including reverse links)
                node.clear_all_connections()

                for to_input_name, from_node_id, output_name in desired:
                    from_node = new_id_to_node[from_node_id]
                    node.connect(to_input_name, from_node, output_name)

        for node in self.nodes:
            if node.id not in updated_nodes:
                node.updated = False

        self.node_counter = max_id + 1

    def update_from_json_file(self, filename, node_classes, callbacks=None):
        with open(filename, "r", encoding="utf-8") as f:
            json_str = f.read()
        self.update_from_json(json_str, node_classes, callbacks)

    def abort_graph(self):
        if self.executing_node is not None:
            self.executing_node.abort = True

    def set_abort_function(self, abort_callable: callable):
        self.abort_function = abort_callable

    def clean_up(self):
        for i in range(len(self.nodes) - 1, -1, -1):
            self.delete_node(self.nodes[i])

        self.node_counter = 0

    def validate_controlnet_inpainting(self):
        """Validate ControlNet inpainting configuration.

        Ensures that:
        - Mask requires ControlNet and either control_image (spatial mode) or init_image (inpainting)
        - Init_image requires ControlNet
        - ControlNet requires either control_image or inpainting components
        - Invalid configurations raise ValueError

        Raises:
            ValueError: If the configuration is invalid
        """
        # Check for controlnet inpainting nodes
        controlnet_node = self.get_node_by_name("controlnet_conditioning")
        has_controlnet = controlnet_node is not None
        has_control_image = self.get_node_by_name("control_image") is not None
        has_mask = self.get_node_by_name("control_mask_image") is not None

        # Check for init_image in two ways (for backward compatibility):
        # 1. NEW: init_image comes from source_image connection (refactored approach)
        # 2. OLD: separate node named "control_init_image" (legacy approach)
        has_init_image = False
        if has_controlnet:
            # Check if init_image input has a connection (refactored approach)
            has_init_image = "init_image" in controlnet_node.connections and len(controlnet_node.connections["init_image"]) > 0

        if not has_init_image:
            # Fallback: check for legacy "control_init_image" node (backward compatibility)
            has_init_image = self.get_node_by_name("control_init_image") is not None

        # Validate combinations
        # Mask without init_image is valid if control_image is present (Spatial ControlNet mode)
        if has_mask and not has_init_image and not has_control_image:
            raise ValueError(
                "ControlNet mask requires either a control image or init_image to be configured. "
                "Please add a control image or source image."
            )

        if has_mask and not has_controlnet:
            raise ValueError(
                "ControlNet mask requires a ControlNet to be configured. "
                "Please add a control image first."
            )

        if has_init_image and not has_controlnet:
            raise ValueError(
                "ControlNet init_image requires a ControlNet to be configured. "
                "Please add a control image first."
            )

        # If controlnet is configured, it must have either control_image or inpainting
        if has_controlnet and not has_control_image and not has_init_image and not has_mask:
            raise ValueError(
                "ControlNet requires either a control image or inpainting to be configured. "
                "Please add a control image or set up inpainting."
            )
