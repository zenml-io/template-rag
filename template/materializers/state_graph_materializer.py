"""Materializer to handle CompiledStateGraph objects."""

from typing import Type

import dill
from langgraph.graph.state import CompiledStateGraph
from zenml.materializers.base_materializer import BaseMaterializer


class StateGraphMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (CompiledStateGraph,)

    file_name = "state_graph.pkl"

    def load(self, data_type: Type[CompiledStateGraph]) -> CompiledStateGraph:
        with open(self.uri + "/" + self.file_name, "rb") as f:
            loaded_data = dill.load(f)
        return loaded_data

    def save(self, data: CompiledStateGraph) -> None:
        serialized_data = dill.dumps(data)
        with open(self.uri + "/" + self.file_name, "wb") as f:
            f.write(serialized_data)
