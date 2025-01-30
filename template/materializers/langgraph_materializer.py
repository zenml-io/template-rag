"""Materializer for langgraph Graph objects."""

import os
import json
from typing import Type, Any, Tuple
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.enums import ArtifactType
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
import inspect
from materializers.graph_serializer import GraphSerializer

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if callable(obj):
                # Handle function objects by storing their string representation
                if inspect.isfunction(obj) or inspect.ismethod(obj):
                    return {
                        "__type__": "function",
                        "name": obj.__name__,
                        "module": obj.__module__,
                        "source": inspect.getsource(obj) if not obj.__name__ == '<lambda>' else str(obj)
                    }
                # Handle other callable objects (like classes)
                return {
                    "__type__": "callable",
                    "name": obj.__class__.__name__,
                    "module": obj.__class__.__module__,
                    "repr": str(obj)
                }
            elif hasattr(obj, '__dict__'):
                # Handle objects with __dict__
                return {
                    "__type__": "object",
                    "class": obj.__class__.__name__,
                    "module": obj.__class__.__module__,
                    "attributes": {
                        k: v for k, v in obj.__dict__.items()
                        if not k.startswith('_')  # Skip private attributes
                    }
                }
            elif hasattr(obj, '__slots__'):
                # Handle objects with __slots__
                return {
                    "__type__": "object",
                    "class": obj.__class__.__name__,
                    "module": obj.__class__.__module__,
                    "attributes": {
                        slot: getattr(obj, slot)
                        for slot in obj.__slots__
                        if not slot.startswith('_')  # Skip private attributes
                    }
                }
            # Let the base class handle the remaining types
            return super().default(obj)
        except Exception as e:
            # If all else fails, return a string representation
            return {
                "__type__": "unknown",
                "class": obj.__class__.__name__,
                "repr": str(obj),
                "error": str(e)
            }

class LangGraphMaterializer(BaseMaterializer):
    """Materializer for langgraph Graph objects."""
    
    ASSOCIATED_TYPES = (StateGraph, CompiledStateGraph)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL

    def load(self, data_type: Type[Any]) -> Any:
        """Load a Graph object from the artifact store.
        
        Args:
            data_type: The type of data to load (Graph)
            
        Returns:
            The deserialized Graph object
        """
        filepath = os.path.join(self.uri, 'graph.json')
        with self.artifact_store.open(filepath, 'r') as f:
            data = json.load(f)
        return GraphSerializer.deserialize_graph(data)

    def save(self, graph: CompiledStateGraph) -> None:
        """Save a Graph object to the artifact store.
        
        Args:
            graph: The Graph object to serialize
        """
        filepath = os.path.join(self.uri, 'graph.json')
        serialized = GraphSerializer.serialize_graph(graph)
        with self.artifact_store.open(filepath, 'w') as f:
            json.dump(serialized, f, indent=2, cls=CustomJSONEncoder)

    def extract_metadata(self, graph: CompiledStateGraph) -> dict:
        """Extract metadata about the graph for ZenML's UI/logs.
        
        Args:
            graph: The Graph object to extract metadata from
            
        Returns:
            A dictionary containing metadata about the graph
        """
        return {
            "num_nodes": len(graph.nodes),
            "num_edges": len(getattr(graph, 'edges', [])),
            "num_branches": sum(len(branches) for branches in getattr(graph, 'branches', {}).values()),
            "is_compiled": getattr(graph, 'compiled', False)
        } 

