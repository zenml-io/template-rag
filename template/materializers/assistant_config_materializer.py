"""Materializer for AssistantConfig objects."""

import json
from typing import Type, Any
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.enums import ArtifactType

from config.store import AssistantConfig

class AssistantConfigMaterializer(BaseMaterializer):
    """Materializer for AssistantConfig objects."""
    
    ASSOCIATED_TYPES = (AssistantConfig,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[Any]) -> AssistantConfig:
        """Load an AssistantConfig object from the artifact store.
        
        Args:
            data_type: The type of data to load
            
        Returns:
            The loaded AssistantConfig object
        """
        with self.artifact_store.open(self.uri, 'r') as f:
            data = json.load(f)
        return AssistantConfig.parse_obj(data)

    def save(self, config: AssistantConfig) -> None:
        """Save an AssistantConfig object to the artifact store.
        
        Args:
            config: The AssistantConfig object to save
        """
        with self.artifact_store.open(self.uri, 'w') as f:
            json.dump(config.dict(), f, indent=2)

    def extract_metadata(self, config: AssistantConfig) -> dict:
        """Extract metadata about the config.
        
        Args:
            config: The AssistantConfig object
            
        Returns:
            A dictionary containing metadata about the config
        """
        return {
            "vector_store_type": config.vector_store.store_type.value,
            "llm_model": config.llm.model,
            "temperature": config.llm.temperature,
            "num_prompt_templates": len(config.prompt_templates),
        } 

