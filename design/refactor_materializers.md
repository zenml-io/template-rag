# Refactoring RAG Template Materializers

## Problem Statement

The current RAG template implementation uses custom materializers to serialize and deserialize vector stores and state graphs. This approach has several drawbacks:

1. **Performance**: Storing full vector embeddings and document content in the artifact store is inefficient
2. **Maintainability**: Using `dill` for serialization makes the code less maintainable and harder to debug
3. **Flexibility**: Current approach tightly couples data storage with pipeline metadata
4. **Scalability**: Not suitable for larger datasets or production deployments

## Proposed Solution

Replace custom materializers with a configuration-based approach that:
1. Stores only minimal configuration data in ZenML artifacts
2. Separates data storage from pipeline metadata
3. Uses standard formats (JSON) instead of pickle/dill
4. Makes the system more modular and maintainable

### Key Components

#### 1. Configuration Classes

```python
@dataclass
class VectorStoreConfig:
    """Configuration for vector store access."""
    store_type: str  # e.g. "faiss", "chroma", "pinecone"
    store_path: str  # local path or remote URI
    embedding_model: str
    metadata: Dict[str, Any] = None

@dataclass
class AssistantConfig:
    """Configuration for the RAG assistant."""
    vector_store: VectorStoreConfig
    llm_config: Dict[str, Any]
    prompt_templates: Dict[str, str]
```

#### 2. Updated Pipeline Steps

```python
@step
def ingest_documents(...) -> VectorStoreConfig
@step
def create_assistant(...) -> AssistantConfig
@step
def query_assistant(config: AssistantConfig, ...) -> str
@step
def evaluate_assistant(config: AssistantConfig, ...) -> Dict[str, float]
```

## Implementation Plan

### Phase 1: Setup and Cleanup
- [ ] Create `design/` directory
- [ ] Delete existing materializers:
  - `template/materializers/vector_store_materializer.py`
  - `template/materializers/state_graph_materializer.py`
- [ ] Remove materializer imports from all files
- [ ] Delete related tests:
  - `tests/test_materializers.py` (if exists)

### Phase 2: New Configuration Implementation
- [ ] Create new module `template/config/store.py`:
  - Implement `VectorStoreConfig`
  - Implement `AssistantConfig`
  - Add helper functions for config management
- [ ] Update `template/config/constants.py` with new constants
- [ ] Add configuration validation functions

### Phase 3: Pipeline Updates
- [ ] Update `template/steps/ingest_and_embed.py`:
  - Remove vector store materializer
  - Implement config-based storage
  - Update return type and docstrings
- [ ] Update `template/steps/create_assistant.py`:
  - Remove state graph materializer
  - Implement config-based assistant creation
  - Update return type and docstrings
- [ ] Update `template/steps/evaluate_assistant.py`:
  - Modify to use new config-based approach
  - Update input types and docstrings

### Phase 4: Testing Updates
- [ ] Create new test files:
  - `tests/test_config.py` for configuration classes
  - `tests/test_storage.py` for storage operations
- [ ] Update existing tests:
  - `tests/test_steps.py`
  - `tests/test_pipelines.py`
- [ ] Add integration tests for full pipeline

### Phase 5: Documentation Updates
- [ ] Update main README.md
- [ ] Update docstrings in all modified files
- [ ] Add configuration examples
- [ ] Update any example notebooks

## Migration Guide

For users of the existing template:

1. **Breaking Changes**:
   - Custom materializers removed
   - Pipeline step signatures changed
   - New configuration requirements

2. **Migration Steps**:
   - Update pipeline definitions
   - Convert existing vector stores
   - Update stack configurations

## Testing Strategy

1. **Unit Tests**:
   - Configuration validation
   - Storage operations
   - Individual step functionality

2. **Integration Tests**:
   - Full pipeline execution
   - Config persistence
   - Data retrieval

3. **Performance Tests**:
   - Memory usage comparison
   - Execution time benchmarks

## Success Criteria

1. **Functionality**:
   - All existing features work with new implementation
   - No data loss during migration
   - Backward compatibility where possible

2. **Performance**:
   - Reduced artifact size
   - Faster pipeline execution
   - Lower memory usage

3. **Code Quality**:
   - Simplified codebase
   - Better test coverage
   - Clear documentation

## Timeline

1. Phase 1: 1 day
2. Phase 2: 2 days
3. Phase 3: 2-3 days
4. Phase 4: 2 days
5. Phase 5: 1 day

Total: 8-9 days

## Risks and Mitigation

1. **Risk**: Data loss during migration
   - **Mitigation**: Comprehensive testing and backup procedures

2. **Risk**: Performance regression
   - **Mitigation**: Benchmark tests before/after

3. **Risk**: Breaking changes for users
   - **Mitigation**: Clear migration guide and examples

## Future Considerations

1. Support for additional vector store types
2. Cloud storage integration
3. Advanced caching strategies
4. Monitoring and observability improvements 
