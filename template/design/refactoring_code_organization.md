# Refactoring Plan: Code Organization

## Problem Statement
The current codebase has several instances of code duplication and could benefit from better organization of shared components. This makes maintenance more difficult and increases the risk of inconsistencies.

## Proposed Changes

### 1. Create Common Types Module
Create a new module `types/` to store shared type definitions:
- Move `State` TypedDict to `types/state.py`
- Add type aliases for common types in `types/aliases.py`

### 2. Create Config Module
Create a new module `config/` to store configuration:
- Move LLM and embedding model configurations to `config/models.py`
- Add constants for chunk sizes and other magic numbers to `config/constants.py`

### 3. Create Utils Module
Create a new module `utils/` for shared utilities:
- Move vector store helper functions to `utils/vector_store.py`
- Move graph helper functions to `utils/graph.py`

### 4. Reorganize Materializers
- Move common serialization logic to `utils/serialization.py`
- Create base classes for common vector store operations

## Implementation Plan

### Phase 1: Setup New Directory Structure
```
.
├── config/
│   ├── __init__.py
│   ├── models.py
│   └── constants.py
├── types/
│   ├── __init__.py
│   ├── state.py
│   └── aliases.py
└── utils/
    ├── __init__.py
    ├── vector_store.py
    ├── graph.py
    └── serialization.py
```

### Phase 2: Move and Refactor Code
1. Create the new modules and move code
2. Update imports in existing files
3. Remove duplicated code
4. Add proper typing and documentation

### Phase 3: Testing
1. Ensure all existing functionality works
2. Add unit tests for new utility functions
3. Verify no regressions in pipeline execution

## Success Criteria
- Reduced code duplication
- Improved code organization
- All tests passing
- No changes to external API or behavior
- Improved maintainability and readability

## Risks and Mitigations
- Risk: Breaking existing functionality
  - Mitigation: Comprehensive testing after each change
- Risk: Circular dependencies
  - Mitigation: Careful planning of module structure

## Timeline
- Phase 1: 30 minutes
- Phase 2: 1 hour
- Phase 3: 30 minutes 
