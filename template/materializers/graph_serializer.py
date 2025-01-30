import json
from typing import Any, Dict, List, Tuple, Union, Callable, Optional
from inspect import getsource, isclass, isfunction, ismethod
import types
import importlib
from dataclasses import is_dataclass, asdict
import logging
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.base import RunnableLike
from langchain_core.runnables.utils import Input, Output
from pydantic import BaseModel
from langgraph.pregel import PregelNode
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from typing import TypedDict, NamedTuple

logger = logging.getLogger(__name__)

class SerializationError(Exception):
    """Custom exception for serialization errors"""
    pass

class DeserializationError(Exception):
    """Custom exception for deserialization errors"""
    pass

class DefaultState(TypedDict):
    """Default state schema for graph deserialization."""
    question: str
    context: List[Dict[str, Any]]
    answer: str

class NodeSpec(NamedTuple):
    """Node specification for graph nodes."""
    runnable: Any
    metadata: Dict[str, Any]
    ends: Optional[Tuple[str, ...]] = None

class GraphSerializer:
    """
    Enhanced serializer/deserializer for langgraph Graph class with comprehensive Runnable support.
    Handles various types of Runnables including functions, classes, chains, and custom objects.
    """
    
    @staticmethod
    def _serialize_config(config: Any) -> Dict[str, Any]:
        """Helper method to serialize configuration objects"""
        if isinstance(config, dict):
            return {k: GraphSerializer._serialize_config(v) for k, v in config.items()}
        elif isinstance(config, (list, tuple)):
            return [GraphSerializer._serialize_config(v) for v in config]
        elif is_dataclass(config):
            return {
                "__dataclass__": config.__class__.__name__,
                "__module__": config.__class__.__module__,
                "data": asdict(config)
            }
        elif isinstance(config, BaseModel):
            return {
                "__pydantic__": config.__class__.__name__,
                "__module__": config.__class__.__module__,
                "data": config.dict()
            }
        elif isinstance(config, (str, int, float, bool, type(None))):
            return config
        else:
            raise SerializationError(f"Unsupported config type: {type(config)}")

    @staticmethod
    def _deserialize_config(config: Any) -> Any:
        """Helper method to deserialize configuration objects"""
        if isinstance(config, dict):
            if "__dataclass__" in config:
                try:
                    module = importlib.import_module(config["__module__"])
                    cls = getattr(module, config["__dataclass__"])
                    return cls(**config["data"])
                except (ImportError, AttributeError) as e:
                    raise DeserializationError(f"Failed to deserialize dataclass: {e}")
            elif "__pydantic__" in config:
                try:
                    module = importlib.import_module(config["__module__"])
                    cls = getattr(module, config["__pydantic__"])
                    return cls(**config["data"])
                except (ImportError, AttributeError) as e:
                    raise DeserializationError(f"Failed to deserialize Pydantic model: {e}")
            else:
                return {k: GraphSerializer._deserialize_config(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [GraphSerializer._deserialize_config(v) for v in config]
        return config

    @staticmethod
    def _get_class_init_params(cls: type) -> Dict[str, Any]:
        """Extract initialization parameters from a class"""
        if hasattr(cls, '__init__'):
            try:
                init_source = getsource(cls.__init__)
                return {
                    'source': init_source,
                    'module': cls.__module__,
                    'name': cls.__name__
                }
            except (IOError, TypeError):
                return {}
        return {}

    @classmethod
    def serialize_runnable(cls, runnable: RunnableLike) -> Dict[str, Any]:
        """
        Enhanced serialization for various types of Runnables.
        """
        try:
            if runnable is None:
                logger.warning("Attempting to serialize None runnable")
                return {}

            # Get basic information about the runnable
            module_name = getattr(runnable, '__module__', None)
            class_name = runnable.__class__.__name__
            func_name = getattr(runnable, '__name__', None)

            logger.debug("Serializing runnable: module=%s, class=%s, func=%s",
                        module_name, class_name, func_name)

            # Handle different types of runnables
            if isinstance(runnable, types.FunctionType):
                return {
                    'type': 'function',
                    'module': module_name,
                    'name': func_name
                }

            elif isinstance(runnable, (list, tuple)):
                # Handle chain-like sequences of runnables
                return {
                    'type': 'chain',
                    'steps': [cls.serialize_runnable(step) for step in runnable]
                }

            elif hasattr(runnable, 'model_name'):
                # Handle LLM runnables
                return {
                    'type': 'llm',
                    'model': getattr(runnable, 'model_name', None)
                }

            elif hasattr(runnable, 'template'):
                # Handle prompt template runnables
                return {
                    'type': 'prompt',
                    'template': getattr(runnable, 'template', None)
                }

            elif hasattr(runnable, 'vectorstore') and hasattr(runnable, 'docstore'):
                # Handle retriever runnables
                config = {}
                if hasattr(runnable, 'child_splitter'):
                    config['chunk_size'] = getattr(runnable.child_splitter, 'chunk_size', 1000)
                    config['chunk_overlap'] = getattr(runnable.child_splitter, 'chunk_overlap', 200)
                return {
                    'type': 'retriever',
                    'config': config
                }

            elif isinstance(runnable, Runnable):
                # Handle generic Runnable instances
                return {
                    'type': 'runnable',
                    'module': module_name,
                    'class_name': class_name,
                    'config': cls._serialize_config(getattr(runnable, 'config', {}))
                }

            else:
                # Handle unknown runnables
                logger.warning(f"Unknown runnable type: {class_name}")
                return {
                    'type': 'unknown',
                    'module': module_name,
                    'class_name': class_name
                }

        except Exception as e:
            logger.error(f"Failed to serialize runnable: {e}")
            return {}

    @classmethod
    def deserialize_runnable(cls, data: Dict[str, Any]) -> Any:
        """Deserialize a runnable object with enhanced error handling."""
        try:
            if not data:
                logger.warning("Empty runnable data provided")
                return None

            runnable_type = data.get('type')
            if not runnable_type:
                logger.warning("No runnable type specified in data")
                return None

            logger.debug("Deserializing runnable of type: %s", runnable_type)

            if runnable_type == 'function':
                # Handle function runnables
                module_name = data.get('module')
                func_name = data.get('name')
                
                if not module_name or not func_name:
                    logger.warning("Missing module or function name for function runnable")
                    return None
                
                try:
                    module = importlib.import_module(module_name)
                    func = getattr(module, func_name)
                    logger.info(f"Successfully loaded function {func_name} from {module_name}")
                    return func
                except Exception as e:
                    logger.warning(f"Failed to load function {func_name} from {module_name}: {e}")
                    return None

            elif runnable_type == 'chain':
                # Handle chain runnables
                steps = data.get('steps', [])
                if not steps:
                    logger.warning("Empty steps list for chain runnable")
                    return None
                
                deserialized_steps = []
                for step in steps:
                    deserialized_step = cls.deserialize_runnable(step)
                    if deserialized_step is not None:
                        deserialized_steps.append(deserialized_step)
                
                if not deserialized_steps:
                    logger.warning("No valid steps found in chain")
                    return None
                
                def chain_runnable(input_data):
                    result = input_data
                    for step in deserialized_steps:
                        try:
                            result = step(result)
                        except Exception as e:
                            logger.error(f"Error in chain step: {e}")
                            raise
                    return result
                
                return chain_runnable

            elif runnable_type == 'llm':
                # Handle LLM runnables
                model_name = data.get('model')
                if not model_name:
                    logger.warning("No model name specified for LLM runnable")
                    return None
                
                try:
                    from langchain_openai import ChatOpenAI
                    llm = ChatOpenAI(model_name=model_name)
                    logger.info(f"Successfully created LLM with model {model_name}")
                    return llm
                except Exception as e:
                    logger.warning(f"Failed to create LLM with model {model_name}: {e}")
                    return None

            elif runnable_type == 'prompt':
                # Handle prompt template runnables
                template = data.get('template')
                if not template:
                    logger.warning("No template specified for prompt runnable")
                    return None
                
                try:
                    from langchain.prompts import PromptTemplate
                    prompt = PromptTemplate.from_template(template)
                    logger.info("Successfully created prompt template")
                    return prompt
                except Exception as e:
                    logger.warning(f"Failed to create prompt template: {e}")
                    return None

            elif runnable_type == 'retriever':
                # Handle retriever runnables
                retriever_config = data.get('config', {})
                if not retriever_config:
                    logger.warning("No config specified for retriever runnable")
                    return None
                
                try:
                    from langchain.retrievers import ParentDocumentRetriever
                    from langchain.text_splitter import RecursiveCharacterTextSplitter
                    from langchain.storage import InMemoryStore
                    from langchain.embeddings import OpenAIEmbeddings
                    from langchain.vectorstores import FAISS
                    
                    # Create text splitter
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=retriever_config.get('chunk_size', 1000),
                        chunk_overlap=retriever_config.get('chunk_overlap', 200),
                    )
                    
                    # Create stores
                    docstore = InMemoryStore()
                    vectorstore = FAISS.from_texts(
                        texts=[""], embedding=OpenAIEmbeddings()
                    )
                    
                    # Create retriever
                    retriever = ParentDocumentRetriever(
                        vectorstore=vectorstore,
                        docstore=docstore,
                        child_splitter=text_splitter,
                        parent_splitter=text_splitter,
                    )
                    
                    logger.info("Successfully created retriever")
                    return retriever
                except Exception as e:
                    logger.warning(f"Failed to create retriever: {e}")
                    return None

            elif runnable_type == 'runnable':
                # Handle generic Runnable instances
                module_name = data.get('module')
                class_name = data.get('class_name')
                config = data.get('config', {})
                
                if not module_name or not class_name:
                    logger.warning("Missing module or class name for runnable")
                    return None
                
                try:
                    module = importlib.import_module(module_name)
                    cls_obj = getattr(module, class_name)
                    instance = cls_obj()
                    
                    # Try to restore config if possible
                    if hasattr(instance, '_deserialize_config'):
                        deserialized_config = instance._deserialize_config(config)
                        if deserialized_config:
                            instance.config = deserialized_config
                    
                    logger.info(f"Successfully created runnable instance of {class_name}")
                    return instance
                except Exception as e:
                    logger.warning(f"Failed to create runnable instance: {e}")
                    return None

            elif runnable_type == 'unknown':
                # Handle unknown runnables by creating a pass-through function
                logger.warning("Creating pass-through function for unknown runnable type")
                return lambda x: x

            else:
                logger.warning(f"Unknown runnable type: {runnable_type}")
                return None

        except Exception as e:
            logger.error(f"Failed to deserialize runnable: {e}")
            return None

    @classmethod
    def serialize_node_spec(cls, node_spec: Any) -> Dict[str, Any]:
        """Serialize a NodeSpec object with enhanced error handling."""
        try:
            if isinstance(node_spec, PregelNode):
                # Handle PregelNode objects from compiled graph
                node_data = {
                    'type': 'pregel_node',
                    'class_name': node_spec.__class__.__name__,
                    'module': node_spec.__class__.__module__,
                }

                # Handle input and output types
                if hasattr(node_spec, 'input') and node_spec.input is not None:
                    node_data['input'] = {
                        'type': 'type',
                        'class_name': node_spec.input.__name__,
                        'module': node_spec.input.__module__,
                    }
                
                if hasattr(node_spec, 'output') and node_spec.output is not None:
                    node_data['output'] = {
                        'type': 'type',
                        'class_name': node_spec.output.__name__,
                        'module': node_spec.output.__module__,
                    }

                # Handle channels (can be list or dict)
                if hasattr(node_spec, 'channels'):
                    if isinstance(node_spec.channels, (list, tuple)):
                        node_data['channels'] = list(node_spec.channels)
                    else:
                        node_data['channels'] = dict(node_spec.channels)

                # Handle triggers
                if hasattr(node_spec, 'triggers'):
                    node_data['triggers'] = list(node_spec.triggers)

                # Handle bound runnable
                if hasattr(node_spec, 'bound') and node_spec.bound is not None:
                    try:
                        node_data['bound'] = cls.serialize_runnable(node_spec.bound)
                    except Exception as e:
                        logger.warning(f"Could not serialize bound runnable: {e}")
                        node_data['bound'] = str(node_spec.bound)

                # Handle writers
                if hasattr(node_spec, 'writers'):
                    try:
                        writers_data = []
                        for writer in node_spec.writers:
                            if isinstance(writer, (types.FunctionType, types.MethodType)):
                                writers_data.append({
                                    'type': 'function',
                                    'source': getsource(writer),
                                    'name': writer.__name__,
                                    'module': writer.__module__,
                                    'is_lambda': writer.__name__ == '<lambda>'
                                })
                            else:
                                writers_data.append(cls.serialize_runnable(writer))
                        node_data['writers'] = writers_data
                    except Exception as e:
                        logger.warning(f"Could not serialize writers: {e}")
                        node_data['writers'] = [str(w) for w in node_spec.writers]

                # Handle metadata
                if hasattr(node_spec, 'metadata') and node_spec.metadata:
                    try:
                        node_data['metadata'] = cls._serialize_config(node_spec.metadata)
                    except Exception as e:
                        logger.warning(f"Could not serialize metadata: {e}")
                        node_data['metadata'] = str(node_spec.metadata)

                # Handle tags
                if hasattr(node_spec, 'tags') and node_spec.tags:
                    node_data['tags'] = list(node_spec.tags)

                return node_data
            else:
                # Handle regular NodeSpec objects
                return {
                    'type': 'node_spec',
                    'runnable': cls.serialize_runnable(node_spec.runnable),
                    'metadata': cls._serialize_config(node_spec.metadata),
                    'ends': node_spec.ends
                }
        except Exception as e:
            raise SerializationError(f"Failed to serialize NodeSpec: {str(e)}")

    @classmethod
    def deserialize_node_spec(cls, data: Dict[str, Any]) -> Any:
        """Deserialize a NodeSpec object with enhanced error handling."""
        try:
            if data.get('type') == 'pregel_node':
                logger.debug("Deserializing PregelNode with data: %s", data)
                # Import PregelNode and related classes
                from langgraph.pregel.read import PregelNode
                from langgraph.pregel.write import ChannelWrite, ChannelWriteEntry
                
                # Create a new PregelNode with the basic attributes
                triggers = data.get('triggers', [])
                channels = data.get('channels', [])
                tags = data.get('tags', [])
                logger.debug("Creating PregelNode with triggers=%s, channels=%s, tags=%s",
                           triggers, channels, tags)
                
                node = PregelNode(
                    triggers=triggers,
                    channels=channels,
                    tags=tags,
                )
                
                # Set input and output types to match state schema
                if 'input' in data:
                    try:
                        logger.debug("Setting input type from data: %s", data['input'])
                        module = importlib.import_module(data['input']['module'])
                        input_type = getattr(module, data['input']['class_name'])
                        logger.info(f"Setting PregelNode input type to: {input_type}")
                        node.input = input_type
                    except Exception as e:
                        logger.warning(f"Could not set input type: {e}", exc_info=True)
                        node.input = DefaultState
                else:
                    node.input = DefaultState
                
                if 'output' in data:
                    try:
                        logger.debug("Setting output type from data: %s", data['output'])
                        module = importlib.import_module(data['output']['module'])
                        output_type = getattr(module, data['output']['class_name'])
                        logger.info(f"Setting PregelNode output type to: {output_type}")
                        node.output = output_type
                    except Exception as e:
                        logger.warning(f"Could not set output type: {e}", exc_info=True)
                        node.output = DefaultState
                else:
                    node.output = DefaultState
                
                # Set ends attribute to None to match NodeSpec interface
                node.ends = None
                
                # Restore bound runnable if present
                if 'bound' in data:
                    try:
                        logger.debug("Deserializing bound runnable from data: %s", data['bound'])
                        bound = cls.deserialize_runnable(data['bound'])
                        if bound is not None:
                            node.bound = bound
                            node.runnable = bound
                            logger.info("Successfully restored bound runnable")
                        else:
                            logger.warning("Failed to deserialize bound runnable, using pass-through")
                            node.bound = lambda x: x
                            node.runnable = lambda x: x
                    except Exception as e:
                        logger.warning(f"Could not deserialize bound runnable: {e}", exc_info=True)
                        node.bound = lambda x: x
                        node.runnable = lambda x: x
                elif 'runnable' in data:
                    try:
                        logger.debug("Deserializing runnable from data: %s", data['runnable'])
                        runnable = cls.deserialize_runnable(data['runnable'])
                        if runnable is not None:
                            node.runnable = runnable
                            logger.info("Successfully restored runnable")
                        else:
                            logger.warning("Failed to deserialize runnable, using pass-through")
                            node.runnable = lambda x: x
                    except Exception as e:
                        logger.warning(f"Could not deserialize runnable: {e}", exc_info=True)
                        node.runnable = lambda x: x
                else:
                    logger.warning("No runnable or bound data found, using pass-through")
                    node.runnable = lambda x: x
                
                # Restore writers if present
                if 'writers' in data:
                    try:
                        logger.debug("Deserializing writers from data: %s", data['writers'])
                        writers = []
                        for writer_data in data['writers']:
                            writer = cls.deserialize_runnable(writer_data)
                            if writer is not None:
                                writers.append(writer)
                                logger.debug("Successfully restored writer")
                        if writers:
                            node.writers = writers
                            logger.info(f"Successfully restored {len(writers)} writers")
                        else:
                            logger.warning("No valid writers found, using default writer")
                            def default_writer(state, output):
                                if isinstance(output, dict):
                                    state.update(output)
                                elif isinstance(output, str):
                                    state['answer'] = output
                                return state
                            node.writers = [default_writer]
                    except Exception as e:
                        logger.warning(f"Could not deserialize writers: {e}", exc_info=True)
                        def default_writer(state, output):
                            if isinstance(output, dict):
                                state.update(output)
                            elif isinstance(output, str):
                                state['answer'] = output
                            return state
                        node.writers = [default_writer]
                else:
                    logger.warning("No writers found, using default writer")
                    def default_writer(state, output):
                        if isinstance(output, dict):
                            state.update(output)
                        elif isinstance(output, str):
                            state['answer'] = output
                        return state
                    node.writers = [default_writer]
                
                # Restore metadata if present
                if 'metadata' in data:
                    node.metadata = cls._deserialize_config(data['metadata'])
                
                return node
            else:
                # For regular nodes, create a NodeSpec
                runnable = cls.deserialize_runnable(data.get('runnable', {}))
                metadata = cls._deserialize_config(data.get('metadata', {}))
                ends = tuple(data['ends']) if data.get('ends') is not None else None
                
                # If runnable is None, create a pass-through function
                if runnable is None:
                    logger.warning("No valid runnable found for NodeSpec, using pass-through")
                    runnable = lambda x: x
                
                return NodeSpec(runnable=runnable, metadata=metadata, ends=ends)
        except Exception as e:
            logger.error(f"Failed to deserialize NodeSpec: {e}", exc_info=True)
            raise DeserializationError(f"Failed to deserialize NodeSpec: {str(e)}")

    @classmethod
    def serialize_branch(cls, branch: 'Branch') -> Dict[str, Any]:
        """Serialize a Branch object with enhanced error handling."""
        try:
            return {
                'path': cls.serialize_runnable(branch.path),
                'ends': branch.ends,
                'then': branch.then
            }
        except Exception as e:
            raise SerializationError(f"Failed to serialize Branch: {str(e)}")

    @classmethod
    def deserialize_branch(cls, data: Dict[str, Any]) -> 'Branch':
        """Deserialize a Branch object with enhanced error handling."""
        try:
            return Branch(
                path=cls.deserialize_runnable(data['path']),
                ends=data['ends'],
                then=data['then']
            )
        except Exception as e:
            raise DeserializationError(f"Failed to deserialize Branch: {str(e)}")

    @classmethod
    def serialize_graph(cls, graph: Any) -> Dict[str, Any]:
        """Serialize the Graph object with enhanced error handling."""
        try:
            # Log graph type and attributes for debugging
            logger.info(f"Serializing graph of type: {type(graph)}")
            logger.info(f"Graph module: {graph.__module__}")
            
            # Get state schema
            state_schema = getattr(graph, 'state_schema', None)
            if state_schema is not None:
                logger.info(f"Found state schema: {state_schema}")
                state_schema = {
                    'type': 'state_schema',
                    'class_name': state_schema.__name__,
                    'module': state_schema.__module__,
                }
            
            # Get input and output types
            input_type = getattr(graph, 'input', None)
            output_type = getattr(graph, 'output', None)
            
            if input_type is not None:
                logger.info(f"Found input type: {input_type}")
                input_type = {
                    'type': 'type',
                    'class_name': input_type.__name__,
                    'module': input_type.__module__,
                }
            
            if output_type is not None:
                logger.info(f"Found output type: {output_type}")
                output_type = {
                    'type': 'type',
                    'class_name': output_type.__name__,
                    'module': output_type.__module__,
                }
            
            # Determine if graph is compiled based on multiple indicators
            is_compiled = (
                isinstance(graph, CompiledStateGraph) or
                getattr(graph, 'compiled', False) or
                hasattr(graph, 'invoke')
            )
            logger.info(f"Graph is_compiled: {is_compiled}")
            
            # Get edges with special handling for END node
            edges = []
            edge_set = getattr(graph, 'edges', set())
            logger.info(f"Original edges: {edge_set}")
            
            for edge in edge_set:
                if isinstance(edge, tuple) and len(edge) == 2:
                    start, end = edge
                    # Convert START/END constants to string representation
                    if start == START:
                        start = '__start__'
                    if end == END:
                        end = '__end__'
                    edges.append([start, end])
            logger.info(f"Serializing edges: {edges}")
            
            # Store original entry point before compilation
            original_entry = None
            if hasattr(graph, 'entry_point'):
                original_entry = graph.entry_point
            elif hasattr(graph, '_entry_point'):
                original_entry = graph._entry_point
            
            # Create base serialization
            serialized = {
                'nodes': {
                    name: cls.serialize_node_spec(spec)
                    for name, spec in graph.nodes.items()
                },
                'edges': edges,
                'branches': {
                    start: {
                        name: cls.serialize_branch(branch)
                        for name, branch in branches.items()
                    }
                    for start, branches in getattr(graph, 'branches', {}).items()
                },
                'state_schema': state_schema,
                'input': input_type,
                'output': output_type,
                'compiled': is_compiled,
                'type': 'compiled_state_graph' if is_compiled else 'state_graph',
                'original_entry_point': original_entry  # Store the original entry point
            }
            
            # Try to capture entry points
            try:
                # First try explicit entry point
                if hasattr(graph, 'entry_point'):
                    entry_point = graph.entry_point
                    # Don't store special nodes as entry points
                    if not cls._is_special_node(entry_point):
                        serialized['entry_point'] = entry_point
                # Then try internal _entry_point
                elif hasattr(graph, '_entry_point'):
                    entry_point = graph._entry_point
                    # Don't store special nodes as entry points
                    if not cls._is_special_node(entry_point):
                        serialized['entry_point'] = entry_point
                # Finally try entry_points list
                elif hasattr(graph, 'entry_points'):
                    # Filter out special nodes from entry points
                    entry_points = cls._filter_special_nodes(list(graph.entry_points))
                    if entry_points:
                        serialized['entry_points'] = entry_points
                
                # Validate entry point(s)
                if 'entry_point' in serialized:
                    if serialized['entry_point'] not in graph.nodes:
                        logger.warning(f"Entry point {serialized['entry_point']} not found in nodes")
                        del serialized['entry_point']
                    elif cls._is_special_node(serialized['entry_point']):
                        logger.warning(f"Entry point {serialized['entry_point']} is a special node, removing")
                        del serialized['entry_point']
                
                if 'entry_points' in serialized:
                    # Filter out special nodes and validate against nodes
                    valid_points = [p for p in serialized['entry_points'] 
                                  if p in graph.nodes and not cls._is_special_node(p)]
                    if not valid_points:
                        logger.warning("No valid entry points found")
                        del serialized['entry_points']
                    else:
                        serialized['entry_points'] = valid_points
            except Exception as e:
                logger.warning(f"Could not serialize entry points: {e}")
            
            logger.info(f"Serialized graph type: {serialized['type']}")
            return serialized
            
        except Exception as e:
            raise SerializationError(f"Failed to serialize Graph: {str(e)}")

    @classmethod
    def save_graph(cls, graph: 'Graph', filepath: str) -> None:
        """Save graph to JSON file with error handling."""
        try:
            serialized = cls.serialize_graph(graph)
            with open(filepath, 'w') as f:
                json.dump(serialized, f, indent=2)
        except Exception as e:
            raise SerializationError(f"Failed to save graph to {filepath}: {str(e)}")

    @classmethod
    def load_graph(cls, filepath: str) -> 'Graph':
        """Load graph from JSON file with error handling."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls.deserialize_graph(data)
        except Exception as e:
            raise DeserializationError(f"Failed to load graph from {filepath}: {str(e)}")

    @classmethod
    def deserialize_graph(cls, data: Dict[str, Any]) -> Any:
        """Deserialize a graph object with enhanced error handling."""
        try:
            graph_type = data.get('type')
            if not graph_type:
                raise DeserializationError("No graph type specified in data")

            logger.debug("Deserializing graph of type: %s", graph_type)
            is_compiled = data.get('is_compiled', False)
            logger.debug("Graph compiled flag: %s", is_compiled)

            # Create a new graph with default state schema
            logger.info("Creating graph with default state schema")
            graph = StateGraph(DefaultState)

            # Deserialize nodes
            nodes = {}
            for node_name, node_data in data.get('nodes', {}).items():
                try:
                    # Skip special nodes during initial deserialization
                    if node_name in ('__start__', '__end__'):
                        continue
                    
                    node = cls.deserialize_node_spec(node_data)
                    if node is not None:
                        nodes[node_name] = node
                except Exception as e:
                    logger.warning(f"Failed to deserialize node {node_name}: {e}")
                    continue

            logger.debug("Deserialized nodes: %s", list(nodes.keys()))

            # Restore edges
            edges = set()
            for edge in data.get('edges', []):
                if len(edge) == 2:
                    source, target = edge
                    # Convert string representation back to constants
                    if source == '__start__':
                        source = START
                    if target == '__end__':
                        target = END
                    edges.add((source, target))

            logger.debug("Restored edges: %s", edges)

            # Add nodes and edges to graph
            for node_name, node in nodes.items():
                graph.add_node(node_name, node)

            for source, target in edges:
                # Handle special nodes in edges
                if source == START:
                    if target in nodes:
                        graph.set_entry_point(target)
                elif target == END:
                    if source in nodes:
                        graph.add_edge(source, END)
                elif source in nodes and target in nodes:
                    graph.add_edge(source, target)

            logger.debug("Graph edges: %s", graph.edges)
            logger.debug("Graph branches: %s", graph.branches)

            # Set entry point if specified
            entry_point = data.get('entry_point')
            if entry_point and entry_point in nodes and entry_point != '__start__':
                logger.info(f"Setting entry point to: {entry_point}")
                graph.set_entry_point(entry_point)
            else:
                # Try to find a valid entry point
                valid_entry_points = [n for n in nodes.keys() if n not in ('__start__', '__end__')]
                if valid_entry_points:
                    entry_point = valid_entry_points[0]
                    logger.info(f"Setting entry point to first valid node: {entry_point}")
                    graph.set_entry_point(entry_point)
                else:
                    logger.warning("No valid entry point could be set")

            # Always compile the graph to ensure it has the invoke method
            logger.info("Compiling graph...")
            logger.debug("Graph state schema: %s", graph.schema)
            logger.debug("Graph input: %s", graph.input)
            logger.debug("Graph output: %s", graph.output)
            logger.debug("Graph edges before compilation: %s", graph.edges)
            logger.debug("Graph nodes before compilation: %s", list(graph.nodes.keys()))

            try:
                # Store original edges
                original_edges = graph.edges.copy()
                logger.debug("Stored original edges: %s", original_edges)

                # Compile the graph
                compiled_graph = graph.compile()
                logger.info("Successfully compiled graph")

                # Restore original edges
                compiled_graph.edges = original_edges
                logger.debug("Restored edges after compilation: %s", compiled_graph.edges)

                return compiled_graph
            except Exception as e:
                logger.error(f"Failed to compile graph: {e}")
                raise DeserializationError(f"Failed to compile graph: {str(e)}")

        except Exception as e:
            raise DeserializationError(f"Failed to deserialize graph: {str(e)}")

    @staticmethod
    def _is_special_node(node_name: str) -> bool:
        """Helper method to identify special nodes that should not be treated as user nodes."""
        if node_name in (START, END, '__start__', '__end__'):
            return True
        if isinstance(node_name, str) and node_name.startswith('__'):
            return True
        return False

    @staticmethod
    def _filter_special_nodes(nodes: List[str]) -> List[str]:
        """Helper method to filter out special nodes from a list of node names."""
        return [n for n in nodes if not GraphSerializer._is_special_node(n)]
