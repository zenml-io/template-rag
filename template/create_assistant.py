from langgraph.graph import END, StateGraph

graph = StateGraph()
graph.add_edge('retrieve', 'generate')
graph.add_edge('generate', END) 
