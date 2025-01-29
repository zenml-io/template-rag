from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.workflow import (
    Context,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from utils.events import RerankEvent, RetrieverEvent


class RAGWorkflow(Workflow):
    def __init__(self, index: VectorStoreIndex | None = None):
        super().__init__()
        self.index = index

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieverEvent | None:
        "Entry point for RAG, triggered by a StartEvent with `query`."
        query = ev.get("query")

        if not query:
            return None

        print(f"Query the database with: {query}")

        # store the query in the global context
        await ctx.set("query", query)

        # get the index from the global context
        if self.index is None:
            print("Index is empty, load some documents before querying!")
            return None

        retriever = self.index.as_retriever(similarity_top_k=2)
        nodes = await retriever.aretrieve(query)
        print(f"Retrieved {len(nodes)} nodes.")
        return RetrieverEvent(nodes=nodes)

    @step
    async def rerank(self, ctx: Context, ev: RetrieverEvent) -> RerankEvent:
        # Rerank the nodes
        ranker = LLMRerank(
            choice_batch_size=5, top_n=3, llm=OpenAI(model="gpt-4o-mini")
        )
        print(await ctx.get("query", default=None), flush=True)
        new_nodes = ranker.postprocess_nodes(
            ev.nodes, query_str=await ctx.get("query", default=None)
        )
        print(f"Reranked nodes to {len(new_nodes)}")
        return RerankEvent(nodes=new_nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RerankEvent) -> StopEvent:
        """Return a streaming response using reranked nodes."""
        llm = OpenAI(model="gpt-4o-mini")
        summarizer = CompactAndRefine(llm=llm, streaming=True, verbose=True)
        query = await ctx.get("query", default=None)

        response = await summarizer.asynthesize(query, nodes=ev.nodes)
        return StopEvent(result=response)
