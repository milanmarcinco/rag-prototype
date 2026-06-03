from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.retrievers.bm25 import BM25Retriever

REPAIR_QUERY_TEMPLATE = """
You are a careful repair assistant.
Answer the question using only the retrieved repair-manual context below.

Rules:
- If the context does not contain enough evidence, say that the retrieved manuals do not provide enough information.
- If the user does not ask about a specific device, try to generalize the answer based on the retrieved manuals. Do not mention devices the user did not ask about.
- Do not invent tools, steps, warnings, measurements, or part names.
- Prefer clear step-by-step repair guidance when the context supports it.
- Mention relevant safety warnings from the context.

Retrieved context:
{context_str}

Question:
{query_str}

Grounded answer:
"""


def build_hybrid_query_engine(
    index: VectorStoreIndex,
    top_k: int,
) -> RetrieverQueryEngine:
    vector_retriever = index.as_retriever(
        similarity_top_k=top_k,
    )

    bm25_retriever = BM25Retriever.from_defaults(
        index=index,
        similarity_top_k=top_k,
    )

    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        retriever_weights=[0.6, 0.4],
        mode=FUSION_MODES.RECIPROCAL_RANK,
        similarity_top_k=top_k,
        num_queries=1,
        use_async=False,
    )

    return RetrieverQueryEngine.from_args(
        hybrid_retriever,
        text_qa_template=PromptTemplate(REPAIR_QUERY_TEMPLATE),
    )
