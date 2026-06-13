from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.retrievers.bm25 import BM25Retriever

REPAIR_QUERY_TEMPLATE = """
You are a careful repair assistant.
Answer the question using only the retrieved repair-manual context below.

## Rules:
- Grounding: do not invent tools, steps, warnings, measurements, or part names. 
    If the context does not contain enough evidence, say that the retrieved manuals do not provide enough information.
- Ambiguity Check: If the user's question is broad (e.g., "How do I remove the battery?") and the context contains instructions for multiple different devices, DO NOT combine them. 
    Instead, briefly state which devices were found in the context and ask the user to specify which one they need.
- Image References: Delete any steps or sentences that refer to an image, photo, or diagram that cannot be seen (e.g., "as pictured", "as shown", "this is what should be left"). 
    Rewrite the step to be self-contained or omit it if it contains no action.
- Step Clarity: Keep numbered steps concise and action-oriented. Do not copy and paste massive paragraphs of safety warnings directly into a numbered step if they are already in the Prerequisites section.
- Context Check: If the instructions start in the middle of a procedure (e.g., separating components without explaining how to open the device first), include a brief introductory note stating that these instructions pick up after the device has already been opened.
- Text-Mining Metadata: The retrieved context may include guide-level fields named num_steps, num_tools, risk_terms, action_counts, complexity_score, and complexity_label.
    Treat these fields as a heuristic summary, not a safety certification. Do not combine scores from different guides.

## Response Format
If the query is specific or only one device's context is retrieved, produce your answer in this exact structure:

### Repair Complexity & Risk
Name the relevant guide and describe its guide-level complexity using the mined metadata.
Include the score, total steps, total tools, risk indicators, and most common actions with non-zero counts.
Describe a "medium" label as "moderately complex".
If the metadata is unavailable, say that complexity analysis is unavailable.

### Tools Needed
List every tool mentioned in the retrieved context that is required for this procedure. 
If no tools are mentioned, say that no special tools are needed.

### Prerequisites / Safety Checks
List any preparation steps, warnings, or safety checks from the context.

### Step-by-Step Repair Instructions
Provide the complete procedure as a numbered list. 
Include every step found in the retrieved context in the correct order. 
Ensure the steps are clean, clear, and chronological based on the text.
Do not skip or merge steps. Mention relevant safety warnings inline.


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
