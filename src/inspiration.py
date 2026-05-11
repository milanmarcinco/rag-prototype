"""
MyFixit RAG Repair Assistant
Single-file lightweight prototype

Requirements:
- Python 3.10+
- Pinecone account + API key
- OpenAI API key

Run:
streamlit run app.py

Environment Variables:
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=...
"""

import os
import json
import re
from typing import List, Dict, Any

import streamlit as st

from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    Settings,
)

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

from llama_index.vector_stores.pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = "./data/myfixit.jsonl"

PINECONE_INDEX_NAME = "myfixit-rag"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

TOP_K = 5

CHUNK_STEP_SIZE = 4

# ============================================================
# INITIALIZE MODELS
# ============================================================

Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL
)

Settings.llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
)

# ============================================================
# DATASET LOADING
# ============================================================


def load_myfixit_dataset(path: str) -> List[Dict]:
    """
    Load MyFixit dataset from JSONL file.
    """

    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                records.append(item)
            except Exception:
                continue

    return records


# ============================================================
# STEP-BASED CHUNKING
# ============================================================


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_tools(record: Dict) -> List[str]:
    """
    Extract tools from Toolbox field.
    """

    tools = []

    toolbox = record.get("Toolbox", [])

    if isinstance(toolbox, list):
        for tool in toolbox:
            if isinstance(tool, dict):
                name = tool.get("Title")
                if name:
                    tools.append(name)
            elif isinstance(tool, str):
                tools.append(tool)

    return list(set(tools))


def create_step_chunks(record: Dict) -> List[Document]:
    """
    Chunk repair guides by groups of steps.
    """

    documents = []

    title = record.get("Title", "Unknown Repair")
    category = record.get("Category", "Unknown Device")
    subject = record.get("Subject", "")

    tools = extract_tools(record)

    steps = record.get("Steps", [])

    cleaned_steps = []

    for idx, step in enumerate(steps):
        text = ""

        if isinstance(step, dict):
            text = step.get("Text_raw", "")
        elif isinstance(step, str):
            text = step

        text = clean_text(text)

        if text:
            cleaned_steps.append((idx + 1, text))

    for i in range(0, len(cleaned_steps), CHUNK_STEP_SIZE):

        group = cleaned_steps[i:i + CHUNK_STEP_SIZE]

        if not group:
            continue

        chunk_text = []

        chunk_text.append(f"Device: {category}")
        chunk_text.append(f"Repair: {title}")

        if subject:
            chunk_text.append(f"Subject: {subject}")

        chunk_text.append("")

        start_step = group[0][0]
        end_step = group[-1][0]

        for step_num, step_text in group:
            chunk_text.append(f"Step {step_num}:")
            chunk_text.append(step_text)
            chunk_text.append("")

        final_text = "\n".join(chunk_text)

        metadata = {
            "device": category,
            "repair_title": title,
            "tools": tools,
            "step_range": f"{start_step}-{end_step}",
        }

        doc = Document(
            text=final_text,
            metadata=metadata,
        )

        documents.append(doc)

    return documents


# ============================================================
# PINECONE SETUP
# ============================================================


def initialize_pinecone():

    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise ValueError("Missing PINECONE_API_KEY")

    pc = Pinecone(api_key=api_key)

    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

    return pc.Index(PINECONE_INDEX_NAME)


# ============================================================
# BUILD INDEX
# ============================================================


def build_index(dataset_path: str):

    records = load_myfixit_dataset(dataset_path)

    all_documents = []

    for record in records:
        try:
            docs = create_step_chunks(record)
            all_documents.extend(docs)
        except Exception:
            continue

    pinecone_index = initialize_pinecone()

    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    index = VectorStoreIndex.from_documents(
        all_documents,
        storage_context=storage_context,
        show_progress=True,
    )

    return index


# ============================================================
# LOAD EXISTING INDEX
# ============================================================


def load_index():

    pinecone_index = initialize_pinecone()

    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
    )

    return index


# ============================================================
# HYBRID FILTERING
# ============================================================


def keyword_filter(query: str) -> List[str]:
    """
    Extract possible exact identifiers/tool names.
    Lightweight lexical helper.
    """

    patterns = re.findall(
        r"[A-Za-z]+\d+|T\d+|Phillips|Torx|Spudger",
        query,
        flags=re.IGNORECASE,
    )

    return list(set(patterns))


# ============================================================
# TOOL EXTRACTION
# ============================================================


def extract_tools_from_nodes(nodes) -> List[str]:

    tools = set()

    for node in nodes:
        metadata = node.node.metadata

        chunk_tools = metadata.get("tools", [])

        for t in chunk_tools:
            tools.add(t)

    return sorted(list(tools))


# ============================================================
# REPAIR SUMMARY
# ============================================================


def summarize_repair(response_text: str) -> str:

    prompt = f"""
Summarize this repair procedure in 2-3 sentences.

Repair Text:
{response_text}
"""

    llm = Settings.llm

    response = llm.complete(prompt)

    return str(response)


# ============================================================
# QUERY ENGINE
# ============================================================


def build_query_engine(index):

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=TOP_K,
    )

    response_synthesizer = get_response_synthesizer()

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    return query_engine


# ============================================================
# GROUNDED PROMPT
# ============================================================


SYSTEM_PROMPT = """
You are a repair assistant.

Answer ONLY using the provided repair instructions.

If the answer is not present in the context, say:
"I could not find this information."

Avoid hallucinations.
Avoid inventing repair steps.
"""


def run_query(index, user_query: str):

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=TOP_K,
    )

    retrieved_nodes = retriever.retrieve(user_query)

    context = "\n\n".join(
        [node.node.text for node in retrieved_nodes]
    )

    prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{user_query}
"""

    llm = Settings.llm

    response = llm.complete(prompt)

    tools = extract_tools_from_nodes(retrieved_nodes)

    summary = summarize_repair(context)

    return {
        "answer": str(response),
        "summary": summary,
        "tools": tools,
        "chunks": [node.node.text for node in retrieved_nodes],
    }


# ============================================================
# SIMPLE EVALUATION
# ============================================================


EVALUATION_SET = [
    {
        "question": "How do I disconnect a MacBook battery cable?",
        "expected": "battery cable disconnect",
    },
    {
        "question": "What tools are needed for iPhone screen repair?",
        "expected": "tools",
    },
    {
        "question": "How do I remove bottom screws from a laptop?",
        "expected": "remove screws",
    },
]


def evaluate(index):

    results = []

    for item in EVALUATION_SET:

        result = run_query(index, item["question"])

        results.append({
            "question": item["question"],
            "answer": result["answer"][:300],
        })

    return results


# ============================================================
# STREAMLIT UI
# ============================================================


def main():

    st.set_page_config(
        page_title="MyFixit Repair Assistant",
        layout="wide",
    )

    st.title("🔧 MyFixit RAG Repair Assistant")

    st.markdown("""
Lightweight Retrieval-Augmented Generation (RAG) prototype
grounded in MyFixit repair manuals.
""")

    if "index_loaded" not in st.session_state:
        st.session_state.index_loaded = False

    col1, col2 = st.columns(2)

    with col1:

        if st.button("Build Pinecone Index"):

            with st.spinner("Building index..."):

                index = build_index(DATASET_PATH)

                st.session_state.index = index
                st.session_state.index_loaded = True

            st.success("Index built successfully.")

    with col2:

        if st.button("Load Existing Index"):

            with st.spinner("Loading index..."):

                index = load_index()

                st.session_state.index = index
                st.session_state.index_loaded = True

            st.success("Index loaded successfully.")

    st.divider()

    if st.session_state.index_loaded:

        query = st.text_input(
            "Ask a repair question:",
            placeholder="How do I replace a MacBook battery?",
        )

        if st.button("Search") and query:

            with st.spinner("Retrieving repair instructions..."):

                result = run_query(
                    st.session_state.index,
                    query,
                )

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Repair Summary")
            st.write(result["summary"])

            st.subheader("Required Tools")

            if result["tools"]:
                for tool in result["tools"]:
                    st.markdown(f"- {tool}")
            else:
                st.write("No tools found.")

            st.subheader("Retrieved Repair Chunks")

            for idx, chunk in enumerate(result["chunks"]):

                with st.expander(f"Chunk {idx+1}"):

                    st.text(chunk)

    st.divider()

    if st.button("Run Small Evaluation"):

        if not st.session_state.index_loaded:
            st.warning("Load or build an index first.")
        else:

            with st.spinner("Running evaluation..."):

                eval_results = evaluate(
                    st.session_state.index
                )

            st.subheader("Evaluation Results")

            for item in eval_results:

                st.markdown(f"### Question")
                st.write(item["question"])

                st.markdown(f"### Retrieved Answer")
                st.write(item["answer"])

                st.divider()


if __name__ == "__main__":
    main()
