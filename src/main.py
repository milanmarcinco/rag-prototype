import os

from loader import load_manuals
from dotenv import load_dotenv

from llama_index.llms.ollama import Ollama
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.ollama import OllamaEmbedding

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
)

load_dotenv()

DATASET_DIR = os.getenv("DATASET_DIR", "dataset")
PERSIST_DIR = os.getenv("PERSIST_DIR", "storage")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3.5:4b")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")

Settings.embed_model = OllamaEmbedding(
    model_name=OLLAMA_EMBED_MODEL,
    base_url=OLLAMA_BASE_URL,
)

if GEMINI_API_KEY:
    print("Using Gemini API for LLM...")

    Settings.llm = GoogleGenAI(
        model=GEMINI_LLM_MODEL,
        api_key=GEMINI_API_KEY,
    )
else:
    print("Using Ollama for LLM...")

    Settings.llm = Ollama(
        model=OLLAMA_LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        context_window=16384,
        thinking=False,
    )

if os.path.exists(PERSIST_DIR):
    print("Loading index from disk...", end="\n\n")

    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
else:
    print("Building index for the first time...")

    dataset_path = os.path.join(os.getcwd(), DATASET_DIR, "dataset.json")
    docs = load_manuals(dataset_path, max_manuals=200)

    print(f"Loaded {len(docs)} chunks. Indexing...")
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print("Saved to disk.", end="\n\n")

QUERY = "How do I open the PC case?"

query_engine = index.as_query_engine(similarity_top_k=3)
print(f"QUERY: {QUERY}")
response = query_engine.query(QUERY)
print(f"RESPONSE: {response}")
