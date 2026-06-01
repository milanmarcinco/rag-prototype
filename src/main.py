import os

from typing import Tuple

from llama_index.llms.ollama import Ollama
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.ollama import OllamaEmbedding

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
)

from google.genai.errors import ServerError

from lib.config import (
    DATASET_DIR,
    PERSIST_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_LLM_MODEL,
    GEMINI_API_KEY,
    GEMINI_LLM_MODEL,
)

from lib.argparser import args
from lib.loader import load_manuals

QUERY = args.query
TOP_K = args.top_k

Settings.embed_model = OllamaEmbedding(
    model_name=OLLAMA_EMBED_MODEL,
    base_url=OLLAMA_BASE_URL,
)

if GEMINI_API_KEY:
    print("Using Gemini API for LLM...")

    Settings.llm = GoogleGenAI(
        model=GEMINI_LLM_MODEL,
        api_key=GEMINI_API_KEY,
        max_retries=0,
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

query_engine = index.as_query_engine(similarity_top_k=TOP_K)


def get_response(query: str) -> Tuple[str, Exception | None]:
    try:
        response = query_engine.query(query)
    except ServerError as e:
        return "Sorry, the LLM server is currently unavailable.", e
    except Exception as e:
        return "Sorry, something went wrong while processing your query...", e

    return str(response), None


def print_response(response: str, error: Exception | None, end: str = "\n") -> None:
    if error:
        if type(error) == ServerError:
            print(f"ERROR: LLM server is currently unavailable.", end=end)
        else:
            print(f"ERROR: {error}", end=end)
    else:
        print(f"RESPONSE: {response}", end=end)


if QUERY:
    print(f"QUERY: {QUERY}")

    response, error = get_response(QUERY)
    print_response(response, error)
else:
    while True:
        print("YOUR QUERY: ", end="")

        try:
            QUERY = input()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        response, error = get_response(QUERY)
        print_response(response, error, end="\n\n")
