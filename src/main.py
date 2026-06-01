from llama_index.llms.ollama import Ollama
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.ollama import OllamaEmbedding

from llama_index.core import (
    Settings,
)

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
from lib.indexer import load_or_build_index

from lib.helpers import run_query, run_query_prompt

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

index = load_or_build_index(
    DATASET_DIR,
    PERSIST_DIR,
    max_manuals=args.max_manuals,
    rebuild_index=args.rebuild_index,
)

query_engine = index.as_query_engine(similarity_top_k=args.top_k)

if args.query:
    run_query(args.query, query_engine)
else:
    run_query_prompt(query_engine)
