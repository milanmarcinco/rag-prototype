from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "qwen3-embedding:8b"
OLLAMA_LLM_MODEL = "qwen3.5:4b"
OLLAMA_LLM_CONTEXT_WINDOW = 16384

ollama_embedder = OllamaEmbedding(
    model_name=OLLAMA_EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL,
)

# ollama_embedder.get_text_embedding("The capital of France is Paris.")
# ollama_embedder.get_query_embedding("What is the capital of France?")

ollama_llm = Ollama(
    model=OLLAMA_LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    context_window=OLLAMA_LLM_CONTEXT_WINDOW,
    thinking=False,
)

# To be continued...
