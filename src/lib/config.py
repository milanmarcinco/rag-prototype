import os

from dotenv import load_dotenv

load_dotenv()

DATASET_DIR = os.getenv("DATASET_DIR", "dataset")
PERSIST_DIR = os.getenv("PERSIST_DIR", "storage")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3.5:4b")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")
