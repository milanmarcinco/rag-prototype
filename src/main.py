from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from loader import load_manuals
from dotenv import load_dotenv
import os

OLLAMA_BASE_URL = "http://localhost:11434"
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url=OLLAMA_BASE_URL,
)

Settings.llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=api_key,
)

PERSIST_DIR = "./storage"

if os.path.exists(PERSIST_DIR):
    print("Loading index from disk...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
else:
    print("Building index for the first time...")
    docs = load_manuals("docs/PC.json", max_manuals=200)
    print(f"Loaded {len(docs)} chunks")
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print("Saved to disk.")

query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("How do I open the PC case?")
print(response)