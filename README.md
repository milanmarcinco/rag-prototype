# MINTRI | RAG Prototype

Command-line RAG assistant for querying device repair manuals from the MyFixit dataset. It combines dense and BM25 retrieval, generates grounded answers, displays source evidence, and estimates guide complexity and risk.

## Requirements

- `mise v2026.4.28` or newer
- Optional Gemini API key from [Google AI Studio](https://aistudio.google.com)

## Setup

```sh
mise install
python -m venv env
source env/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Start Ollama in a separate terminal,


```sh
ollama serve
```

and download the required models:

```sh
ollama pull nomic-embed-text

# Only needed for local answer generation if not using Gemini
ollama pull qwen3.5:4b
```

`nomic-embed-text` is always required for embeddings. `qwen3.5:4b` is required only for local answer generation.

## Configuration

Edit `.env` as needed. The main defaults are:

```dotenv
DATASET_DIR=dataset
PERSIST_DIR=storage
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=qwen3.5:4b
GEMINI_LLM_MODEL=gemini-2.5-flash
```

Replace the placeholder `GEMINI_API_KEY` with a real key to use Gemini for answer generation, or remove `GEMINI_API_KEY` to use the local Ollama LLM. Ollama remains required because embeddings are generated locally in both configurations.

## Usage

Run a one-shot query:

```sh
python src/main.py --query "How do I disconnect an iPhone 3G display?"
```

Start interactive mode:

```sh
python src/main.py
```

Rebuild the persisted index:

```sh
python src/main.py --rebuild-index
```

Print retrieved sources and change retrieval depth:

```sh
python src/main.py --query "How do I disconnect an iPhone 3G display?" --print-sources --top-k 5
```

Retrieve sources without generating an answer:

```sh
python src/main.py --retriever-only --query "How do I disconnect an iPhone 3G display?"
```

The first run builds and persists the index. Later runs load it from `PERSIST_DIR`. Use `--rebuild-index` after changing the dataset, embedding model, corpus size, or chunking settings.
