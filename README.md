# MINTRI | RAG Prototype

## Requirements

- `mise v2026.4.28` or newer
- Gemini API key (free at https://aistudio.google.com)

## Run locally

Steps:

```zsh
mise install
python -m venv env
source env/bin/activate
pip install -r requirements.txt
ollama serve # in a separate shell
ollama pull <model-name>  # only needed once per model
```

Copy `.env.example` to `.env` and add necessary environment variables. Either set `OLLAMA` variables to use Ollama's local LLM models, or set `GEMINI_API_KEY` to use Gemini.

Copy `PC.json` from the [MyFixit dataset](https://github.com/rub-ksv/MyFixit-Dataset) into the dataset directory (configured by `DATASET_DIR`).

```zsh
python src/main.py "How do I open the PC case?"
```

The first run builds the index from the manuals and saves it to the persistence directory (configured by `PERSIST_DIR`) — this takes a few minutes. Every run after loads from disk instantly.

Use `--top-k` to change how many chunks are retrieved, and `--rebuild-index` to regenerate the persisted index:

```zsh
python src/main.py --top-k 5 --rebuild-index "How do I open the PC case?"
```
