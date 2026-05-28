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
ollama pull nomic-embed-text  # only needed once
```

Copy `.env.example` to `.env` and add your Gemini API key:

Copy `PC.json` from the [MyFixit dataset](https://github.com/rub-ksv/MyFixit-Dataset) into `docs/`.

```zsh
python src/main.py
```

The first run builds the index from the manuals and saves it to `./storage/` — this takes a few minutes. Every run after loads from disk instantly.
