# MINTRI | RAG Prototype

## Requirements

- `mise v2026.4.28` or newer

## Run locally

Steps:

```zsh
mise install
python -m venv env
source env/bin/activate
pip install -r requirements.txt
ollama serve # in a separate shell
# At this point copy the dataset.json file to the root directory
python src/main.py
```
