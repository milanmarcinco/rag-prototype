# TODO

## 1. Prototype Usability

- [x] Replace the hard-coded query in `src/main.py` with CLI input.
- [x] Add a `--query` argument for one-shot demo questions.
- [x] Add an interactive mode for asking multiple questions in one run.
- [x] Add a `--top-k` argument for retrieval depth.
- [x] Add a `--rebuild-index` flag to force index regeneration.
- [x] Print clear answer output followed by retrieved source evidence.
- [x] Show each source chunk's guide title, category, step number, and text.

## 2. Code Structure

- [x] Move environment/config loading into `src/config.py`.
- [x] Move index build/load logic into `src/indexer.py`.
- [x] Move query and answer logic into `src/rag.py`.
- [x] Keep `src/main.py` as the CLI entry point only.
- [x] Add small helper functions for formatting sources and metadata.
- [x] Make `max_manuals` configurable instead of fixed in code.

## 3. RAG Quality

- [x] Add a custom prompt template for grounded repair answers.
- [x] Instruct the model to answer only from retrieved context.
- [x] Instruct the model to say when retrieved evidence is insufficient.
- [ ] Test at least 5 representative repair questions manually.
- [ ] Tune `similarity_top_k` based on observed retrieval quality.

## 4. Corpus Preparation

- [ ] Document the dataset source and how to obtain it.
- [ ] Record corpus size: manuals, chunks, categories, average steps.
- [ ] Verify all required metadata fields are loaded correctly.
- [ ] Decide whether to index all manuals or a subset for final demo.
- [ ] Document chunking strategy: one manual step per document.
- [ ] Document trade-offs of step-level chunking.

## 5. Text Mining Features

- [ ] Create `src/text_mining.py`.
- [ ] Implement keyword extraction for retrieved chunks.
- [ ] Implement summarization of retrieved repair evidence.
- [ ] Implement category/topic distribution for retrieved chunks.
- [ ] Implement corpus statistics by category/tool/subject.
- [ ] Add one stronger feature: clustering or classification.
- [ ] Display text-mining output in the CLI demo.
- [ ] Save at least one text-mining example for the report.

## 6. Evaluation

- [ ] Create `eval/eval_queries.json`.
- [ ] Add 10-20 test questions.
- [ ] For each question, define expected guide/category/keywords.
- [ ] Create `src/evaluate.py`.
- [ ] Measure retrieval hit rate at `k=3`.
- [ ] Measure retrieval hit rate at `k=5`.
- [ ] Record whether answers cite relevant sources.
- [ ] Score answer usefulness manually on a 1-5 scale.
- [ ] Save evaluation outputs to `eval/results.json`.
- [ ] Identify 2-3 failure cases.
- [ ] Explain why the failure cases happen.

## 7. README

- [ ] Add exact command for running a one-shot query.
- [ ] Add exact command for interactive mode.
- [ ] Add exact command for rebuilding the index.
- [ ] Clarify Ollama-only vs Gemini-backed generation.
- [ ] List required Ollama models to pull.
- [ ] Add dataset placement instructions.
- [ ] Add a short project summary and use case.

## 8. Report

- [ ] Create `report/report.md`.
- [ ] Write problem definition and target users.
- [ ] Explain why PC repair manuals are a suitable RAG use case.
- [ ] Add high-level architecture diagram.
- [ ] Describe corpus source, fields, cleaning, and chunking.
- [ ] Describe embedding model and vector retrieval.
- [ ] Describe generator model and prompt strategy.
- [ ] Describe text-mining features with examples.
- [ ] Add evaluation methodology.
- [ ] Add evaluation results table.
- [ ] Add strengths, limitations, and failure analysis.
- [ ] Add future improvements.
- [ ] Add conclusion.

## 9. Presentation

- [ ] Create a short slide outline.
- [ ] Include problem and motivation.
- [ ] Include architecture.
- [ ] Include corpus and preprocessing.
- [ ] Include one live/demo query.
- [ ] Include text-mining outputs.
- [ ] Include evaluation results.
- [ ] Include limitations and future work.
- [ ] Prepare a 3-5 minute demo script.

## 10. Final Checks

- [ ] Run the prototype from a clean terminal.
- [ ] Rebuild the index successfully.
- [ ] Run evaluation successfully.
- [ ] Confirm ignored files are not committed: `.env`, `dataset/`, `storage/`, `env/`.
- [ ] Check that README instructions match the actual commands.
- [ ] Check that report claims match implemented features.
- [ ] Commit source code, docs, report, and evaluation files.
