import json

from typing import List

from llama_index.core import Document

from lib.argparser import args
from lib.text_mining import analyse_guide, calculate_raw_complexity_score

CHUNK_TEMPLATE = """
Id: {id}
Guide: {title}
Category: {category}
Tools: {tools}

{steps_text}
"""


def load_manuals(json_path: str):
    documents = []
    manual_count = 0

    corpus_scores = []

    with open(json_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            manual = json.loads(line)
            tools = [t["Name"] for t in manual.get("Toolbox", []) if t.get("Name")]

            steps = [
                step.get("Text_raw", "").strip() for step in manual.get("Steps", [])
            ]

            complexity_score = calculate_raw_complexity_score(steps, tools)
            corpus_scores.append(complexity_score)

    corpus_scores.sort()

    with open(json_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            manual = json.loads(line)
            id = manual.get("Guidid", "")
            title = manual.get("Title", "")
            category = manual.get("Category", "")
            tools = [t["Name"] for t in manual.get("Toolbox", []) if t.get("Name")]

            steps: List[str] = []

            for step in manual.get("Steps", []):
                text = step.get("Text_raw", "").strip()
                step_order = int(step.get("Order", 0)) + 1
                steps.append(f"Step {step_order}: {text}")

            analysis = analyse_guide(steps, tools, corpus_scores)

            chunk_size = args.steps_per_chunk or len(steps)
            overlap = args.steps_overlap if args.steps_per_chunk else 0

            chunk_start = 0
            while chunk_start < len(steps):
                step_chunk = steps[chunk_start : chunk_start + chunk_size]
                steps_text = "\n".join(step_text for step_text in step_chunk)

                chunk = CHUNK_TEMPLATE.format(
                    id=id,
                    title=title,
                    category=category,
                    tools=", ".join(tools),
                    steps_text=steps_text,
                )

                metadata = {
                    "id": id,
                    "title": title,
                    "category": category,
                    "tools": tools,
                    **analysis,
                }

                documents.append(Document(text=chunk, extra_info=metadata))

                if chunk_start + chunk_size >= len(steps):
                    break

                chunk_start += chunk_size - overlap

            manual_count += 1
            if manual_count >= args.max_manuals:
                break

    return documents
