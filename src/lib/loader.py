import json
from llama_index.core import Document

CHUNK_TEMPLATE = """
Guide: {title}
Category: {category}
Tools: {tools}

{steps_text}
"""


def load_manuals(json_path, max_manuals=10):
    documents = []
    manual_count = 0

    with open(json_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            manual = json.loads(line)
            title = manual.get("Title", "")
            category = manual.get("Category", "")
            tools = [t["Name"] for t in manual.get("Toolbox", []) if t.get("Name")]

            steps_text = ""
            for step in manual.get("Steps", []):
                text = step.get("Text_raw", "").strip()

                if not text:
                    continue


                step = int(step.get("Order", 0)) + 1
                steps_text += f"Step {step}: {text}\n"

            if not steps_text:
                continue

            chunk = CHUNK_TEMPLATE.format(
                title=title,
                category=category,
                tools=", ".join(tools),
                steps_text=steps_text,
            )

            metadata = {
                "title": title,
                "category": category,
                "tools": tools,
            }

            documents.append(Document(text=chunk, extra_info=metadata))

            manual_count += 1
            if manual_count >= max_manuals:
                break

    return documents
