import json
from llama_index.core import Document


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
                if text:
                    steps_text += f"Step {step['Order']}: {text}\n"

            if not steps_text:
                continue

            chunk = f"Guide: {title}\nCategory: {category}\nTools: {', '.join(tools)}\n\n{steps_text}"
            
            documents.append(
                    Document(
                        text=chunk,
                        metadata={
                            "title": title,
                            "category": category,
                            "tools": tools,
                        },
                    )
            )

            manual_count += 1
            if manual_count >= max_manuals:
                break

    return documents


# docs = load_manuals("Phone.json", max_manuals=3)
# print(f"Number of documents: {len(docs)}")
# print("\n--- First document ---")
# print(docs[0].text)