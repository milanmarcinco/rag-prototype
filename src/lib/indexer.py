import os
import shutil

from typing import cast

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage

from lib.argparser import args
from lib.loader import load_manuals


def build_index(dataset_dir: str, persist_dir: str) -> VectorStoreIndex:
    dataset_path = os.path.join(os.getcwd(), dataset_dir, "dataset.json")
    docs = load_manuals(dataset_path)

    print(f"Loaded {len(docs)} chunks. Indexing...")

    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=persist_dir)
    print("Index built and saved to disk.", end="\n\n")

    return index


def load_or_build_index(dataset_dir: str, persist_dir: str) -> VectorStoreIndex:
    if args.rebuild_index:
        print("Rebuilding index from source manuals...")

        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

        return build_index(dataset_dir, persist_dir)

    try:
        print("Loading index from disk...")

        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = cast(VectorStoreIndex, load_index_from_storage(storage_context))

        print("Index loaded successfully.", end="\n\n")

        return index
    except Exception:
        print("No existing index found. Building index for the first time...")
        return build_index(dataset_dir, persist_dir)
