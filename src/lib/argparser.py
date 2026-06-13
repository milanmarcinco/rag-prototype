from argparse import ArgumentParser, Namespace
from typing import Optional
from pydantic.dataclasses import dataclass


@dataclass
class Args(Namespace):
    query: Optional[str]
    top_k: int
    print_sources: bool
    retriever_only: bool
    max_manuals: int
    steps_per_chunk: int | None
    steps_overlap: int
    rebuild_index: bool


parser = ArgumentParser(description="Ask a question over the repair manual index.")

parser.add_argument(
    "--query",
    type=str,
    default=None,
    help="Question to answer from the indexed manuals.",
)

parser.add_argument(
    "--top-k",
    type=int,
    default=3,
    help="Number of document chunks to retrieve before answering.",
)

parser.add_argument(
    "--print-sources",
    action="store_true",
    default=False,
    help="Whether to print the source documents used to answer the question.",
)

parser.add_argument(
    "--retriever-only",
    action="store_true",
    default=False,
    help="Only retrieve matching chunks. Do not call an LLM to generate an answer. Does not require --print-sources to be set, as source documents will be printed by default in this mode.",
)

parser.add_argument(
    "--max-manuals",
    type=int,
    default=100,
    help="Maximum number of manuals to load and index from the dataset.",
)

parser.add_argument(
    "--steps-per-chunk",
    type=int,
    default=None,
    help="Number of steps to include in each document chunk. If not set, all steps will be included in a single chunk.",
)

parser.add_argument(
    "--steps-overlap",
    type=int,
    default=0,
    help="Number of steps to overlap between consecutive document chunks.",
)

parser.add_argument(
    "--rebuild-index",
    action="store_true",
    default=False,
    help="Force the persisted index to be regenerated before querying.",
)

ns = parser.parse_args()

args = Args(
    query=ns.query,
    top_k=ns.top_k,
    print_sources=ns.print_sources,
    retriever_only=ns.retriever_only,
    max_manuals=ns.max_manuals,
    steps_per_chunk=ns.steps_per_chunk,
    steps_overlap=ns.steps_overlap,
    rebuild_index=ns.rebuild_index,
)
