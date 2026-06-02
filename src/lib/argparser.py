from argparse import ArgumentParser, Namespace
from typing import Optional
from pydantic.dataclasses import dataclass


@dataclass
class Args(Namespace):
    query: Optional[str] = None
    top_k: int = 3
    print_sources: bool = False
    max_manuals: int = 100
    rebuild_index: bool = False


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
    help="Whether to print the source documents used to answer the question.",
)

parser.add_argument(
    "--max-manuals",
    type=int,
    default=100,
    help="Maximum number of manuals to load and index from the dataset.",
)

parser.add_argument(
    "--rebuild-index",
    action="store_true",
    help="Force the persisted index to be regenerated before querying.",
)

ns = parser.parse_args()

args = Args(
    query=ns.query,
    top_k=ns.top_k,
    print_sources=ns.print_sources,
    max_manuals=ns.max_manuals,
    rebuild_index=ns.rebuild_index,
)
