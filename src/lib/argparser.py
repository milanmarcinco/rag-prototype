from argparse import ArgumentParser

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
    "--rebuild-index",
    action="store_true",
    help="Force the persisted index to be regenerated before querying.",
)

args = parser.parse_args()
