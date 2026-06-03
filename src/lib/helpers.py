from typing import Any, Tuple, cast

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response

from google.genai.errors import ServerError

from lib.argparser import args


def get_response(
    query: str, query_engine: BaseQueryEngine
) -> Tuple[Response | None, Exception | None]:
    try:
        if args.retriever_only:
            retriever = getattr(query_engine, "retriever", None) or getattr(
                query_engine, "_retriever", None
            )

            if retriever is None:
                raise ValueError("Query engine does not expose a retriever.")

            source_nodes = retriever.retrieve(query)
            response = Response(response="", source_nodes=source_nodes)
        else:
            response = cast(Response, query_engine.query(query))
    except ServerError as e:
        return None, e
    except Exception as e:
        return None, e

    return response, None


def print_response(
    response: Response | None, error: Exception | None, end: str = "\n"
) -> None:
    if error:
        if type(error) == ServerError:
            print(f"ERROR: LLM server is currently unavailable.", end=end)
        else:
            print(f"ERROR: {error}", end=end)
    else:
        print(f"RESPONSE: {response}", end=end)


def format_source(source: Any, index: int) -> str:
    node = source.node
    metadata = node.metadata or {}
    score = getattr(source, "score", None)
    score_text = f" | Score: {score:.4f}" if score is not None else ""

    title = metadata.get("title") or "Unknown guide"
    category = metadata.get("category") or "Unknown category"
    text = node.get_content().strip()

    return f"[{index}] {title} | {category}{score_text}\n" f"{text}"


def print_sources(response: Response | None, end: str = "\n") -> None:
    sources = getattr(response, "source_nodes", None) or []

    if not sources:
        print("SOURCES: No source documents were returned.", end=end)
        return

    print("SOURCES:")
    for index, source in enumerate(sources, start=1):
        print(format_source(source, index))
        if index < len(sources):
            print()

    print(end=end)


def answer_query(
    query: str,
    query_engine: BaseQueryEngine,
    print_source_documents: bool = False,
    end: str = "\n",
) -> None:
    response, error = get_response(query, query_engine)

    if args.retriever_only and error is None:
        print_sources(response, end=end)
    elif print_source_documents and error is None:
        print_response(response, error, end="\n\n")
        print_sources(response, end=end)
    else:
        print_response(response, error, end=end)


def run_query(
    query: str,
    query_engine: BaseQueryEngine,
    print_source_documents: bool = False,
) -> None:
    print(f"QUERY: {query}")

    answer_query(
        query,
        query_engine,
        print_source_documents=print_source_documents,
    )


def run_query_prompt(
    query_engine: BaseQueryEngine,
    print_source_documents: bool = False,
) -> None:
    while True:
        print("YOUR QUERY: ", end="")

        try:
            query = input()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        answer_query(
            query,
            query_engine,
            print_source_documents=print_source_documents,
            end="\n\n",
        )
