from typing import Tuple

from llama_index.core.base.base_query_engine import BaseQueryEngine
from google.genai.errors import ServerError


def get_response(
    query: str, query_engine: BaseQueryEngine
) -> Tuple[str, Exception | None]:
    try:
        response = query_engine.query(query)
    except ServerError as e:
        return "Sorry, the LLM server is currently unavailable.", e
    except Exception as e:
        return "Sorry, something went wrong while processing your query...", e

    return str(response), None


def print_response(response: str, error: Exception | None, end: str = "\n") -> None:
    if error:
        if type(error) == ServerError:
            print(f"ERROR: LLM server is currently unavailable.", end=end)
        else:
            print(f"ERROR: {error}", end=end)
    else:
        print(f"RESPONSE: {response}", end=end)


def answer_query(query: str, query_engine: BaseQueryEngine, end: str = "\n") -> None:
    response, error = get_response(query, query_engine)
    print_response(response, error, end=end)


def run_query(query: str, query_engine: BaseQueryEngine) -> None:
    print(f"QUERY: {query}")
    answer_query(query, query_engine)


def run_query_prompt(query_engine: BaseQueryEngine) -> None:
    while True:
        print("YOUR QUERY: ", end="")

        try:
            query = input()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        answer_query(query, query_engine, end="\n\n")
