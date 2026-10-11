
from app.rag.retriever import get_retriever
from app.rag.bm25_retriever import get_bm25_retriever


TEST_CASES = [
    {
        "name": "General annual leave question",
        "question": "What is the annual leave policy?",
        "filters": {},
    },
    {
        "name": "DOCX-only annual leave question",
        "question": "What is the annual leave policy?",
        "filters": {"file_type": "docx"},
    },
    {
        "name": "Specific leave policy",
        "question": "How many days of annual leave are allowed?",
        "filters": {"source": "leave_policy.docx"},
    },
    {
        "name": "Travel expense question",
        "question": "What is the travel expense limit?",
        "filters": {"source": "expense_policy.csv"},
    },
    {
        "name": "IT password question",
        "question": "What does the IT policy say about passwords?",
        "filters": {"source": "it_policy.txt"},
    },
]


def check_filters(documents, filters):
    """Ensure returned documents respect requested metadata filters."""

    for document in documents:
        metadata = document.metadata

        if "file_type" in filters:
            assert (
                metadata.get("file_type")
                == filters["file_type"]
            ), f"Unexpected file type: {metadata}"

        if "source" in filters:
            assert (
                metadata.get("source")
                == filters["source"]
            ), f"Unexpected source: {metadata}"


def print_results(method_name, documents, filters):
    print(f"\n{method_name} returned {len(documents)} chunks:")

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        file_type = document.metadata.get("file_type", "unknown")

        print(f"\n  Result {index}")
        print(f"  Source: {source}")
        print(f"  File type: {file_type}")
        print(f"  Text: {document.page_content[:250]!r}")

    check_filters(documents, filters)
    print("  Metadata filter check: PASSED")


def main():
    for case in TEST_CASES:
        name = case["name"]
        question = case["question"]
        filters = case["filters"]

        print("\n" + "=" * 70)
        print(f"TEST: {name}")
        print(f"QUESTION: {question}")
        print(f"FILTERS: {filters or 'None'}")

        # 1. Test the existing semantic retriever.
        semantic_retriever = get_retriever(
            k=4,
            **filters,
        )
        semantic_documents = semantic_retriever.invoke(question)

        print_results(
            "Semantic search",
            semantic_documents,
            filters,
        )

        # 2. Test the new BM25 keyword retriever.
        bm25_retriever = get_bm25_retriever(
            k=4,
            **filters,
        )
        bm25_documents = bm25_retriever.invoke(question)

        print_results(
            "BM25 keyword search",
            bm25_documents,
            filters,
        )

    print("\n" + "=" * 70)
    print("All retriever tests completed successfully.")


if __name__ == "__main__":
    main()
