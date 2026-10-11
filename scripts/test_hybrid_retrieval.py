
from app.rag.hybrid_retriever import get_hybrid_retriever


TEST_CASES = [
    {
        "question": "What is the annual leave policy?",
        "file_type": None,
        "source": None,
    },
    {
        "question": "What is the travel expense limit?",
        "file_type": "csv",
        "source": None,
    },
    {
        "question": "What does the IT policy say about passwords?",
        "file_type": "txt",
        "source": None,
    },
    {
        "question": "What is the annual leave policy?",
        "file_type": None,
        "source": "leave_policy.docx",
    },
]


def main() -> None:
    for index, test_case in enumerate(TEST_CASES, start=1):
        question = test_case["question"]
        file_type = test_case["file_type"]
        source = test_case["source"]

        print("\n" + "=" * 80)
        print(f"TEST {index}: {question}")
        print(f"File type filter: {file_type}")
        print(f"Source filter: {source}")

        try:
            documents = get_hybrid_retriever(
                query=question,
                k=4,
                fetch_k=10,
                file_type=file_type,
                source=source,
            )

            if not documents:
                print("No matching documents were retrieved.")
                continue

            for rank, document in enumerate(documents, start=1):
                metadata = document.metadata or {}

                print(f"\nRank: {rank}")
                print(f"Source: {metadata.get('source', 'Unknown')}")
                print(f"File type: {metadata.get('file_type', 'Unknown')}")
                print(f"Chunk: {document.page_content[:350]}")

                # Verify that the requested source/type is respected.
                if file_type is not None:
                    actual_type = str(
                        metadata.get("file_type", "")
                    ).lower().lstrip(".")

                    assert actual_type == file_type, (
                        f"Expected file type {file_type}, "
                        f"received {actual_type}"
                    )

                if source is not None:
                    assert metadata.get("source") == source, (
                        f"Expected source {source}, "
                        f"received {metadata.get('source')}"
                    )

            print("\nTest completed.")

        except Exception as exc:
            print(f"\nTest failed: {type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
