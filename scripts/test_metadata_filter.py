
from app.rag.retriever import get_retriever


def main():
    test_cases = [
        ("PDF", {"file_type": "pdf"}),
        ("CSV", {"file_type": "csv"}),
        ("TXT", {"file_type": "txt"}),
        ("DOCX", {"file_type": "docx"}),
        (
            "Specific source",
            {"source": "leave_policy.docx"},
        ),
        (
            "Combined filter",
            {
                "file_type": "docx",
                "source": "leave_policy.docx",
            },
        ),
    ]

    for name, filters in test_cases:
        print(f"\n{'=' * 60}")
        print(f"TEST: {name}")
        print(f"FILTERS: {filters}")

        retriever = get_retriever(k=10, **filters)
        documents = retriever.invoke(
            "What is the annual leave policy?"
        )

        print(f"Retrieved chunks: {len(documents)}")

        for document in documents:
            metadata = document.metadata
            print(
                f"Source: {metadata.get('source')} | "
                f"Type: {metadata.get('file_type')}"
            )

            assert metadata.get("source"), (
                "Retrieved document is missing source metadata."
            )

            if "file_type" in filters:
                assert (
                    metadata.get("file_type")
                    == filters["file_type"]
                ), "A document with the wrong file type was returned."

            if "source" in filters:
                assert (
                    metadata.get("source")
                    == filters["source"]
                ), "A document from the wrong source was returned."

        if not documents:
            print("WARNING: No chunks returned for this filter.")

    print("\nMetadata filter tests completed.")


if __name__ == "__main__":
    main()
