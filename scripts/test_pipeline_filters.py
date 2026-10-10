
from app.rag.pipeline import ask_rag


def main():
    test_cases = [
        (
            "General retrieval",
            {},
        ),
        (
            "DOCX-only retrieval",
            {"file_type": "docx"},
        ),
        (
            "Specific source retrieval",
            {"source": "leave_policy.docx"},
        ),
        (
            "Combined metadata filters",
            {
                "file_type": "docx",
                "source": "leave_policy.docx",
            },
        ),
    ]

    question = "What is the annual leave policy?"

    for name, filters in test_cases:
        print("\n" + "=" * 60)
        print(f"TEST: {name}")
        print(f"FILTERS: {filters}")

        result = ask_rag(
            question=question,
            k=4,
            **filters,
        )

        print("\nANSWER:")
        print(result["answer"])

        print("\nSOURCES:")
        for source_name in result["sources"]:
            print(f"- {source_name}")

        if "source" in filters:
            assert all(
                source_name == filters["source"]
                for source_name in result["sources"]
            ), "Unexpected source returned by pipeline."

        print("\nTest completed.")


if __name__ == "__main__":
    main()
