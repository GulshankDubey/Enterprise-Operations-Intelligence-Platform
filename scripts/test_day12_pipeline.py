
from app.rag.pipeline import ask_rag


def main():
    test_cases = [
        {
            "name": "Annual leave",
            "question": "What is the annual leave policy?",
            "history": None,
        },
        {
            "name": "Annual leave follow-up",
            "question": "What about carrying it forward?",
            "history": (
                "User: What is the annual leave policy?\n"
                "Assistant: The policy describes annual leave rules."
            ),
        },
        {
            "name": "Travel expenses",
            "question": "What is the travel expense policy?",
            "history": None,
        },
        {
            "name": "IT password policy",
            "question": "What are the password requirements?",
            "history": None,
        },
    ]

    for index, case in enumerate(test_cases, start=1):
        print("\n" + "=" * 65)
        print(f"TEST {index}: {case['name']}")
        print("=" * 65)

        result = ask_rag(
            question=case["question"],
            chat_history=case["history"],
            k=3,
        )

        print("\nOriginal question:")
        print(result["original_question"])

        print("\nRewritten query:")
        print(result["rewritten_query"])

        print("\nAnswer:")
        print(result["answer"])

        print("\nSources:")
        for source_name in result["sources"]:
            print("-", source_name)

        print(
            "\nEvidence sufficient:",
            result["evidence_sufficient"],
        )
        print(
            "Retry performed:",
            result["retry_performed"],
        )
        print(
            "Relevant document count:",
            result["relevant_document_count"],
        )
        print("Evidence assessment:", result["evidence_reason"])


if __name__ == "__main__":
    main()
