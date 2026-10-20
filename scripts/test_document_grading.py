#day 12
from langchain_core.documents import Document

from app.rag.grade_documents import (
    grade_documents,
    assess_evidence,
)


def main():
    question = (
        "What is the company's annual leave carry-forward policy?"
    )

    test_documents = [
        Document(
            page_content=(
                "Employees may carry forward up to 5 unused "
                "annual leave days into the next calendar year."
            ),
            metadata={"source": "leave_policy.docx"},
        ),
        Document(
            page_content=(
                "Employees must use strong passwords and "
                "enable multi-factor authentication."
            ),
            metadata={"source": "it_policy.txt"},
        ),
        Document(
            page_content=(
                "Employees receive annual leave according to "
                "company policy. The carry-forward limit and "
                "approval conditions are not stated here."
            ),
            metadata={"source": "employee_handbook.pdf"},
        ),
    ]

    print("=" * 65)
    print("DAY 12: DOCUMENT GRADING TEST")
    print("=" * 65)
    print(f"\nQuestion: {question}")

    relevant = grade_documents(
        question=question,
        documents=test_documents,
    )

    print("\nRelevant documents:")
    for document in relevant:
        print("-", document.metadata["source"])

    evidence = assess_evidence(
        question=question,
        documents=relevant,
    )

    print("\nEvidence sufficient:", evidence["sufficient"])
    print("Reason:", evidence["reason"])

    print("\nTest finished.")
    print(
        "Review the judgments against the document contents; "
        "the model's decisions are not guaranteed."
    )


if __name__ == "__main__":
    main()
