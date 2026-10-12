
from langchain_core.documents import Document

from app.rag.reranker import rerank_documents


def main() -> None:
    query = "What is the travel expense limit?"

    documents = [
        Document(
            page_content=(
                "Employees must not share company passwords."
            ),
            metadata={"source": "it_policy.txt"},
        ),
        Document(
            page_content=(
                "category: Travel\n"
                "limit: 5000\n"
                "approval_required: Yes"
            ),
            metadata={"source": "expense_policy.csv"},
        ),
        Document(
            page_content=(
                "Unused annual leave may be carried forward."
            ),
            metadata={"source": "leave_policy.docx"},
        ),
    ]

    results = rerank_documents(
        query=query,
        documents=documents,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].metadata["source"] == "expense_policy.csv"
    assert "rerank_score" in results[0].metadata

    print("Reranker test passed.")
    print("\nRanked results:")

    for rank, document in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Source: {document.metadata['source']}")
        print(f"Score: {document.metadata['rerank_score']:.4f}")
        print(f"Content: {document.page_content}")


if __name__ == "__main__":
    main()
