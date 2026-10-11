
from app.rag.retriever import get_retriever


TEST_QUESTIONS = [
    "What is the annual leave policy?",
    "What is the travel expense limit?",
    "What does the IT policy say about passwords?",
]


def main() -> None:
    retriever = get_retriever(
        k=4,
        fetch_k=12,
        max_distance=1.0,
    )

    for question in TEST_QUESTIONS:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        documents = retriever.invoke(question)

        if not documents:
            print("No documents passed the distance threshold.")
            print("Try inspecting raw distances and tuning the limit.")
            continue

        for index, document in enumerate(documents, start=1):
            print(f"\nResult {index}")
            print("Source:", document.metadata.get("source", "Unknown"))
            print("File type:", document.metadata.get("file_type", "Unknown"))
            print("Content:", document.page_content[:500])


if __name__ == "__main__":
    main()
