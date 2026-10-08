from app.rag.retriever import get_retriever


def main():

    print("=" * 60)
    print("ENTERPRISE RAG - RETRIEVAL TEST")
    print("=" * 60)

    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:
        print("Question cannot be empty.")
        return

    print("\nSearching ChromaDB...")

    retriever = get_retriever(k=4)

    documents = retriever.invoke(question)

    print(
        f"\nRetrieved {len(documents)} relevant chunks."
    )

    print("\n" + "=" * 60)
    print("RETRIEVED CONTEXT")
    print("=" * 60)

    for index, document in enumerate(
        documents,
        start=1,
    ):

        print(
            f"\n--- Result {index} ---"
        )

        print(
            f"Source: "
            f"{document.metadata.get('source', 'unknown')}"
        )

        print(
            f"Content:\n"
            f"{document.page_content}"
        )

        print(
            f"Metadata: "
            f"{document.metadata}"
        )


if __name__ == "__main__":
    main()