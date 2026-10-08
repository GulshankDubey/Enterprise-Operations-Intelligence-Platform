from app.rag.vectorstore import load_vectorstore


def get_retriever(
    k: int = 4,
):
    """
    Return a ChromaDB retriever.

    Args:
        k: Number of relevant chunks to retrieve.

    Returns:
        LangChain retriever.
    """

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
        },
    )

    return retriever