
from app.rag.vectorstore import load_vectorstore


def get_retriever(
    k: int = 4,
    file_type: str | None = None,
    source: str | None = None,
):
    """
    Return a ChromaDB retriever with optional metadata filters.

    Args:
        k: Number of relevant chunks to retrieve.
        file_type: Optional type: pdf, docx, txt, or csv.
        source: Optional source filename.

    Returns:
        LangChain retriever.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")

    allowed_types = {"pdf", "docx", "txt", "csv"}

    if file_type is not None:
        file_type = file_type.lower().lstrip(".")

        if file_type not in allowed_types:
            raise ValueError(
                f"Unsupported file_type: {file_type}"
            )

    vectorstore = load_vectorstore()

    conditions = []

    if file_type is not None:
        conditions.append({"file_type": file_type})

    if source is not None:
        conditions.append({"source": source})

    if len(conditions) == 1:
        metadata_filter = conditions[0]
    elif len(conditions) > 1:
        metadata_filter = {"$and": conditions}
    else:
        metadata_filter = None

    search_kwargs = {"k": k}

    if metadata_filter is not None:
        search_kwargs["filter"] = metadata_filter

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )
