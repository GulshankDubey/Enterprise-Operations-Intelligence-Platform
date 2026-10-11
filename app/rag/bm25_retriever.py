
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.rag.vectorstore import load_vectorstore


ALLOWED_FILE_TYPES = {"pdf", "csv", "txt", "docx"}


def build_metadata_filter(
    file_type: str | None = None,
    source: str | None = None,
) -> dict | None:
    """Build a ChromaDB metadata filter."""

    conditions = []

    if file_type is not None:
        file_type = file_type.lower().lstrip(".")

        if file_type not in ALLOWED_FILE_TYPES:
            raise ValueError(
                f"Unsupported file_type: {file_type}"
            )

        conditions.append({"file_type": file_type})

    if source is not None:
        if not source.strip():
            raise ValueError("source cannot be empty.")

        conditions.append({"source": source})

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


def get_bm25_retriever(
    k: int = 4,
    file_type: str | None = None,
    source: str | None = None,
) -> BM25Retriever:
    """
    Build a BM25 keyword retriever from stored ChromaDB chunks.

    ChromaDB supplies the text and metadata. BM25 ranks the
    eligible chunks using keyword matching.
    """

    if k < 1:
        raise ValueError("k must be at least 1.")

    vectorstore = load_vectorstore()

    metadata_filter = build_metadata_filter(
        file_type=file_type,
        source=source,
    )

    get_kwargs = {
        "include": ["documents", "metadatas"],
    }

    if metadata_filter is not None:
        get_kwargs["where"] = metadata_filter

    stored_data = vectorstore.get(**get_kwargs)

    documents = []

    for text, metadata in zip(
        stored_data["documents"],
        stored_data["metadatas"],
    ):
        if text is None:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata=metadata or {},
            )
        )

    if not documents:
        raise ValueError(
            "No documents found for the requested BM25 search scope."
        )

    return BM25Retriever.from_documents(
        documents,
        k=k,
    )
