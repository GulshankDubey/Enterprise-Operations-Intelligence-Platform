import hashlib
from pathlib import Path

from langchain_chroma import Chroma

from app.rag.embeddings import get_embedding_model


CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "enterprise_documents"


def generate_chunk_id(document) -> str:
    """
    Generate a deterministic ID for a document chunk.

    The same source, page, and chunk position produce the same ID.
    """
    metadata = document.metadata

    source = (
        metadata.get("source")
        or metadata.get("file_path")
        or metadata.get("filename")
        or "unknown"
    )

    page = metadata.get("page", "")
    chunk_index = metadata.get("chunk_index", 0)

    unique_key = f"{source}|{page}|{chunk_index}"

    return hashlib.sha256(
        unique_key.encode("utf-8")
    ).hexdigest()


def create_vectorstore(documents):
    """
    Add or update document chunks in ChromaDB using stable IDs.
    """
    if not documents:
        raise ValueError("No document chunks were provided.")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = get_embedding_model()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    ids = [generate_chunk_id(doc) for doc in documents]

    # Detect accidental duplicate IDs within this batch.
    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate chunk IDs found. Check source metadata "
            "and chunk_index assignment."
        )

    # Chroma upserts these IDs: existing IDs are updated,
    # and new IDs are inserted.
    vectorstore.add_documents(
        documents=documents,
        ids=ids,
    )

    return vectorstore


def load_vectorstore():
    """
    Load the existing persistent ChromaDB collection.
    """
    embeddings = get_embedding_model()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def get_vectorstore_count() -> int:
    """
    Return the number of records in the ChromaDB collection.
    """
    vectorstore = load_vectorstore()
    return vectorstore._collection.count()
