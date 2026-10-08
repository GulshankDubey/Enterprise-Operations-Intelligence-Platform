from pathlib import Path

from langchain_chroma import Chroma

from app.rag.embeddings import get_embedding_model


CHROMA_DIR = Path("data/chroma_db")

COLLECTION_NAME = "enterprise_documents"


def create_vectorstore(documents):
    """
    Create a ChromaDB vector store from document chunks.
    """

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    return vectorstore


def load_vectorstore():
    """
    Load the existing ChromaDB vector store.
    """

    embeddings = get_embedding_model()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    return vectorstore