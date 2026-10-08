from app.ingestion.processor import ingest_directory
from app.rag.splitter import split_documents
from app.rag.vectorstore import create_vectorstore


def main():

    print("=" * 60)
    print("ENTERPRISE RAG - VECTOR DATABASE BUILD")
    print("=" * 60)

    # --------------------------------------------------
    # Step 1: Load documents
    # --------------------------------------------------

    print("\n[1/3] Loading documents...")

    documents = ingest_directory("data/raw")

    print(
        f"Loaded {len(documents)} document(s)."
    )

    # --------------------------------------------------
    # Step 2: Split documents
    # --------------------------------------------------

    print("\n[2/3] Splitting documents into chunks...")

    chunks = split_documents(documents)

    print(
        f"Created {len(chunks)} chunks."
    )

    # --------------------------------------------------
    # Step 3: Create vector database
    # --------------------------------------------------

    print("\n[3/3] Creating ChromaDB...")

    vectorstore = create_vectorstore(chunks)

    print("\n" + "=" * 60)
    print("VECTOR DATABASE CREATED SUCCESSFULLY")
    print("=" * 60)

    print(f"Collection: enterprise_documents")
    print(f"Chunks stored: {len(chunks)}")
    print("Location: data/chroma_db")

    # Prevent unused-variable warnings
    _ = vectorstore


if __name__ == "__main__":
    main()