from app.ingestion.processor import ingest_directory
from app.rag.splitter import split_documents
from app.rag.vectorstore import (
    create_vectorstore,
    get_vectorstore_count,
)


def main():
    print("Step 1: Loading source documents...")

    documents = ingest_directory("data/raw")

    if not documents:
        raise ValueError(
            "No documents found in data/raw."
        )

    print(f"Documents loaded: {len(documents)}")

    print("\nStep 2: Splitting documents into chunks...")

    chunks = split_documents(documents)

    # Assign a repeatable position to each chunk within
    # its source document and page.
    chunk_positions = {}

    for chunk in chunks:
        metadata = chunk.metadata

        source = (
            metadata.get("source")
            or metadata.get("file_path")
            or metadata.get("filename")
            or "unknown"
        )
        page = metadata.get("page", "")

        position_key = (source, page)

        chunk_index = chunk_positions.get(position_key, 0)
        metadata["chunk_index"] = chunk_index

        chunk_positions[position_key] = chunk_index + 1

    print(f"Chunks created: {len(chunks)}")

    print("\nStep 3: Upserting chunks into ChromaDB...")

    vectorstore = create_vectorstore(chunks)

    print("\nStep 4: Checking database count...")

    count = get_vectorstore_count()

    print(f"ChromaDB records stored: {count}")
    print("Vector store build completed successfully.")


if __name__ == "__main__":
    main()
