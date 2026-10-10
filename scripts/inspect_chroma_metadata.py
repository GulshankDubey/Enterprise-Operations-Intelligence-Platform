
from app.rag.vectorstore import load_vectorstore


def main():
    vectorstore = load_vectorstore()
    collection = vectorstore._collection

    print(f"Total ChromaDB records: {collection.count()}")

    result = collection.get(
        limit=20,
        include=["metadatas"],
    )

    print("\nStored metadata:\n")

    for record_id, metadata in zip(
        result["ids"],
        result["metadatas"],
    ):
        print(f"ID: {record_id}")
        print(f"Metadata: {metadata}")
        print("-" * 60)


if __name__ == "__main__":
    main()
