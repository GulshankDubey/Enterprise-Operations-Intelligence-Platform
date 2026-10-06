from app.ingestion.processor import ingest_directory


def main():
    data_directory = "data/raw"

    print("=" * 60)
    print("ENTERPRISE DOCUMENT INGESTION TEST")
    print("=" * 60)

    documents = ingest_directory(data_directory)

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)

    print(f"Total documents loaded: {len(documents)}")

    if documents:
        print("\nFirst document:")
        print("-" * 60)

        print("Content:")
        print(documents[0].page_content[:500])

        print("\nMetadata:")
        print(documents[0].metadata)

    print("\n✓ Ingestion successful!")


if __name__ == "__main__":
    main()