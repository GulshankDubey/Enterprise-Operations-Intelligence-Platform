from pathlib import Path

from app.ingestion.loaders import (
    load_document,
    SUPPORTED_EXTENSIONS,
)


def ingest_directory(directory: str):
    """
    Load all supported documents from a directory.

    Args:
        directory: Path to the document directory.

    Returns:
        list[Document]: All loaded documents.
    """

    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    all_documents = []

    for file_path in sorted(directory_path.iterdir()):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(
                f"Skipping unsupported file: {file_path.name}"
            )
            continue

        print(f"Loading: {file_path.name}")

        documents = load_document(str(file_path))

        all_documents.extend(documents)

        print(
            f"  Loaded {len(documents)} document(s)"
        )

    return all_documents