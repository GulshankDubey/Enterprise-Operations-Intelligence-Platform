from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
)


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".docx",
    ".csv",
}


def load_document(file_path: str):
    """
    Load a single document based on its file extension.

    Returns:
        list[Document]: Loaded LangChain documents.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(str(path))

    elif extension == ".txt":
        loader = TextLoader(
            str(path),
            encoding="utf-8"
        )

    elif extension == ".docx":
        loader = Docx2txtLoader(str(path))

    elif extension == ".csv":
        loader = CSVLoader(str(path))

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    documents = loader.load()

    # Add common metadata
    for document in documents:
        document.metadata.update({
            "source": path.name,
            "file_type": extension.replace(".", ""),
        })

    return documents