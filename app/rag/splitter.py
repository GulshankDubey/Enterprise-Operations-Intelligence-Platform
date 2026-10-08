from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    """
    Split loaded documents into smaller chunks.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        List of chunked Document objects.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = text_splitter.split_documents(documents)

    return chunks