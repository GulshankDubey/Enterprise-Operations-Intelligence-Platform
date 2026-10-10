
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from app.rag.retriever import get_retriever


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def extract_text_from_response(response):
    """
    Extract plain text from Gemini's response.

    Handles string and structured content responses.
    """
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text", "")

                    if text:
                        text_parts.append(text)

            elif isinstance(item, str):
                text_parts.append(item)

        return "\n".join(text_parts)

    return str(content)


def build_context(documents):
    """
    Convert retrieved documents into a context string.
    """
    context_parts = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")

        context_parts.append(
            f"[Source {index}: {source}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(context_parts)


def ask_rag(
    question: str,
    k: int = 4,
    file_type: str | None = None,
    source: str | None = None,
):
    """
    Complete RAG pipeline:

    Question
        -> Metadata-filtered retriever
        -> Relevant documents
        -> Context
        -> Gemini
        -> Answer and sources

    Args:
        question: User's question.
        k: Maximum number of chunks to retrieve.
        file_type: Optional file type filter.
        source: Optional source filename filter.

    Returns:
        Dictionary containing the answer and source filenames.
    """
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is not configured."
        )

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    # Step 1: Retrieve relevant documents.
    retriever = get_retriever(
        k=k,
        file_type=file_type,
        source=source,
    )

    documents = retriever.invoke(question)

    if not documents:
        return {
            "answer": (
                "I could not find relevant information "
                "in the enterprise documents."
            ),
            "sources": [],
        }

    # Step 2: Build context from retrieved documents.
    context = build_context(documents)

    # Step 3: Prepare the prompt.
    prompt = f"""
You are an Enterprise Operations Intelligence Assistant.

Answer the user's question using ONLY the supplied
enterprise document context.

If the answer is not available in the context, state
clearly that the information was not found in the
available enterprise documents.

Do not invent policies, rules, numbers, or facts.

Keep the answer clear and professional.

---------------- ENTERPRISE CONTEXT ----------------

{context}

---------------- USER QUESTION ----------------

{question}

---------------- INSTRUCTIONS ----------------

Answer the user's question using the enterprise
context above.
"""

    # Step 4: Generate an answer using Gemini.
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0,
    )

    response = llm.invoke(prompt)

    answer = extract_text_from_response(response)

    # Step 5: Collect unique source filenames.
    sources = []

    for document in documents:
        source_name = document.metadata.get(
            "source",
            "unknown",
        )

        if source_name not in sources:
            sources.append(source_name)

    return {
        "answer": answer,
        "sources": sources,
    }
