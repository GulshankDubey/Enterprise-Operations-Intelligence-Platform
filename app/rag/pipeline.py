import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from app.rag.retriever import get_retriever


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def extract_text_from_response(response):
    """
    Extract plain text from Gemini's response.

    Handles both traditional string responses and
    structured content responses.
    """

    content = response.content

    # ----------------------------------------------
    # Normal string
    # ----------------------------------------------

    if isinstance(content, str):
        return content

    # ----------------------------------------------
    # Structured content
    # ----------------------------------------------

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":

                    text = item.get(
                        "text",
                        "",
                    )

                    if text:
                        text_parts.append(text)

            elif isinstance(item, str):

                text_parts.append(item)

        return "\n".join(text_parts)

    # ----------------------------------------------
    # Fallback
    # ----------------------------------------------

    return str(content)


def build_context(documents):
    """
    Convert retrieved documents into a single context string.
    """

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        source = document.metadata.get(
            "source",
            "unknown",
        )

        context_parts.append(
            f"[Source {index}: {source}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(context_parts)


def ask_rag(
    question: str,
    k: int = 4,
):
    """
    Complete RAG pipeline:

    Question
       ↓
    Retriever
       ↓
    Relevant documents
       ↓
    Context
       ↓
    Gemini
       ↓
    Answer
    """

    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is not configured."
        )

    # ----------------------------------------------
    # Step 1: Retriever
    # ----------------------------------------------

    retriever = get_retriever(k=k)

    documents = retriever.invoke(question)

    if not documents:
        return {
            "answer": (
                "I could not find relevant information "
                "in the enterprise documents."
            ),
            "sources": [],
        }

    # ----------------------------------------------
    # Step 2: Build context
    # ----------------------------------------------

    context = build_context(documents)

    # ----------------------------------------------
    # Step 3: Create prompt
    # ----------------------------------------------

    prompt = f"""
You are an Enterprise Operations Intelligence Assistant.

Your job is to answer questions using ONLY the
provided enterprise context.

If the answer is not available in the context,
say clearly that the information was not found
in the available enterprise documents.

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

    # ----------------------------------------------
    # Step 4: Gemini
    # ----------------------------------------------

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0,
    )

    response = llm.invoke(prompt)

    answer = extract_text_from_response(response)

    # ----------------------------------------------
    # Step 5: Sources
    # ----------------------------------------------

    sources = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "unknown",
        )

        if source not in sources:
            sources.append(source)

    return {
        "answer": answer,
        "sources": sources,
    }