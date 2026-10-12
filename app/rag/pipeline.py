
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from app.rag.hybrid_retriever import get_hybrid_retriever
from app.rag.reranker import rerank_documents


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def extract_text_from_response(response):
    """Extract plain text from Gemini's response."""
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
    """Build context from the final selected chunks."""
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
    k: int = 5,
    file_type: str | None = None,
    source: str | None = None,
):
    """
    Hybrid RAG pipeline with cross-encoder reranking.

    Question
        -> Hybrid candidate retrieval
        -> Cross-encoder reranking
        -> Final context
        -> Gemini answer
        -> Answer and sources
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not configured.")

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    # Day 8: Retrieve a larger candidate pool.
    candidates = get_hybrid_retriever(
        query=question,
        k=20,
        fetch_k=20,
        file_type=file_type,
        source=source,
        rrf_k=60,
        max_distance=1.0,
    )

    if not candidates:
        return {
            "answer": (
                "I could not find relevant information "
                "in the enterprise documents."
            ),
            "sources": [],
        }

    # Day 9: Reorder candidates and keep only the best k.
    documents = rerank_documents(
        query=question,
        documents=candidates,
        top_k=k,
    )

    if not documents:
        return {
            "answer": (
                "I could not find relevant information "
                "in the enterprise documents."
            ),
            "sources": [],
        }

    # Build context only from the final selected chunks.
    context = build_context(documents)

    prompt = f"""
You are an Enterprise Operations Intelligence Assistant.

Answer the user's question using ONLY the supplied
enterprise document context.

Rules:
1. Answer the exact question directly.
2. Be concise and easy to understand.
3. Preserve important numbers, conditions, and exceptions.
4. Do not repeat the same information.
5. Do not invent policies, rules, or facts.
6. If the answer is unavailable in the context, say so.
7. Do not assume a currency if the documents do not specify it.

---------------- ENTERPRISE CONTEXT ----------------

{context}

---------------- USER QUESTION ----------------

{question}

Answer using only the supplied context.
"""

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0,
    )

    response = llm.invoke(prompt)
    answer = extract_text_from_response(response)

    # Sources are derived from the final reranked chunks.
    sources = []

    for document in documents:
        source_name = document.metadata.get("source", "unknown")

        if source_name not in sources:
            sources.append(source_name)

    return {
        "answer": answer,
        "sources": sources,
    }
