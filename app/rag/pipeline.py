
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from app.rag.hybrid_retriever import get_hybrid_retriever
from app.rag.reranker import rerank_documents
from app.rag.query_rewriter import rewrite_query


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANSWER_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
)


def extract_text_from_response(response):
    """Extract plain text from Gemini's response."""
    content = response.content

    if isinstance(content, str):
        return content.strip()

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

        return "\n".join(text_parts).strip()

    return str(content).strip()


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
    chat_history: str | None = None,
):
    """
    Day 11: Query rewriting + existing hybrid RAG pipeline.

    Original question
        -> Query rewriting using conversation history
        -> Hybrid candidate retrieval
        -> Cross-encoder reranking
        -> Context construction
        -> Gemini answer using the original question
        -> Answer and sources
    """

    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not configured.")

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    # Preserve the exact question for final answer generation.
    original_question = question.strip()

    # Day 11: Rewrite the question into a standalone search query.
    # If rewriting fails, rewrite_query() should return the
    # original question as a fallback.
    search_query = rewrite_query(
        question=original_question,
        chat_history=chat_history,
    )

    print(f"Original question: {original_question}")
    print(f"Rewritten query:   {search_query}")

    # Existing Day 8: Hybrid retrieval.
    # Only the retrieval query changes; existing settings remain.
    candidates = get_hybrid_retriever(
        query=search_query,
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
            "original_question": original_question,
            "rewritten_query": search_query,
        }

    # Existing Day 9: Cross-encoder reranking.
    # Use the same rewritten query to rank documents against
    # the search intent used during retrieval.
    documents = rerank_documents(
        query=search_query,
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
            "original_question": original_question,
            "rewritten_query": search_query,
        }

    # Build context only from the final selected chunks.
    context = build_context(documents)

    # Important: answer the user's original question, not
    # the rewritten search query.
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

{original_question}

Answer using only the supplied context.
"""

    llm = ChatGoogleGenerativeAI(
        model=ANSWER_MODEL,
        temperature=0,
    )

    response = llm.invoke(prompt)
    answer = extract_text_from_response(response)

    # Sources come from the final reranked documents.
    sources = []

    for document in documents:
        source_name = document.metadata.get("source", "unknown")

        if source_name not in sources:
            sources.append(source_name)

    return {
        "answer": answer,
        "sources": sources,
        "original_question": original_question,
        "rewritten_query": search_query,
    }
