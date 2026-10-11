
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from app.rag.hybrid_retriever import get_hybrid_retriever


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
    """Build context with source labels for Gemini."""
    context_parts = []

    for index, document in enumerate(documents, start=1):
        source_name = document.metadata.get("source", "unknown")

        context_parts.append(
            f"[Source {index}: {source_name}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(context_parts)


def ask_rag(
    question: str,
    k: int = 3,
    file_type: str | None = None,
    source: str | None = None,
):
    """
    Hybrid RAG pipeline:
    Semantic search + BM25 -> RRF -> Gemini -> answer and sources.
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not configured.")

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    # Step 1: Retrieve the best chunks using hybrid search.
    documents = get_hybrid_retriever(
        query=question,
        k=k,
        fetch_k=10,
        file_type=file_type,
        source=source,
        rrf_k=60,
        max_distance=1.0,
    )

    if not documents:
        return {
            "answer": (
                "I could not find relevant information "
                "in the enterprise documents."
            ),
            "sources": [],
        }

    # Step 2: Build context from the selected chunks.
    context = build_context(documents)

    # Step 3: Ask Gemini for a concise, grounded answer.
    prompt = f"""
You are an Enterprise Operations Intelligence Assistant.

Answer the user's question using only the supplied
enterprise document context.

Rules:
1. Answer the exact question directly.
2. Keep the answer concise and easy to understand.
3. For simple questions, prefer one or two sentences.
4. For policy questions, use short bullet points when
   multiple important conditions must be explained.
5. Include important numbers, deadlines, approvals,
   and exceptions when relevant to the question.
6. Do not repeat the same information.
7. Do not invent policies, numbers, or facts.
8. If the context does not contain the answer, clearly
   say that the information was not found.
9. Do not claim that a source supports a fact unless
   its supplied content supports that fact.

---------------- ENTERPRISE CONTEXT ----------------

{context}

---------------- USER QUESTION ----------------

{question}

---------------- ANSWER ----------------

Provide the concise answer now.
"""

    # Step 4: Generate the answer using Gemini.
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0,
    )

    response = llm.invoke(prompt)
    answer = extract_text_from_response(response)

    # Step 5: List sources represented in the final context.
    sources = []

    for document in documents:
        source_name = document.metadata.get("source", "unknown")

        if source_name not in sources:
            sources.append(source_name)

    return {
        "answer": answer,
        "sources": sources,
    }
