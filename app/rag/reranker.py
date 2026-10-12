
from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_cross_encoder() -> CrossEncoder:
    """
    Load the model once and reuse it across reranking calls.
    """
    return CrossEncoder(RERANKER_MODEL)


def rerank_documents(
    query: str,
    documents: list[Document],
    top_k: int = 5,
) -> list[Document]:
    """
    Reorder retrieved chunks by cross-encoder relevance score.

    Args:
        query: User's question.
        documents: Candidate chunks from hybrid retrieval.
        top_k: Number of final chunks to return.

    Returns:
        The top-k chunks ordered by reranker score.
    """
    if not query or not query.strip():
        raise ValueError("query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if not documents:
        return []

    # Do not ask the model to rerank more chunks than exist.
    top_k = min(top_k, len(documents))

    model = get_cross_encoder()

    # Each pair contains the question and one candidate chunk.
    pairs = [
        (query, document.page_content)
        for document in documents
    ]

    scores = model.predict(
        pairs,
        show_progress_bar=False,
    )

    # Sort indices by score, highest first.
    ranked_indices = sorted(
        range(len(documents)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )

    reranked_documents = []

    for index in ranked_indices[:top_k]:
        original = documents[index]

        # Preserve existing metadata and add the reranker score.
        metadata = dict(original.metadata or {})
        metadata["rerank_score"] = float(scores[index])

        reranked_documents.append(
            Document(
                page_content=original.page_content,
                metadata=metadata,
            )
        )

    return reranked_documents
