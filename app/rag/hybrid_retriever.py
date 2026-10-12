
from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.documents import Document

from app.rag.bm25_retriever import (
    build_metadata_filter,
    get_bm25_retriever,
)
from app.rag.vectorstore import load_vectorstore


DEFAULT_RRF_K = 60
DEFAULT_FETCH_K = 10
DEFAULT_FINAL_K = 4
DEFAULT_MAX_DISTANCE = 1.0


def _document_key(document: Document) -> str:
    """Create a stable identity for a document chunk."""
    metadata = document.metadata or {}

    for field in ("chunk_id", "id"):
        value = metadata.get(field)
        if value:
            return str(value)

    identity = "\n".join(
        [
            str(metadata.get("source", "")),
            str(metadata.get("page", "")),
            str(metadata.get("chunk_index", "")),
            document.page_content.strip(),
        ]
    )

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    rrf_k: int = DEFAULT_RRF_K,
) -> list[Document]:
    """Fuse ranked lists using Reciprocal Rank Fusion."""
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1.")

    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    first_seen: dict[str, int] = {}
    sequence = 0

    for ranked_documents in ranked_lists:
        seen_in_list: set[str] = set()

        for rank, document in enumerate(
            ranked_documents, start=1
        ):
            key = _document_key(document)

            if key in seen_in_list:
                continue

            seen_in_list.add(key)
            sequence += 1

            scores[key] = scores.get(key, 0.0) + (
                1.0 / (rrf_k + rank)
            )

            if key not in documents:
                documents[key] = document
                first_seen[key] = sequence

    ranked_keys = sorted(
        scores,
        key=lambda key: (
            -scores[key],
            first_seen[key],
        ),
    )

    return [documents[key] for key in ranked_keys]


def get_hybrid_retriever(
    query: str,
    k: int = DEFAULT_FINAL_K,
    fetch_k: int = DEFAULT_FETCH_K,
    file_type: str | None = None,
    source: str | None = None,
    rrf_k: int = DEFAULT_RRF_K,
    max_distance: float | None = DEFAULT_MAX_DISTANCE,
) -> list[Document]:
    """
    Retrieve candidates using semantic search and BM25,
    combine rankings with RRF, and return up to k chunks.

    max_distance applies to semantic search only.
    BM25 candidates are not filtered by semantic distance.
    """
    if not query or not query.strip():
        raise ValueError("query cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    if fetch_k < k:
        raise ValueError(
            "fetch_k must be greater than or equal to k."
        )

    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1.")

    if max_distance is not None and max_distance < 0:
        raise ValueError(
            "max_distance must be non-negative or None."
        )

    # Apply identical metadata scope to both retrieval methods.
    metadata_filter = build_metadata_filter(
        file_type=file_type,
        source=source,
    )

    # 1. Semantic retrieval.
    vectorstore = load_vectorstore()

    search_kwargs: dict[str, Any] = {"k": fetch_k}

    if metadata_filter is not None:
        search_kwargs["filter"] = metadata_filter

    semantic_results = (
        vectorstore.similarity_search_with_score(
            query,
            **search_kwargs,
        )
    )

    semantic_documents: list[Document] = []

    for document, distance in semantic_results:
        if (
            max_distance is None
            or distance <= max_distance
        ):
            semantic_documents.append(document)

    # 2. BM25 keyword retrieval.
    keyword_documents: list[Document] = []

    try:
        bm25_retriever = get_bm25_retriever(
            k=fetch_k,
            file_type=file_type,
            source=source,
        )
        keyword_documents = bm25_retriever.invoke(query)

    except ValueError as error:
        expected_message = (
            "No documents found for the requested "
            "BM25 search scope."
        )

        if expected_message not in str(error):
            raise

    # 3. Fuse rankings and deduplicate chunks.
    fused_documents = reciprocal_rank_fusion(
        [semantic_documents, keyword_documents],
        rrf_k=rrf_k,
    )

    # 4. Return the requested number of candidates.
    # The reranker will perform the next relevance-selection step.
    return fused_documents[:k]
