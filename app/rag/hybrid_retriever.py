
from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.documents import Document

from app.rag.bm25_retriever import (
    build_metadata_filter,
    get_bm25_retriever,
)
from app.rag.vectorstore import load_vectorstore

import re


STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "what", "which", "who", "when", "where", "why",
    "how", "does", "do", "did", "can", "could",
    "would", "should", "about", "for", "from", "with",
    "and", "or", "to", "of", "in", "on", "by",
    "me", "my", "it", "say", "tell", "please",
}


def meaningful_terms(text: str) -> set[str]:
    """Extract meaningful words from a question or chunk."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    return {
        word for word in words
        if word not in STOP_WORDS and len(word) > 1
    }


def is_relevant_chunk(
    query: str,
    document: Document,
    min_matches: int = 2,
) -> bool:
    """
    Apply a simple lexical relevance check.

    A chunk must share at least min_matches meaningful
    terms with the query.
    """
    query_terms = meaningful_terms(query)
    document_terms = meaningful_terms(document.page_content)

    # If the query is too short, do not reject everything.
    if len(query_terms) < min_matches:
        return True

    matched_terms = query_terms.intersection(document_terms)

    return len(matched_terms) >= min_matches


DEFAULT_RRF_K = 60
DEFAULT_FETCH_K = 10
DEFAULT_FINAL_K = 4
DEFAULT_MAX_DISTANCE = 1.0


def _document_key(document: Document) -> str:
    """Create a stable identity for a retrieved document chunk."""
    metadata = document.metadata or {}

    for key in ("chunk_id", "id"):
        value = metadata.get(key)
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
    """
    Combine ranked lists using Reciprocal Rank Fusion.

    Each unique chunk receives 1 / (rrf_k + rank)
    from each list in which it appears.
    """
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1.")

    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    first_seen: dict[str, int] = {}
    sequence = 0

    for ranked_documents in ranked_lists:
        seen_in_this_list: set[str] = set()

        for rank, document in enumerate(
            ranked_documents, start=1
        ):
            key = _document_key(document)

            if key in seen_in_this_list:
                continue

            seen_in_this_list.add(key)
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
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> list[Document]:
    """
    Combine semantic search and BM25 using RRF.

    Semantic retrieval preserves the Day 6 distance threshold.
    Both retrieval methods use the same metadata filters.
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

    if max_distance < 0:
        raise ValueError(
            "max_distance must be non-negative."
        )

    # Build consistent metadata filters for both methods.
    metadata_filter = build_metadata_filter(
        file_type=file_type,
        source=source,
    )

    # 1. Semantic retrieval with raw ChromaDB distances.
    vectorstore = load_vectorstore()

    search_kwargs: dict[str, Any] = {
        "k": fetch_k,
    }

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
        if distance <= max_distance:
            semantic_documents.append(document)

    # 2. Keyword retrieval using BM25.
    # An empty BM25 scope should not prevent semantic results
    # from being returned.
    keyword_documents: list[Document] = []

    try:
        bm25_retriever = get_bm25_retriever(
            k=fetch_k,
            file_type=file_type,
            source=source,
        )

        keyword_documents = bm25_retriever.invoke(query)

    except ValueError as error:
        if (
            "No documents found for the requested "
            "BM25 search scope."
        ) not in str(error):
            raise

    # 3. Fuse the two ranked lists.
    fused_documents = reciprocal_rank_fusion(
        [
            semantic_documents,
            keyword_documents,
        ],
        rrf_k=rrf_k,
    )


    # 4. Remove weak matches before selecting the final chunks.
    relevant_documents = [
        document
        for document in fused_documents
        if is_relevant_chunk(query, document, min_matches=2)
    ]

    # 5. Return only the best relevant chunks.
    return relevant_documents[:k]
