
from typing import Any

from langchain_core.documents import Document

from app.rag.vectorstore import load_vectorstore


class QualityRetriever:
    """Retrieve relevant, unique document chunks from ChromaDB."""

    def __init__(
        self,
        k: int = 4,
        file_type: str | None = None,
        source: str | None = None,
        fetch_k: int = 12,
        max_distance: float = 1.0,
    ) -> None:
        if k < 1:
            raise ValueError("k must be at least 1.")

        if fetch_k < k:
            raise ValueError(
                "fetch_k must be greater than or equal to k."
            )

        if max_distance < 0:
            raise ValueError("max_distance must be non-negative.")

        self.vectorstore = load_vectorstore()
        self.k = k
        self.fetch_k = fetch_k
        self.max_distance = max_distance

        # Preserve Day 5 metadata filtering.
        conditions: list[dict[str, Any]] = []

        if file_type:
            conditions.append({"file_type": file_type.lower()})

        if source:
            conditions.append({"source": source})

        if not conditions:
            self.metadata_filter = None
        elif len(conditions) == 1:
            self.metadata_filter = conditions[0]
        else:
            self.metadata_filter = {"$and": conditions}

    def invoke(self, query: str) -> list[Document]:
        """Return the closest unique chunks within the distance limit."""

        if not query or not query.strip():
            return []

        search_kwargs: dict[str, Any] = {
            "k": self.fetch_k,
        }

        if self.metadata_filter is not None:
            search_kwargs["filter"] = self.metadata_filter

        # Raw distance: smaller values indicate closer matches.
        results = self.vectorstore.similarity_search_with_score(
            query,
            **search_kwargs,
        )

        selected: list[Document] = []
        seen: set[tuple[str, str]] = set()

        for document, distance in results:
            if distance > self.max_distance:
                continue

            source_name = str(
                document.metadata.get("source", "")
            ).strip()

            content = document.page_content.strip()
            content_key = (source_name, content)

            # Remove exact duplicate content from the same source.
            if content_key in seen:
                continue

            seen.add(content_key)
            selected.append(document)

            if len(selected) >= self.k:
                break

        return selected


def get_retriever(
    k: int = 4,
    file_type: str | None = None,
    source: str | None = None,
    fetch_k: int = 12,
    max_distance: float = 1.0,
) -> QualityRetriever:
    """Create a retriever with optional metadata filters."""

    return QualityRetriever(
        k=k,
        file_type=file_type,
        source=source,
        fetch_k=fetch_k,
        max_distance=max_distance,
    )
