
from langchain_core.documents import Document

from app.rag.hybrid_retriever import reciprocal_rank_fusion


def test_shared_document_gets_contributions_from_both_lists():
    shared = Document(
        page_content="Annual leave is 24 days.",
        metadata={"source": "leave_policy.docx", "chunk_index": 0},
    )
    semantic_only = Document(
        page_content="Employees should request leave in advance.",
        metadata={"source": "leave_policy.docx", "chunk_index": 1},
    )
    keyword_only = Document(
        page_content="Leave approval is required.",
        metadata={"source": "leave_policy.docx", "chunk_index": 2},
    )

    result = reciprocal_rank_fusion(
        [
            [shared, semantic_only],
            [shared, keyword_only],
        ],
        rrf_k=60,
    )

    assert len(result) == 3
    assert result[0].page_content == shared.page_content


def test_duplicate_in_one_list_is_counted_once():
    first = Document(
        page_content="Travel expenses are limited to 5000.",
        metadata={"source": "expense_policy.csv", "chunk_index": 0},
    )

    result = reciprocal_rank_fusion(
        [[first, first]],
        rrf_k=60,
    )

    assert len(result) == 1


def test_empty_lists_return_empty_result():
    result = reciprocal_rank_fusion([[], []])
    assert result == []


def test_invalid_rrf_constant_is_rejected():
    try:
        reciprocal_rank_fusion([[]], rrf_k=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for rrf_k=0")


if __name__ == "__main__":
    test_shared_document_gets_contributions_from_both_lists()
    test_duplicate_in_one_list_is_counted_once()
    test_empty_lists_return_empty_result()
    test_invalid_rrf_constant_is_rejected()

    print("All RRF logic tests passed.")
