
from __future__ import annotations

from dataclasses import dataclass

from app.rag.hybrid_retriever import get_hybrid_retriever
from app.rag.reranker import rerank_documents


# ============================================================
# 1. TEST CASE MODEL
# ============================================================

@dataclass(frozen=True)
class TestCase:
    question: str
    expected_source: str
    expected_terms: tuple[str, ...]
    file_type: str | None = None


# ============================================================
# 2. TEST CASES
# ============================================================
# expected_terms must occur in the relevant chunk.
# Verify CSV terms against your actual expense_policy.csv.

TEST_CASES = [
    TestCase(
        question="What is the annual leave policy?",
        expected_source="leave_policy.docx",
        expected_terms=("24 days", "annual leave"),
    ),
    TestCase(
        question="How many days in advance should planned leave be requested?",
        expected_source="leave_policy.docx",
        expected_terms=("three working days",),
    ),
    TestCase(
        question="What happens to unused annual leave?",
        expected_source="leave_policy.docx",
        expected_terms=("unused annual leave", "carried forward"),
    ),
    TestCase(
        question="What is the travel expense limit?",
        expected_source="expense_policy.csv",
        expected_terms=("5000",),
        file_type="csv",
    ),
    TestCase(
        question="Is approval required for food expenses?",
        expected_source="expense_policy.csv",
        expected_terms=("approval_required", "yes"),
        file_type="csv",
    ),
    TestCase(
        question="What does the IT policy say about passwords?",
        expected_source="it_policy.txt",
        expected_terms=("passwords", "share"),
        file_type="txt",
    ),
    TestCase(
        question="How should employees report technical issues?",
        expected_source="it_policy.txt",
        expected_terms=("support portal",),
        file_type="txt",
    ),
]


# ============================================================
# 3. RETRIEVAL CONFIGURATIONS
# ============================================================
# candidate_k: maximum fused candidates requested
# final_k: maximum chunks retained after reranking
# rrf_k: Reciprocal Rank Fusion constant
# max_distance: semantic retrieval distance threshold

CONFIGURATIONS = [
    (5, 3, 60, 1.0),
    (10, 3, 60, 1.0),
    (20, 3, 60, 1.0),
    (10, 5, 60, 1.0),
    (20, 5, 60, 1.0),
]


# ============================================================
# 4. CONTENT-BASED RELEVANCE CHECK
# ============================================================

def is_relevant_document(
    document,
    expected_source: str,
    expected_terms: tuple[str, ...],
) -> bool:
    """
    A chunk is considered relevant if:
    1. It belongs to the expected source file.
    2. All expected terms appear in its content.

    This is a lexical baseline, not a complete semantic metric.
    """

    source = document.metadata.get("source", "")
    content = document.page_content.casefold()

    if source != expected_source:
        return False

    return all(
        term.casefold() in content
        for term in expected_terms
    )


# ============================================================
# 5. EVALUATE ONE CONFIGURATION
# ============================================================

def evaluate_configuration(
    candidate_k: int,
    final_k: int,
    rrf_k: int,
    max_distance: float,
) -> dict:
    hits = 0
    reciprocal_ranks: list[float] = []

    total_candidates = 0
    total_final_chunks = 0

    print("\n" + "-" * 88)
    print(
        f"Testing candidates={candidate_k}, "
        f"final_k={final_k}, rrf_k={rrf_k}, "
        f"max_distance={max_distance}"
    )
    print("-" * 88)

    for case in TEST_CASES:
        print(f"\nQuestion: {case.question}")
        print(f"Expected source: {case.expected_source}")
        print(f"Expected terms: {case.expected_terms}")

        # ----------------------------------------------------
        # A. HYBRID RETRIEVAL
        # ----------------------------------------------------

        candidates = get_hybrid_retriever(
            query=case.question,
            k=candidate_k,
            fetch_k=candidate_k,
            file_type=case.file_type,
            rrf_k=rrf_k,
            max_distance=max_distance,
        )

        total_candidates += len(candidates)

        print(
            f"Requested candidates: {candidate_k}"
            f"\nActual candidates: {len(candidates)}"
        )

        for rank, document in enumerate(candidates, start=1):
            source = document.metadata.get(
                "source", "unknown"
            )

            print(f"  Candidate {rank}: {source}")

        # ----------------------------------------------------
        # B. CROSS-ENCODER RERANKING
        # ----------------------------------------------------

        final_documents = rerank_documents(
            query=case.question,
            documents=candidates,
            top_k=final_k,
        )

        total_final_chunks += len(final_documents)

        print(
            f"Final chunks after reranking: "
            f"{len(final_documents)}"
        )

        # ----------------------------------------------------
        # C. EVALUATE RELEVANCE AND RECIPROCAL RANK
        # ----------------------------------------------------

        first_relevant_rank = None

        for rank, document in enumerate(
            final_documents, start=1
        ):
            source = document.metadata.get(
                "source", "unknown"
            )
            score = document.metadata.get(
                "rerank_score", "N/A"
            )

            relevant = is_relevant_document(
                document=document,
                expected_source=case.expected_source,
                expected_terms=case.expected_terms,
            )

            status = (
                "RELEVANT"
                if relevant
                else "NOT RELEVANT"
            )

            print(
                f"  Final rank {rank}: {source} "
                f"| score={score} | {status}"
            )

            if relevant and first_relevant_rank is None:
                first_relevant_rank = rank

        if first_relevant_rank is not None:
            hits += 1

            reciprocal_ranks.append(
                1.0 / first_relevant_rank
            )
        else:
            reciprocal_ranks.append(0.0)

            print(
                "  WARNING: No relevant chunk found "
                "in the final results."
            )

    # --------------------------------------------------------
    # D. CALCULATE METRICS
    # --------------------------------------------------------

    total_cases = len(TEST_CASES)

    hit_rate = (
        hits / total_cases
        if total_cases
        else 0.0
    )

    mrr = (
        sum(reciprocal_ranks) / total_cases
        if total_cases
        else 0.0
    )

    avg_candidates = (
        total_candidates / total_cases
        if total_cases
        else 0.0
    )

    avg_final_chunks = (
        total_final_chunks / total_cases
        if total_cases
        else 0.0
    )

    return {
        "candidate_k": candidate_k,
        "final_k": final_k,
        "rrf_k": rrf_k,
        "max_distance": max_distance,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "avg_candidates": avg_candidates,
        "avg_final_chunks": avg_final_chunks,
    }


# ============================================================
# 6. FINAL COMPARISON TABLE
# ============================================================

def print_comparison(results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("FINAL RETRIEVAL OPTIMIZATION COMPARISON")
    print("=" * 100)

    print(
        f"{'Candidates':>10} "
        f"{'Final K':>8} "
        f"{'RRF K':>7} "
        f"{'Hit Rate':>10} "
        f"{'MRR':>8} "
        f"{'Avg candidates':>15} "
        f"{'Avg final':>11}"
    )

    for result in results:
        print(
            f"{result['candidate_k']:>10} "
            f"{result['final_k']:>8} "
            f"{result['rrf_k']:>7} "
            f"{result['hit_rate']:>9.2%} "
            f"{result['mrr']:>8.3f} "
            f"{result['avg_candidates']:>15.2f} "
            f"{result['avg_final_chunks']:>11.2f}"
        )

    # Prefer higher MRR, then Hit Rate.
    # When tied, prefer fewer final chunks.
    best = max(
        results,
        key=lambda result: (
            result["mrr"],
            result["hit_rate"],
            -result["avg_final_chunks"],
        ),
    )

    print("\nBest configuration under current metrics:")
    print(f"  Candidate K: {best['candidate_k']}")
    print(f"  Final K: {best['final_k']}")
    print(f"  RRF K: {best['rrf_k']}")
    print(f"  Hit Rate: {best['hit_rate']:.2%}")
    print(f"  MRR: {best['mrr']:.3f}")
    print(
        f"  Average candidates: "
        f"{best['avg_candidates']:.2f}"
    )
    print(
        f"  Average final chunks: "
        f"{best['avg_final_chunks']:.2f}"
    )

    print(
        "\nNote: Expected terms must be verified against "
        "the source documents. These metrics measure "
        "lexical relevance, not full semantic correctness."
    )


# ============================================================
# 7. MAIN FUNCTION
# ============================================================

def main() -> None:
    print("\nDAY 10: RETRIEVAL OPTIMIZATION")
    print("=" * 88)

    print(f"Total test cases: {len(TEST_CASES)}")
    print(f"Configurations to test: {len(CONFIGURATIONS)}")

    results = []

    for candidate_k, final_k, rrf_k, max_distance in CONFIGURATIONS:
        result = evaluate_configuration(
            candidate_k=candidate_k,
            final_k=final_k,
            rrf_k=rrf_k,
            max_distance=max_distance,
        )

        results.append(result)

        print(
            f"\nConfiguration summary: "
            f"Hit Rate={result['hit_rate']:.2%} | "
            f"MRR={result['mrr']:.3f} | "
            f"Average candidates="
            f"{result['avg_candidates']:.2f}"
        )

    print_comparison(results)


if __name__ == "__main__":
    main()
