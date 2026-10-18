
from app.rag.query_rewriter import rewrite_query


TEST_CASES = [
    {
        "name": "Annual leave follow-up",
        "history": (
            "User: What is the company's annual leave policy?\n"
            "Assistant: The annual leave policy describes "
            "how employees use their annual leave."
        ),
        "question": "What about carrying it forward?",
    },
    {
        "name": "Expense policy follow-up",
        "history": (
            "User: What is the travel expense policy?\n"
            "Assistant: The travel expense policy describes "
            "eligible travel expenses and approvals."
        ),
        "question": "Who needs to approve it?",
    },
    {
        "name": "IT policy follow-up",
        "history": (
            "User: What does the IT policy say about passwords?\n"
            "Assistant: The IT policy contains password-related "
            "security requirements."
        ),
        "question": "What if I forget mine?",
    },
]


def main():
    print("=" * 65)
    print("DAY 11: QUERY REWRITING TEST")
    print("=" * 65)

    for index, case in enumerate(TEST_CASES, start=1):
        print(f"\nTest {index}: {case['name']}")
        print("-" * 65)

        original = case["question"]

        rewritten = rewrite_query(
            question=original,
            chat_history=case["history"],
        )

        print(f"Original question : {original}")
        print(f"Rewritten query   : {rewritten}")

        if rewritten:
            print("Status            : PASS - Query returned")
        else:
            print("Status            : FAIL - Empty query returned")

    print("\n" + "=" * 65)
    print("QUERY REWRITING TEST FINISHED")
    print("=" * 65)


if __name__ == "__main__":
    main()
