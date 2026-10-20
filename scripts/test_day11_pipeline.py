
from app.rag.pipeline import ask_rag


def main():
    history = (
        "User: What is the company's annual leave policy?\n"
        "Assistant: The policy explains annual leave rules."
    )

    result = ask_rag(
        question="What about carrying it forward?",
        chat_history=history,
        k=3,
    )

    print("\nOriginal question:")
    print(result["original_question"])

    print("\nRewritten query:")
    print(result["rewritten_query"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source}")


if __name__ == "__main__":
    main()
