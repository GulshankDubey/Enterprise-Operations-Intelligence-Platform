
import streamlit as st

from app.rag.pipeline import ask_rag


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise Operations Intelligence",
    page_icon="🏢",
    layout="wide",
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("Enterprise Operations Intelligence Platform")

st.write(
    "Ask questions about enterprise policies and documents "
    "using Retrieval-Augmented Generation (RAG)."
)

st.divider()


# --------------------------------------------------
# System status
# --------------------------------------------------

st.success(
    "RAG system is ready. "
    "Answers are generated using retrieved enterprise documents."
)


# --------------------------------------------------
# Retrieval scope
# --------------------------------------------------

st.subheader("Search Configuration")

scope_options = {
    "All documents": {},
    "PDF documents only": {"file_type": "pdf"},
    "CSV documents only": {"file_type": "csv"},
    "TXT documents only": {"file_type": "txt"},
    "DOCX documents only": {"file_type": "docx"},
    "Employee handbook only": {
        "source": "employee_handbook.pdf"
    },
    "Expense policy only": {
        "source": "expense_policy.csv"
    },
    "IT policy only": {
        "source": "it_policy.txt"
    },
    "Leave policy only": {
        "source": "leave_policy.docx"
    },
}

selected_scope = st.selectbox(
    "Choose which documents to search",
    options=list(scope_options.keys()),
    index=0,
    help=(
        "Choose All documents for general questions, "
        "a file type to search that format, or a specific "
        "policy to restrict retrieval to that file."
    ),
)

filters = scope_options[selected_scope]

if selected_scope == "All documents":
    st.caption(
        "Searching across all supported enterprise documents."
    )
else:
    st.caption(
        f"Retrieval scope: {selected_scope}. "
        "Only matching documents will be considered."
    )


# --------------------------------------------------
# Question input
# --------------------------------------------------

st.subheader("Ask a Question")

question = st.text_area(
    "Enterprise question",
    placeholder=(
        "Example: What is the annual leave policy?"
    ),
    height=120,
)


# --------------------------------------------------
# Ask button
# --------------------------------------------------

if st.button(
    "Ask Enterprise Assistant",
    type="primary",
    use_container_width=True,
):

    # ----------------------------------------------
    # Validate question
    # ----------------------------------------------

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    # ----------------------------------------------
    # Run RAG
    # ----------------------------------------------

    with st.spinner(
        "Searching enterprise documents and generating an answer..."
    ):
        try:
            result = ask_rag(
                question=question.strip(),
                k=4,
                **filters,
            )

            # --------------------------------------
            # Answer
            # --------------------------------------

            st.subheader("Answer")
            st.markdown(result["answer"])

            # --------------------------------------
            # Sources
            # --------------------------------------

            st.subheader("Sources")

            sources = result.get("sources", [])

            if sources:
                for source in sources:
                    st.write(f"- {source}")
            else:
                st.info(
                    "No enterprise document sources were retrieved."
                )

            # --------------------------------------
            # Retrieval details
            # --------------------------------------

            with st.expander("Search details"):
                st.write(f"Selected scope: {selected_scope}")
                st.write(f"Number of sources: {len(sources)}")

                if filters:
                    st.json(filters)
                else:
                    st.write(
                        "No metadata filter applied. "
                        "Searching across all document types."
                    )

        except Exception:
            st.error(
                "An error occurred while processing your question."
            )
            st.exception(
                __import__("sys").exc_info()[1]
            )
