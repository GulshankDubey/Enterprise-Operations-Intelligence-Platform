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

st.title(
    "🏢 Enterprise Operations Intelligence Platform"
)

st.write(
    "Ask questions about enterprise policies and "
    "documents using Retrieval-Augmented Generation."
)

st.divider()


# --------------------------------------------------
# System status
# --------------------------------------------------

st.success(
    "RAG system is ready. "
    "Questions will be answered using enterprise documents."
)


# --------------------------------------------------
# Question input
# --------------------------------------------------

question = st.text_area(
    "Ask an enterprise question",
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
):

    # ----------------------------------------------
    # Validate question
    # ----------------------------------------------

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()

    # ----------------------------------------------
    # Run RAG
    # ----------------------------------------------

    with st.spinner(
        "Searching enterprise knowledge..."
    ):

        try:

            result = ask_rag(
                question=question,
                k=4,
            )

            # --------------------------------------
            # Answer
            # --------------------------------------

            st.subheader("Answer")

            st.markdown(
                result["answer"]
            )

            # --------------------------------------
            # Sources
            # --------------------------------------

            if result["sources"]:

                st.subheader(
                    "Sources"
                )

                for source in result["sources"]:

                    st.write(
                        f"📄 {source}"
                    )

            else:

                st.info(
                    "No enterprise document "
                    "sources were retrieved."
                )

        except Exception as e:

            st.error(
                "An error occurred while processing "
                f"your question: {e}"
            )