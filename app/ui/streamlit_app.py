import os

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise Operations Intelligence",
    page_icon="🏢",
    layout="wide",
)


# --------------------------------------------------
# Check API key
# --------------------------------------------------

if not GOOGLE_API_KEY:
    st.error(
        "GOOGLE_API_KEY is not configured. "
        "Please check your .env file."
    )
    st.stop()


# --------------------------------------------------
# Initialize Gemini
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
)


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def extract_text_from_response(response):
    """
    Extract plain text from Gemini's response.

    Newer versions of langchain-google-genai may return
    response.content as a list of structured content blocks.
    """

    content = response.content

    # Case 1: Normal string response
    if isinstance(content, str):
        return content

    # Case 2: Structured list response
    if isinstance(content, list):

        text_parts = []

        for item in content:

            # Example:
            # {"type": "text", "text": "Machine Learning is..."}

            if isinstance(item, dict):

                if item.get("type") == "text":
                    text = item.get("text", "")

                    if text:
                        text_parts.append(text)

            # Sometimes content can contain plain strings
            elif isinstance(item, str):

                text_parts.append(item)

        return "\n".join(text_parts)

    # Case 3: Unexpected response format
    return str(content)


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🏢 Enterprise Operations Intelligence Platform")

st.write(
    "Ask a question and get an answer from Gemini."
)

st.divider()


# --------------------------------------------------
# Question input
# --------------------------------------------------

question = st.text_area(
    "Ask your question",
    placeholder="Example: What are the benefits of remote work?",
    height=120,
)


# --------------------------------------------------
# Ask Gemini
# --------------------------------------------------

if st.button("Ask Gemini", type="primary"):

    # Validate question
    if not question.strip():

        st.warning("Please enter a question.")
        st.stop()

    # Call Gemini
    with st.spinner("Gemini is thinking..."):

        try:

            response = llm.invoke(question)

            # Extract clean text
            answer = extract_text_from_response(response)

            # Display answer
            st.subheader("Answer")

            st.markdown(answer)

        except Exception as e:

            st.error(
                f"An error occurred while calling Gemini: {e}"
            )