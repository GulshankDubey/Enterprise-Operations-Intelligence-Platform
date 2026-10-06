from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import GOOGLE_API_KEY


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)


response = llm.invoke(
    "Explain Enterprise Operations Intelligence Platform in one sentence."
)


print(response.content)