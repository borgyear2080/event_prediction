import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


class LLMs:
    def __init__(self, model_name="gemini-2.5-flash"):
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Check that .env is in the notebook's working directory.")

        self.llm1 = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
        self.llm2 = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    def get(self):
        return self.llm1, self.llm2


llm1, llm2 = LLMs().get()