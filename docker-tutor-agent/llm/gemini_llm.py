import google.generativeai as genai
from llm.base import LLM
import os

class GeminiLLM(LLM):
    def __init__(self, model="gemini-1.5-flash"):
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(model)

    def chat(self, system_prompt, user_prompt):
        prompt = f"{system_prompt}\n\nUser: {user_prompt}"
        resp = self.model.generate_content(prompt)
        return resp.text
