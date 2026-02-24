from anthropic import Anthropic
from llm.base import LLM
import os

class ClaudeLLM(LLM):
    def __init__(self, model="claude-3-haiku-20240307"):
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def chat(self, system_prompt, user_prompt):
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        return resp.content[0].text
