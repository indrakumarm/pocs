from abc import ABC, abstractmethod

class LLM(ABC):
    @abstractmethod
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        pass
