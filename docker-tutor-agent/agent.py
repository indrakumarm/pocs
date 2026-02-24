import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

from llm.claude_llm import ClaudeLLM
from llm.gemini_llm import GeminiLLM

# -----------------------------
# Root directory detection
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parent

# Load .env from root
load_dotenv(ROOT_DIR / ".env")

def main():
    parser = argparse.ArgumentParser(description="Docker Tutor Agent")
    parser.add_argument("--provider", choices=["claude", "gemini"], default="claude")
    parser.add_argument("--model", required=False)
    args = parser.parse_args()
    if args.provider == "claude":
         model = args.model or "claude-3-haiku-20240307"
         llm = ClaudeLLM(model=model)
    elif args.provider == "gemini":
         model = args.model or "gemini-1.5-flash"
         llm = GeminiLLM(model=model)

    # Load system prompt
    SYSTEM_PROMPT = (ROOT_DIR / "prompts/system.txt").read_text()

    print("Docker Tutor Agent")
    print(f"Provider: {args.provider}")
    print(f"Model: {model}")
    print("Type 'exit' to quit\n")

    while True:
        q = input("> ")
        if q.lower() == "exit":
            break
        print(chat(q))

# -----------------------------
# Memory helpers
# -----------------------------
def load_memory():
    with open(ROOT_DIR / "memory/learner.json") as f:
        return json.load(f)

def save_memory(mem):
    with open(ROOT_DIR / "memory/learner.json", "w") as f:
        json.dump(mem, f, indent=2)

# -----------------------------
# Agent loop
# -----------------------------
def chat(user_input):
    return llm.chat(SYSTEM_PROMPT, user_input)

if __name__ == "__main__":
    main()
