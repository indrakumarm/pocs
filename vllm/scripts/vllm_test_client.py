#!/usr/bin/env python3
"""
Standalone vLLM OpenAI-compatible API test client.

Examples:
  python3 vllm_test_client.py --base-url http://localhost:8000/v1 --model qwen
  python3 vllm_test_client.py --base-url http://10.192.10.184:8000/v1 --model qwen --prompt "Explain Airflow retries."

Environment variables:
  VLLM_BASE_URL  default: http://localhost:8000/v1
  VLLM_MODEL     default: qwen
  VLLM_API_KEY   default: EMPTY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class ChatResult:
    content: str
    model: str
    usage: dict[str, Any]
    raw_response: dict[str, Any]
    elapsed_seconds: float


class VLLMClient:
    """Small vLLM client using the OpenAI-compatible REST API."""

    def __init__(
        self,
        base_url: str = "http://10.19.210.184:9090/v1",
        api_key: str = "EMPTY",
        timeout: int = 300,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        request = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
                if not response_body:
                    return {}
                return json.loads(response_body)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {url}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not connect to {url}: {exc.reason}") from exc

    def health(self) -> bool:
        health_url = self.base_url.removesuffix("/v1") + "/health"
        request = urllib.request.Request(health_url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return 200 <= response.status < 300
        except urllib.error.URLError:
            return False

    def models(self) -> list[str]:
        response = self._request("GET", "/models")
        return [item["id"] for item in response.get("data", [])]

    def chat(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "You are a concise and helpful assistant.",
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> ChatResult:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        started_at = time.perf_counter()
        response = self._request("POST", "/chat/completions", payload)
        elapsed = time.perf_counter() - started_at

        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError(f"No choices returned by vLLM: {response}")

        message = choices[0].get("message", {})
        content = message.get("content", "")

        return ChatResult(
            content=content,
            model=response.get("model", model),
            usage=response.get("usage", {}),
            raw_response=response,
            elapsed_seconds=elapsed,
        )

    def converse(self, *args: Any, **kwargs: Any) -> ChatResult:
        """Alias for chat(), useful if your application calls this operation converse."""
        return self.chat(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a vLLM OpenAI-compatible server.")
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://10.192.10.184:9090/v1"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL", "qwen"))
    parser.add_argument("--prompt", default="Say hello and mention that vLLM is working.")
    parser.add_argument("--system-prompt", default="You are a concise and helpful assistant.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--raw", action="store_true", help="Print full raw JSON response.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = VLLMClient(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)

    print(f"Base URL: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Health: {'OK' if client.health() else 'FAILED'}")

    available_models = client.models()
    print(f"Available models: {available_models}")

    result = client.converse(
        model=args.model,
        prompt=args.prompt,
        system_prompt=args.system_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print("\nResponse:")
    print(result.content)

    print("\nToken usage:")
    print(json.dumps(result.usage, indent=2))

    print(f"\nLatency seconds: {result.elapsed_seconds:.2f}")

    if args.raw:
        print("\nRaw response:")
        print(json.dumps(result.raw_response, indent=2))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
