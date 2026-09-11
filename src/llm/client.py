from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


class LLMClient(ABC):
    @abstractmethod
    def classify(self, system_prompt: str, ticket: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def summarize(self, system_prompt: str, document: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_sql(self, system_prompt: str, question: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def judge(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raise NotImplementedError


class NvidiaLLMClient(LLMClient):
    """NVIDIA NIM uses the OpenAI-compatible chat-completions API."""

    def __init__(
        self,
        requests_per_minute: int = 25,
        max_retries: int = 3,
    ) -> None:
        self.api_key = os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY is required. Use --mock for an offline demo.")
        self.base_url = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
        self.model = os.environ.get("NVIDIA_MODEL", "deepseek-ai/deepseek-v4-flash-0731")
        self._timeout = (10, int(os.environ.get("NVIDIA_TIMEOUT", "180")))
        from src.llm.rate_limiter import RateLimitedHTTPClient

        self._http = RateLimitedHTTPClient(
            requests_per_minute=requests_per_minute,
            max_retries=max_retries,
            timeout=self._timeout,
        )

    def _chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        response = self._http.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "temperature": 0, "max_tokens": max_tokens,
                  "reasoning_effort": "low",
                  "messages": [{"role": "system", "content": system_prompt},
                               {"role": "user", "content": user_prompt}]},
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if content is None:
            raise RuntimeError(
                f"NVIDIA API returned null content. "
                f"Full message: {response.json()['choices'][0]['message']}"
            )
        return content

    def classify(self, system_prompt: str, ticket: str) -> str:
        return self._chat(system_prompt, f"Classify this ticket:\n{ticket}", max_tokens=100)

    def summarize(self, system_prompt: str, document: str) -> str:
        return self._chat(system_prompt, f"Summarize this document:\n{document}", max_tokens=512)

    def generate_sql(self, system_prompt: str, question: str) -> str:
        return self._chat(system_prompt, f"Generate SQL for this question:\n{question}", max_tokens=256)

    def judge(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        content = self._chat(system_prompt, user_prompt, max_tokens=512)
        # Try to parse JSON from the response (may be wrapped in markdown fences)
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"Judge returned invalid JSON: {content[:200]}")


class GroqLLMClient(LLMClient):
    """Groq uses an OpenAI-compatible chat-completions API."""

    def __init__(self, requests_per_minute: int = 25, max_retries: int = 3) -> None:
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is required. Set it in .env or the environment. "
                "Use --mock for an offline demo."
            )
        self.base_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
        self.model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        self._timeout = (10, int(os.environ.get("GROQ_TIMEOUT", "45")))
        from src.llm.rate_limiter import RateLimitedHTTPClient

        self._http = RateLimitedHTTPClient(
            requests_per_minute=requests_per_minute,
            max_retries=max_retries,
            timeout=self._timeout,
        )

    def _chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        response = self._http.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "temperature": 0,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if content is None:
            raise RuntimeError(
                f"Groq API returned null content. "
                f"Full message: {response.json()['choices'][0]['message']}"
            )
        return content

    def classify(self, system_prompt: str, ticket: str) -> str:
        return self._chat(system_prompt, f"Classify this ticket:\n{ticket}", max_tokens=100)

    def summarize(self, system_prompt: str, document: str) -> str:
        return self._chat(system_prompt, f"Summarize this document:\n{document}", max_tokens=512)

    def generate_sql(self, system_prompt: str, question: str) -> str:
        return self._chat(system_prompt, f"Generate SQL for this question:\n{question}", max_tokens=256)

    def judge(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        content = self._chat(system_prompt, user_prompt, max_tokens=512)
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"Judge returned invalid JSON: {content[:200]}")


class MockLLMClient(LLMClient):
    """Deterministic test client. v2's narrower technical wording causes regressions."""

    def classify(self, system_prompt: str, ticket: str) -> str:
        text = ticket.lower()
        if re.search(r"charge|invoice|refund|payment|subscription|card|bill", text):
            return "billing"
        if re.search(r"log.?in|password|profile|verification|account|username", text):
            return "account"
        technical = r"crash|error|upload|blank screen|freez|app"
        if "speed" in system_prompt.lower() or "notifications" in system_prompt.lower() or "exports" in system_prompt.lower():
            technical += r"|slow|notification|export|website"
        if re.search(technical, text):
            return "technical"
        return "general"

    def summarize(self, system_prompt: str, document: str) -> str:
        # Deterministic mock: return the first sentence of the document
        sentences = re.split(r"(?<=[.!?])\s+", document.strip())
        return sentences[0] if sentences else ""

    def generate_sql(self, system_prompt: str, question: str) -> str:
        # Deterministic mock: return a simple SELECT based on keywords
        text = question.lower()
        if "count" in text or "how many" in text:
            return "SELECT COUNT(*) FROM users;"
        if "name" in text or "who" in text:
            return "SELECT name FROM users;"
        return "SELECT * FROM users;"

    def judge(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        # Deterministic mock judge: return fixed scores
        return {
            "faithfulness": 0.95,
            "relevance": 0.90,
            "completeness": 0.84,
            "coherence": 0.92,
            "conciseness": 0.88,
            "overall": 0.90,
            "reason": "Mock judge: deterministic scores.",
        }


class AsyncLLMClient(ABC):
    """Async variant of LLMClient using httpx for truly concurrent HTTP requests."""

    @abstractmethod
    async def classify(self, system_prompt: str, ticket: str) -> str:
        raise NotImplementedError

    @abstractmethod
    async def summarize(self, system_prompt: str, document: str) -> str:
        raise NotImplementedError

    @abstractmethod
    async def generate_sql(self, system_prompt: str, question: str) -> str:
        raise NotImplementedError

    @abstractmethod
    async def judge(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raise NotImplementedError


class AsyncNvidiaLLMClient(AsyncLLMClient):
    """Async NVIDIA client using httpx.AsyncClient for concurrent requests."""

    def __init__(self) -> None:
        import httpx

        self.api_key = os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY is required. Use --mock for an offline demo.")
        self.base_url = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
        self.model = os.environ.get("NVIDIA_MODEL", "deepseek-ai/deepseek-v4-flash-0731")
        self._timeout = httpx.Timeout(
            connect=10,
            read=int(os.environ.get("NVIDIA_TIMEOUT", "180")),
            write=10,
            pool=10,
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def _chat(self, client: Any, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        response = await client.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={"model": self.model, "temperature": 0, "max_tokens": max_tokens,
                  "reasoning_effort": "low",
                  "messages": [{"role": "system", "content": system_prompt},
                               {"role": "user", "content": user_prompt}]},
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if content is None:
            raise RuntimeError(
                f"NVIDIA API returned null content. "
                f"Full message: {response.json()['choices'][0]['message']}"
            )
        return content

    async def classify(self, system_prompt: str, ticket: str) -> str:
        import httpx
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await self._chat(client, system_prompt, f"Classify this ticket:\n{ticket}", max_tokens=100)

    async def summarize(self, system_prompt: str, document: str) -> str:
        import httpx
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await self._chat(client, system_prompt, f"Summarize this document:\n{document}", max_tokens=512)

    async def generate_sql(self, system_prompt: str, question: str) -> str:
        import httpx
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await self._chat(client, system_prompt, f"Generate SQL for this question:\n{question}", max_tokens=256)

    async def judge(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        import httpx
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            content = await self._chat(client, system_prompt, user_prompt, max_tokens=512)
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"Judge returned invalid JSON: {content[:200]}")


class AsyncGroqLLMClient(AsyncLLMClient):
    """Async Groq client using httpx.AsyncClient for concurrent requests."""

    def __init__(self) -> None:
        import httpx

        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is required. Use --mock for an offline demo.")
        self.base_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
        self.model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        self._timeout = httpx.Timeout(connect=10, read=45, write=10, pool=10)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def _chat(self, client: Any, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        response = await client.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={"model": self.model, "temperature": 0, "max_tokens": max_tokens,
                  "messages": [{"role": "system", "content": system_prompt},
                               {"role": "user", "content": user_prompt}]},
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if content is None:
            raise RuntimeError(
                f"Groq API returned null content. "
                f"Full message: {response.json()['choices'][0]['message']}"
            )
        return content

    async def classify(self, system_prompt: str, ticket: str) -> str:
        import httpx
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await self._chat(client, system_prompt, f"Classify this ticket:\n{ticket}", max_tokens=100)

    async def summarize(self, system_prompt: str, document: str) -> str:
        import httpx
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await self._chat(client, system_prompt, f"Summarize this document:\n{document}", max_tokens=512)

    async def generate_sql(self, system_prompt: str, question: str) -> str:
        import httpx
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await self._chat(client, system_prompt, f"Generate SQL for this question:\n{question}", max_tokens=256)

    async def judge(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        import httpx
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            content = await self._chat(client, system_prompt, user_prompt, max_tokens=512)
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            raise ValueError(f"Judge returned invalid JSON: {content[:200]}")


class AsyncMockLLMClient(AsyncLLMClient):
    """Deterministic async test client."""

    def __init__(self) -> None:
        self._sync = MockLLMClient()

    async def classify(self, system_prompt: str, ticket: str) -> str:
        return self._sync.classify(system_prompt, ticket)

    async def summarize(self, system_prompt: str, document: str) -> str:
        return self._sync.summarize(system_prompt, document)

    async def generate_sql(self, system_prompt: str, question: str) -> str:
        return self._sync.generate_sql(system_prompt, question)

    async def judge(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return self._sync.judge(system_prompt, user_prompt)