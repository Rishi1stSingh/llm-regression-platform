from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod

import requests


class LLMClient(ABC):
    @abstractmethod
    def classify(self, system_prompt: str, ticket: str) -> str:
        raise NotImplementedError


class NvidiaLLMClient(LLMClient):
    """NVIDIA NIM uses the OpenAI-compatible chat-completions API."""

    def __init__(self) -> None:
        self.api_key = os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY is required. Use --mock for an offline demo.")
        self.base_url = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
        self.model = os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")

    def classify(self, system_prompt: str, ticket: str) -> str:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "temperature": 0, "max_tokens": 8,
                  "messages": [{"role": "system", "content": system_prompt},
                               {"role": "user", "content": f"Classify this ticket:\n{ticket}"}]},
            timeout=45,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


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
