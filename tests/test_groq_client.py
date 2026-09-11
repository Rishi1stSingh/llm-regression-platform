from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.llm.client import GroqLLMClient


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure GROQ env vars are controlled for each test."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_BASE_URL", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)


class TestGroqLLMClientInit:
    def test_requires_api_key(self):
        with pytest.raises(RuntimeError, match="GROQ_API_KEY is required"):
            GroqLLMClient()

    def test_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
        client = GroqLLMClient()

class TestGroqLLMClientChat:
    def _make_client(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        return GroqLLMClient()

    def _mock_response(self, content: str) -> MagicMock:
        resp = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        resp.raise_for_status.return_value = None
        return resp

    def test_classify_calls_chat_with_ticket(self, monkeypatch):
        client = self._make_client(monkeypatch)
        with patch.object(client._http, "post") as mock_post:
            mock_post.return_value = self._mock_response("billing")
            result = client.classify("system prompt", "I was charged twice")
            assert result == "billing"
            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs["json"]
            assert payload["messages"][1]["content"] == "Classify this ticket:\nI was charged twice"
            assert payload["max_tokens"] == 100
            assert payload["model"] == "openai/gpt-oss-120b"

    def test_summarize_calls_chat_with_document(self, monkeypatch):
        client = self._make_client(monkeypatch)
        with patch.object(client._http, "post") as mock_post:
            mock_post.return_value = self._mock_response("A summary.")
            result = client.summarize("prompt", "Long document text.")
            assert result == "A summary."
            payload = mock_post.call_args.kwargs["json"]
            assert payload["messages"][1]["content"] == "Summarize this document:\nLong document text."
            assert payload["max_tokens"] == 512

    def test_generate_sql_calls_chat_with_question(self, monkeypatch):
        client = self._make_client(monkeypatch)
        with patch.object(client._http, "post") as mock_post:
            mock_post.return_value = self._mock_response("SELECT * FROM users;")
            result = client.generate_sql("prompt", "How many users?")
            assert result == "SELECT * FROM users;"
            payload = mock_post.call_args.kwargs["json"]
            assert payload["messages"][1]["content"] == "Generate SQL for this question:\nHow many users?"
            assert payload["max_tokens"] == 256

    def test_judge_parses_json(self, monkeypatch):
        client = self._make_client(monkeypatch)
        judge_data = {"faithfulness": 0.95, "relevance": 0.90, "overall": 0.92, "reason": "Good."}
        with patch.object(client._http, "post") as mock_post:
            mock_post.return_value = self._mock_response(json.dumps(judge_data))
            result = client.judge("prompt", "user prompt")
            assert result == judge_data

    def test_judge_strips_markdown_fences(self, monkeypatch):
        client = self._make_client(monkeypatch)
        judge_data = {"overall": 0.85, "reason": "OK"}
        wrapped = "```json\n" + json.dumps(judge_data) + "\n```"
        with patch("src.llm.client.requests.post") as mock_post:
            mock_post.return_value = self._mock_response(wrapped)
            result = client.judge("prompt", "user prompt")
            assert result == judge_data

    def test_judge_strips_plain_fences(self, monkeypatch):
        client = self._make_client(monkeypatch)
        judge_data = {"overall": 0.75}
        wrapped = "```\n" + json.dumps(judge_data) + "\n```"
        with patch("src.llm.client.requests.post") as mock_post:
            mock_post.return_value = self._mock_response(wrapped)
            result = client.judge("prompt", "user prompt")
            assert result == judge_data

    def test_judge_raises_on_invalid_json(self, monkeypatch):
        client = self._make_client(monkeypatch)
        with patch.object(client._http, "post") as mock_post:
            mock_post.return_value = self._mock_response("not json at all")
            with pytest.raises(ValueError, match="Judge returned invalid JSON"):
                client.judge("prompt", "user prompt")

    def test_chat_uses_correct_url_and_headers(self, monkeypatch):
        client = self._make_client(monkeypatch)
        with patch.object(client._http, "post") as mock_post:
            mock_post.return_value = self._mock_response("test")
            client.classify("sp", "ticket")
            call_kwargs = mock_post.call_args
            assert call_kwargs.args[0] == "https://api.groq.com/openai/v1/chat/completions"
            headers = call_kwargs.kwargs["headers"]
            assert headers["Authorization"] == "Bearer test-key"
            assert headers["Content-Type"] == "application/json"

    def test_chat_uses_configured_model(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("GROQ_MODEL", "custom-model")
        client = GroqLLMClient()
        with patch.object(client._http, "post") as mock_post:
            mock_post.return_value = self._mock_response("result")
            client.classify("sp", "ticket")
            payload = mock_post.call_args.kwargs["json"]
            assert payload["model"] == "custom-model"

    def test_chat_uses_timeout(self, monkeypatch):
        client = self._make_client(monkeypatch)
        with patch.object(client._http, "post") as mock_post:
            mock_post.return_value = self._mock_response("test")
            client.classify("sp", "ticket")
            assert mock_post.call_args.kwargs["timeout"] == client._timeout

    def test_chat_raises_on_http_error(self, monkeypatch):
        client = self._make_client(monkeypatch)
        resp = MagicMock()
        resp.raise_for_status.side_effect = Exception("HTTP 401")
        with patch.object(client._http, "post", return_value=resp):
            with pytest.raises(Exception, match="HTTP 401"):
                client.classify("sp", "ticket")

    def test_default_base_url(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        client = GroqLLMClient()
        assert client.base_url == "https://api.groq.com/openai/v1"

    def test_custom_base_url(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("GROQ_BASE_URL", "https://custom.groq.example.com/v1")
        client = GroqLLMClient()
        assert client.base_url == "https://custom.groq.example.com/v1"

    def test_default_model(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        client = GroqLLMClient()
        assert client.model == "openai/gpt-oss-120b"

    def test_custom_model(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("GROQ_MODEL", "mixtral-8x7b-32768")
        client = GroqLLMClient()
        assert client.model == "mixtral-8x7b-32768"

    def test_timeout_is_tuple(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        client = GroqLLMClient()
        assert isinstance(client._timeout, tuple)
        assert len(client._timeout) == 2
