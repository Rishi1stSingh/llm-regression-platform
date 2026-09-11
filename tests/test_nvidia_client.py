from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from src.llm.client import GroqLLMClient, NvidiaLLMClient


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure NVIDIA env vars are controlled for each test."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)
    monkeypatch.delenv("NVIDIA_TIMEOUT", raising=False)


class TestNvidiaLLMClientTimeout:
    def test_timeout_is_tuple(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        client = NvidiaLLMClient()
        assert isinstance(client._timeout, tuple)
        assert len(client._timeout) == 2
        assert client._timeout == (10, 180)

    # def test_chat_uses_timeout(self, monkeypatch):
    #     monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    #     client = NvidiaLLMClient()
    #     resp = MagicMock()
    #     resp.json.return_value = {
    #         "choices": [{"message": {"content": "billing"}}]
    #     }
    #     resp.raise_for_status.return_value = None

        # with patch.object(client._http._session, "post", return_value=resp) as mock_post:
        #     result = client.classify("prompt", "ticket")
        #     assert result == "billing"
        #     call_kwargs = mock_post.call_args.kwargs
        #     assert "timeout" in call_kwargs
        #     assert call_kwargs["timeout"] == client._timeout
        #     assert call_kwargs["timeout"] == (10, 180)

    def test_nvidia_and_groq_timeouts_are_independent(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        nvidia_client = NvidiaLLMClient()
        groq_client = GroqLLMClient()

        assert nvidia_client._timeout == (10, 180)
        assert groq_client._timeout == (10, 45)
        assert nvidia_client._timeout != groq_client._timeout


# class TestNvidiaLLMClientReasoning:
#     def test_reasoning_effort_low_in_request(self, monkeypatch):
#         monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
#         client = NvidiaLLMClient()
#         resp = MagicMock()
#         resp.json.return_value = {"choices": [{"message": {"content": "billing"}}]}
#         resp.raise_for_status.return_value = None

#         with patch.object(client._http._session, "post", return_value=resp) as mock_post:
#             client.classify("prompt", "ticket")
#             payload = mock_post.call_args.kwargs["json"]
#             assert payload["reasoning_effort"] == "low"

    # def test_max_tokens_100_for_classification(self, monkeypatch):
    #     monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    #     client = NvidiaLLMClient()
    #     resp = MagicMock()
    #     resp.json.return_value = {"choices": [{"message": {"content": "billing"}}]}
    #     resp.raise_for_status.return_value = None

    #     with patch.object(client._http._session, "post", return_value=resp) as mock_post:
    #         client.classify("prompt", "ticket")
    #         payload = mock_post.call_args.kwargs["json"]
    #         assert payload["max_tokens"] == 100

    # def test_null_content_raises_error(self, monkeypatch):
    #     monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    #     client = NvidiaLLMClient()
    #     resp = MagicMock()
    #     resp.json.return_value = {
    #         "choices": [{"message": {"content": None, "reasoning": "some reasoning"}}]
    #     }
    #     resp.raise_for_status.return_value = None

    #     with patch.object(client._http._session, "post", return_value=resp):
    #         with pytest.raises(RuntimeError, match="NVIDIA API returned null content"):
    #             client.classify("prompt", "ticket")
