"""
Tests for llm_clients.py.

The offline tests check that clients are pointed at the proxy with the
virtual key and the end user, and that proxy rejections are recognised. The
proxy tests use the LiteLLM proxy at LITELLM_BASE_URL and are skipped if it is
not reachable.

Run with:  uv run pytest
"""

import os
import sys
from pathlib import Path

import httpx
import openai
import pytest
import requests

# Make the project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm_clients  # noqa: E402
from llm_clients import (  # noqa: E402
    create_chat_model,
    create_embeddings_model,
    get_key_spend,
    proxy_error_message,
)


@pytest.fixture
def proxy_env(monkeypatch):
    """Point the clients at a known proxy URL, key and model names."""
    monkeypatch.setenv("LITELLM_BASE_URL", "http://proxy.test:4000/")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-virtual-test")
    monkeypatch.setenv("LITELLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LITELLM_EMBEDDINGS_MODEL", "text-embedding-ada-002")


def _api_error(error_class, status_code: int, message: str) -> openai.APIStatusError:
    """
    Build an OpenAI API error like the ones the proxy causes.

    :param
        error_class: openai.BadRequestError, openai.RateLimitError, etc.
        status_code (int): HTTP status code of the fake response.
        message (str): Error message from the proxy.

    :returns
        openai.APIStatusError: The constructed error.
    """
    request = httpx.Request("POST", "http://proxy.test:4000/chat/completions")
    response = httpx.Response(status_code, request=request)
    return error_class(message, response=response, body=None)


# ---------------------------
# Client construction (offline)
# ---------------------------

def test_chat_model_uses_proxy_virtual_key_and_user(proxy_env):
    """The chat client calls the proxy with the virtual key and tags the end user."""
    chat_model = create_chat_model("James")

    assert chat_model.openai_api_base == "http://proxy.test:4000"  # trailing slash removed
    assert chat_model.openai_api_key.get_secret_value() == "sk-virtual-test"
    assert chat_model.model_name == "gpt-4o-mini"
    assert chat_model.model_kwargs == {"user": "James"}


def test_chat_model_attaches_callbacks(proxy_env):
    """Callbacks passed in are attached to the client."""
    from langchain_core.callbacks import BaseCallbackHandler

    handler = BaseCallbackHandler()
    assert create_chat_model("James", callbacks=[handler]).callbacks == [handler]


def test_embeddings_model_uses_proxy_virtual_key_and_user(proxy_env):
    """The embeddings client calls the proxy with the virtual key and tags the end user."""
    embeddings_model = create_embeddings_model("Mike")

    assert embeddings_model.openai_api_base == "http://proxy.test:4000"
    assert embeddings_model.openai_api_key.get_secret_value() == "sk-virtual-test"
    assert embeddings_model.model == "text-embedding-ada-002"
    assert embeddings_model.model_kwargs == {"user": "Mike"}


# ---------------------------
# proxy_error_message (offline)
# ---------------------------

def test_budget_exceeded_is_recognised():
    """
    A budget rejection from the proxy gets the budget message.

    The status code and message are the ones LiteLLM v1.108.15 returned when
    an end user went over budget during manual testing.
    """
    error = _api_error(
        openai.UnprocessableEntityError,
        422,
        "ExceededBudget: End User=budget-test over budget. Spend=2.55e-06, Budget=1e-06",
    )
    assert "budget" in proxy_error_message(error).lower()


def test_rate_limit_is_recognised():
    """A 429 from the proxy gets the rate limit message."""
    error = _api_error(openai.RateLimitError, 429, "Rate limit reached for key")
    assert "too many requests" in proxy_error_message(error).lower()


def test_wrapped_proxy_error_is_found():
    """A proxy error wrapped in another exception (e.g. by NeMo) is still recognised."""
    inner = _api_error(openai.BadRequestError, 400, "Budget has been exceeded")
    try:
        try:
            raise inner
        except openai.BadRequestError as caught:
            raise RuntimeError("LLM call failed") from caught
    except RuntimeError as wrapped:
        assert proxy_error_message(wrapped) is not None


def test_other_errors_are_not_treated_as_proxy_rejections():
    """Unrelated errors return None so the caller can re-raise them."""
    assert proxy_error_message(ValueError("boom")) is None
    assert proxy_error_message(_api_error(openai.BadRequestError, 400, "invalid model name")) is None


# ---------------------------
# get_key_spend (offline)
# ---------------------------

def test_get_key_spend_reads_spend(proxy_env, monkeypatch):
    """get_key_spend returns the spend the proxy reports for the key [USD]."""

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"info": {"spend": 0.00025935}}

    captured = {}

    def fake_get(url, headers, timeout):
        captured.update(url=url, headers=headers)
        return FakeResponse()

    monkeypatch.setattr(llm_clients.requests, "get", fake_get)

    assert get_key_spend() == pytest.approx(0.00025935)
    assert captured["url"] == "http://proxy.test:4000/key/info"
    assert captured["headers"] == {"Authorization": "Bearer sk-virtual-test"}


def test_get_key_spend_returns_none_when_proxy_down(proxy_env, monkeypatch):
    """If the proxy cannot be reached, get_key_spend returns None instead of raising."""

    def failing_get(*args, **kwargs):
        raise requests.ConnectionError("proxy down")

    monkeypatch.setattr(llm_clients.requests, "get", failing_get)
    assert get_key_spend() is None


# ---------------------------
# Live proxy (skipped if not running)
# ---------------------------

def _proxy_available() -> bool:
    """Return True if the LiteLLM proxy answers its liveness check."""
    import dotenv

    dotenv.load_dotenv()
    try:
        return requests.get(f"{llm_clients.proxy_url()}/health/liveliness", timeout=2).ok
    except requests.RequestException:
        return False


requires_proxy = pytest.mark.skipif(not _proxy_available(), reason="LiteLLM proxy not reachable")


@requires_proxy
def test_virtual_key_reports_spend():
    """The app's virtual key is valid on the proxy and reports its spend [USD]."""
    assert os.getenv("LITELLM_API_KEY"), "LITELLM_API_KEY missing from .env (run scripts/setup_litellm.py)"
    assert get_key_spend() is not None


@requires_proxy
def test_virtual_key_cannot_use_unlisted_models():
    """The virtual key is limited to the app's models; other models are refused."""
    response = requests.post(
        f"{llm_clients.proxy_url()}/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('LITELLM_API_KEY')}"},
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
        timeout=10,
    )
    assert response.status_code in (400, 401, 403)
