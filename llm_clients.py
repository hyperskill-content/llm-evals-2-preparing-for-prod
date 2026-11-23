"""
Model clients for the smartphone assistant, routed through the LiteLLM proxy.

The app never holds the real provider API key. Every chat and embeddings call
goes to the LiteLLM proxy with a virtual key (LITELLM_API_KEY). The proxy holds
the provider key and enforces the key's budget and rate limits.

Each request also carries a "user" field naming the end user. LiteLLM tracks
spend per end user and applies that user's budget (a named budget such as
HyperBudget, or the proxy's max_end_user_budget default).

All client construction lives here so the proxy URL, key, model names and
user attribution are configured in one place.
"""

import os
from typing import Any, Optional

import openai
import requests
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# End users of the demo app. main.py picks one per session; the LiteLLM setup
# script registers each of them as a customer with a budget.
DEMO_USERS = ["James", "George", "Mike", "Sherlock"]

# Proxy settings, read from .env
DEFAULT_PROXY_URL = "http://localhost:4000"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDINGS_MODEL = "text-embedding-ada-002"


def proxy_url() -> str:
    """
    Base URL of the LiteLLM proxy.

    :returns
        str: LITELLM_BASE_URL from the environment, or http://localhost:4000.
    """
    return os.getenv("LITELLM_BASE_URL", DEFAULT_PROXY_URL).rstrip("/")


def create_chat_model(user_id: str, callbacks: Optional[list] = None) -> ChatOpenAI:
    """
    Build a chat model client that calls the LiteLLM proxy as user_id.

    :param
        user_id (str): End user the calls are billed to. Sent as the OpenAI
            "user" field, which LiteLLM uses for per-user spend and budgets.
        callbacks (list | None): LangChain callbacks to attach to every call
            made with this client (e.g. the Langfuse handler).

    :returns
        ChatOpenAI: Client for the proxy's chat model (LITELLM_MODEL).
    """
    return ChatOpenAI(
        model=os.getenv("LITELLM_MODEL", DEFAULT_CHAT_MODEL),
        base_url=proxy_url(),
        api_key=os.getenv("LITELLM_API_KEY"),
        model_kwargs={"user": user_id},
        callbacks=callbacks,
    )


def create_embeddings_model(user_id: str) -> OpenAIEmbeddings:
    """
    Build an embeddings client that calls the LiteLLM proxy as user_id.

    :param
        user_id (str): End user the calls are billed to (OpenAI "user" field).

    :returns
        OpenAIEmbeddings: Client for the proxy's embeddings model
        (LITELLM_EMBEDDINGS_MODEL).
    """
    return OpenAIEmbeddings(
        model=os.getenv("LITELLM_EMBEDDINGS_MODEL", DEFAULT_EMBEDDINGS_MODEL),
        base_url=proxy_url(),
        api_key=os.getenv("LITELLM_API_KEY"),
        model_kwargs={"user": user_id},
        # The proxy serves a generic OpenAI-compatible endpoint, so send plain
        # text instead of pre-tokenized input
        check_embedding_ctx_length=False,
        show_progress_bar=True,
    )


def get_key_spend(timeout_seconds: float = 5.0) -> Optional[float]:
    """
    Ask the proxy how much the app's virtual key has spent in its current budget period.

    :param
        timeout_seconds (float): Request timeout [seconds].

    :returns
        float | None: Spend on LITELLM_API_KEY [USD], or None if the proxy
        could not be reached or did not report it.
    """
    try:
        response = requests.get(
            f"{proxy_url()}/key/info",
            headers={"Authorization": f"Bearer {os.getenv('LITELLM_API_KEY')}"},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        return float(response.json()["info"]["spend"])
    except (requests.RequestException, KeyError, TypeError, ValueError):
        return None


def proxy_error_message(error: BaseException) -> Optional[str]:
    """
    Turn a proxy rejection (budget exceeded or rate limited) into a user-facing message.

    Errors can arrive wrapped (for example inside a NeMo Guardrails
    exception), so the chain of causes is searched for an OpenAI API error.

    :param
        error (BaseException): Exception raised while handling a turn.

    :returns
        str | None: Message to show the user, or None if the error is not a
        budget or rate limit rejection from the proxy.
    """
    current: Optional[BaseException] = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, openai.APIStatusError):
            detail = str(current).lower()
            if "budget" in detail:
                return "You've reached your usage budget for now. Please try again after it resets."
            if isinstance(current, openai.RateLimitError) or "rate limit" in detail:
                return "Too many requests right now. Please wait a minute and try again."
        current = current.__cause__ or current.__context__
    return None
