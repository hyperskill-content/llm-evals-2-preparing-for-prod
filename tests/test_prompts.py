"""
Tests for prompts.py and scripts/create_langfuse_prompts.py.

These tests run offline. A fake Langfuse client serves real Langfuse prompt
objects (ChatPromptClient / TextPromptClient) built from the prompt text in the
creation script, so they exercise the same {{variable}} conversion the live
client uses, without needing Langfuse credentials or network access.

Run with:  uv run pytest
"""

import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langfuse.api import ChatMessage, Prompt_Chat, Prompt_Text
from langfuse.model import ChatPromptClient, TextPromptClient

# Make the project root and scripts folder importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import create_langfuse_prompts as source_prompts  # noqa: E402
from prompts import (  # noqa: E402
    CONTEXT_PROMPT_NAME,
    GOODBYE_PROMPT_NAME,
    PROMPT_LABEL,
    REVIEW_PROMPT_NAME,
    load_chat_prompt,
    load_text_prompt,
)


class FakeLangfuseClient:
    """
    Stand-in for the Langfuse client that serves prompts from memory.

    It records every get_prompt call so tests can check which name, label and
    type the loaders asked for.
    """

    def __init__(self):
        """Build the in-memory prompt store from the creation script's prompt text."""
        self.calls = []
        self.prompts = {
            CONTEXT_PROMPT_NAME: _chat_prompt(CONTEXT_PROMPT_NAME, source_prompts.CONTEXT_SYSTEM_PROMPT),
            REVIEW_PROMPT_NAME: _chat_prompt(REVIEW_PROMPT_NAME, source_prompts.REVIEW_SYSTEM_PROMPT),
            GOODBYE_PROMPT_NAME: _text_prompt(GOODBYE_PROMPT_NAME, source_prompts.GOODBYE_SYSTEM_PROMPT),
        }

    def get_prompt(self, name, *, label=None, type="text"):
        """
        Return the stored prompt, mimicking Langfuse's get_prompt signature.

        :param
            name (str): Prompt name.
            label (str): Requested label (recorded, not used for lookup).
            type (str): Requested prompt type, "chat" or "text".

        :returns
            ChatPromptClient or TextPromptClient for the named prompt.
        """
        self.calls.append({"name": name, "label": label, "type": type})
        return self.prompts[name]


def _chat_prompt(name: str, system_text: str) -> ChatPromptClient:
    """Wrap a system message in a real Langfuse ChatPromptClient (version 1, production)."""
    api_prompt = Prompt_Chat(
        name=name,
        version=1,
        config={},
        labels=[PROMPT_LABEL],
        tags=[],
        prompt=[ChatMessage(role="system", content=system_text)],
    )
    return ChatPromptClient(api_prompt)


def _text_prompt(name: str, text: str) -> TextPromptClient:
    """Wrap prompt text in a real Langfuse TextPromptClient (version 1, production)."""
    api_prompt = Prompt_Text(
        name=name,
        version=1,
        config={},
        labels=[PROMPT_LABEL],
        tags=[],
        prompt=text,
    )
    return TextPromptClient(api_prompt)


@pytest.fixture
def fake_client():
    """Fresh fake Langfuse client for each test."""
    return FakeLangfuseClient()


# A short conversation used as the chat history input
SAMPLE_CONVERSATION = [
    HumanMessage("Tell me about the Galaxy S23"),
    AIMessage("The Galaxy S23 has a great camera."),
]


def test_chat_prompt_requests_production_chat_version(fake_client):
    """load_chat_prompt asks Langfuse for the named chat prompt with the production label."""
    load_chat_prompt(fake_client, CONTEXT_PROMPT_NAME)
    assert fake_client.calls == [{"name": CONTEXT_PROMPT_NAME, "label": PROMPT_LABEL, "type": "chat"}]


def test_chat_prompt_fills_variables_and_appends_history(fake_client):
    """
    The context prompt renders as one system message (with user_input filled in)
    followed by the conversation history in its original order.
    """
    chat_prompt = load_chat_prompt(fake_client, CONTEXT_PROMPT_NAME)
    messages = chat_prompt.invoke(
        {"user_input": "Tell me about the Galaxy S23", "conversation": SAMPLE_CONVERSATION}
    ).to_messages()

    assert isinstance(messages[0], SystemMessage)
    assert "Current question: Tell me about the Galaxy S23" in messages[0].content
    # No unconverted Langfuse or LangChain variables should remain
    assert "{" not in messages[0].content
    assert messages[1:] == SAMPLE_CONVERSATION


def test_review_prompt_fills_user_id_and_input(fake_client):
    """The review prompt substitutes both user_id and user_input."""
    review_prompt = load_chat_prompt(fake_client, REVIEW_PROMPT_NAME)
    system_message = review_prompt.invoke(
        {"user_id": "Sherlock", "user_input": "Which phone is best?", "conversation": []}
    ).to_messages()[0]

    assert "Current user: Sherlock" in system_message.content
    assert "Current question: Which phone is best?" in system_message.content


def test_chat_prompt_metadata_links_langfuse_prompt(fake_client):
    """The template's metadata holds the Langfuse prompt so traces link to its version."""
    chat_prompt = load_chat_prompt(fake_client, REVIEW_PROMPT_NAME)
    assert chat_prompt.metadata["langfuse_prompt"] is fake_client.prompts[REVIEW_PROMPT_NAME]


def test_text_prompt_requests_production_text_version(fake_client):
    """load_text_prompt asks Langfuse for the named text prompt with the production label."""
    load_text_prompt(fake_client, GOODBYE_PROMPT_NAME)
    assert fake_client.calls == [{"name": GOODBYE_PROMPT_NAME, "label": PROMPT_LABEL, "type": "text"}]


def test_text_prompt_fills_user_id_and_links_metadata(fake_client):
    """The goodbye prompt substitutes user_id and carries the Langfuse prompt in metadata."""
    goodbye_prompt = load_text_prompt(fake_client, GOODBYE_PROMPT_NAME)
    rendered_text = goodbye_prompt.invoke({"user_id": "James"}).to_string()

    assert "helping the user: James with smartphone" in rendered_text
    assert goodbye_prompt.input_variables == ["user_id"]
    assert goodbye_prompt.metadata["langfuse_prompt"] is fake_client.prompts[GOODBYE_PROMPT_NAME]


def test_create_prompts_uploads_all_three_with_production_label():
    """create_prompts uploads each prompt with the expected type and the production label."""

    class RecordingClient:
        """Records create_prompt calls and returns an object with a version number."""

        def __init__(self):
            self.created = []

        def create_prompt(self, **kwargs):
            self.created.append(kwargs)
            return type("CreatedPrompt", (), {"version": 1})()

    recording_client = RecordingClient()
    source_prompts.create_prompts(recording_client)

    created_types = {call["name"]: call["type"] for call in recording_client.created}
    assert created_types == {
        CONTEXT_PROMPT_NAME: "chat",
        REVIEW_PROMPT_NAME: "chat",
        GOODBYE_PROMPT_NAME: "text",
    }
    assert all(call["labels"] == [PROMPT_LABEL] for call in recording_client.created)
