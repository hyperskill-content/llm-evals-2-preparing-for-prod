"""
Tests for chat_history.py.

The trimmer and save_turn tests run offline with an in-memory history and a
simple word-count token counter. The Redis tests use the Redis server at
REDIS_URL (default redis://localhost:6380/0) and are skipped if it is not
reachable.

Run with:  uv run pytest
"""

import os
import sys
import uuid
from pathlib import Path

import pytest
import redis
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Make the project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chat_history import build_trimmer, create_chat_history, save_turn  # noqa: E402

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6380/0")


def count_words(messages) -> int:
    """
    Deterministic stand-in for a tokenizer: one token per whitespace-separated word.

    :param
        messages (list[BaseMessage]): Messages to count.

    :returns
        int: Total number of words across all message contents [tokens].
    """
    return sum(len(str(message.content).split()) for message in messages)


def make_history(turn_count: int) -> list:
    """
    Build a chat history of alternating user questions and assistant replies.

    :param
        turn_count (int): Number of question/reply pairs.

    :returns
        list: 2 * turn_count messages, oldest first. Each message is 3 words.
    """
    messages = []
    for turn in range(turn_count):
        messages.append(HumanMessage(f"question number {turn}"))
        messages.append(AIMessage(f"answer number {turn}"))
    return messages


# ---------------------------
# save_turn
# ---------------------------

def test_save_turn_stores_user_message_then_reply():
    """save_turn appends exactly the user's message and the reply, in that order."""
    history = InMemoryChatMessageHistory()
    user_message = HumanMessage("Tell me about the Pixel 8")
    ai_message = AIMessage("The Pixel 8 has a great camera.")

    save_turn(history, user_message, ai_message)

    assert history.messages == [user_message, ai_message]


# ---------------------------
# build_trimmer
# ---------------------------

def test_trimmer_keeps_short_history_unchanged():
    """History that fits within the budget passes through untouched."""
    messages = [SystemMessage("you are helpful")] + make_history(2) + [HumanMessage("latest question here")]
    trimmer = build_trimmer(token_counter=count_words, max_tokens=100)

    assert trimmer.invoke(messages) == messages


def test_trimmer_drops_oldest_messages_and_keeps_system():
    """
    Over budget, the trimmer keeps the system message and the newest messages,
    and the kept history starts on a human message.

    Every message is 3 words, so a 15-token budget holds the system message
    plus the 4 newest messages (answer 2, question 3, answer 3, latest
    question). That would start on an AI reply, so the trimmer also drops
    answer 2 and keeps: system, question 3, answer 3, latest question.
    """
    system_message = SystemMessage("you are helpful")
    latest_question = HumanMessage("latest question here")
    messages = [system_message] + make_history(4) + [latest_question]
    trimmer = build_trimmer(token_counter=count_words, max_tokens=15)

    trimmed = trimmer.invoke(messages)

    assert trimmed[0] == system_message
    assert trimmed[1].type == "human"  # history starts on a user message
    assert trimmed[-1] == latest_question  # the current question is kept
    assert count_words(trimmed) <= 15
    assert [m.content for m in trimmed] == ["you are helpful", "question number 3", "answer number 3", "latest question here"]


def test_trimmer_never_starts_history_on_tool_result():
    """A tool result is not kept without the human message that led to it."""
    messages = [
        SystemMessage("system"),
        HumanMessage("old question that is fairly long"),
        AIMessage("", tool_calls=[{"name": "SmartphoneInfo", "args": {"model": "x"}, "id": "call_1"}]),
        ToolMessage("tool result words", tool_call_id="call_1"),
        HumanMessage("new question"),
    ]
    trimmer = build_trimmer(token_counter=count_words, max_tokens=6)

    trimmed = trimmer.invoke(messages)

    assert [message.type for message in trimmed] == ["system", "human"]
    assert trimmed[-1].content == "new question"


def test_trimmer_accepts_prompt_value_in_chain():
    """The trimmer works on a prompt's output, as it does in the context chain."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Current question: {user_input}"), MessagesPlaceholder("conversation")]
    )
    history = make_history(5) + [HumanMessage("which phone is best")]
    chain = prompt | build_trimmer(token_counter=count_words, max_tokens=12)

    trimmed = chain.invoke({"user_input": "which phone is best", "conversation": history})

    assert trimmed[0].type == "system"
    assert trimmed[-1].content == "which phone is best"
    assert count_words(trimmed) <= 12


# ---------------------------
# Redis integration
# ---------------------------

def _redis_available() -> bool:
    """Return True if the Redis server at REDIS_URL answers a ping."""
    try:
        return redis.Redis.from_url(REDIS_URL, socket_connect_timeout=1).ping()
    except redis.exceptions.RedisError:
        return False


requires_redis = pytest.mark.skipif(not _redis_available(), reason=f"Redis not reachable at {REDIS_URL}")


@pytest.fixture
def redis_history():
    """Chat history for a unique throwaway session, cleared after the test."""
    history = create_chat_history(f"test-{uuid.uuid4().hex}", REDIS_URL, ttl_seconds=120)
    yield history
    history.clear()


@requires_redis
def test_redis_history_persists_across_instances(redis_history):
    """A saved turn can be read back by a new history object for the same session."""
    save_turn(redis_history, HumanMessage("hello"), AIMessage("hi, how can I help?"))

    reopened = create_chat_history(redis_history.session_id, REDIS_URL, ttl_seconds=120)

    assert [(m.type, m.content) for m in reopened.messages] == [
        ("human", "hello"),
        ("ai", "hi, how can I help?"),
    ]


@requires_redis
def test_redis_history_expires_with_ttl(redis_history):
    """Every stored message key has a TTL no longer than the one requested [seconds]."""
    save_turn(redis_history, HumanMessage("hello"), AIMessage("hi"))

    redis_client = redis.Redis.from_url(REDIS_URL)
    message_keys = redis_client.keys(f"{redis_history.key_prefix}{redis_history.session_id}*")

    assert message_keys, "no keys were written for the session"
    for key in message_keys:
        assert 0 < redis_client.ttl(key) <= 120
