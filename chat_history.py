"""
Chat history management for the smartphone assistant.

Chat history used to live in a global in-memory list, which grew for the whole
session and was lost when the program exited. It is now stored in Redis:

  - Only the user's messages and the assistant's final replies are saved.
    Tool calls and tool results are kept in memory for the current turn only,
    so they do not inflate stored history or the tokens sent on later turns.
  - Each session's history expires after a time to live (TTL), so old
    conversations are cleared from Redis automatically.
  - A trimmer caps the history sent to the context chain at a token budget,
    keeping the most recent messages.
"""

from typing import Any, Callable, Sequence, Union

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, trim_messages
from langchain_core.runnables import Runnable
from langchain_redis import RedisChatMessageHistory

# How long a session's history is kept in Redis after its last write [seconds].
# 3600 s = 1 hour.
CHAT_HISTORY_TTL_SECONDS = 3600

# Maximum size of the prompt (system message plus history) sent to the context
# chain [tokens]. Older messages are dropped first when this is exceeded.
CONTEXT_MAX_TOKENS = 500


def create_chat_history(
    session_id: str,
    redis_url: str,
    ttl_seconds: int = CHAT_HISTORY_TTL_SECONDS,
) -> RedisChatMessageHistory:
    """
    Open the Redis-backed chat history for one session.

    :param
        session_id (str): Identifier of the chat session. Each session's
            messages are stored under their own Redis key.
        redis_url (str): Redis connection URL, e.g. "redis://localhost:6380/0".
        ttl_seconds (int): Time to live of the session's history [seconds].
            Each write resets the timer, so history expires this long after
            the last message.

    :returns
        RedisChatMessageHistory: History object whose .messages are read
        from, and whose add_message() writes to, Redis.
    """
    return RedisChatMessageHistory(session_id=session_id, redis_url=redis_url, ttl=ttl_seconds)


def save_turn(
    history: BaseChatMessageHistory,
    user_message: HumanMessage,
    ai_message: AIMessage,
) -> None:
    """
    Persist one conversation turn: the user's message and the final reply.

    Intermediate messages from the turn (tool calls and tool results) are
    deliberately not saved.

    :param
        history (BaseChatMessageHistory): Chat history to write to (Redis in
            the app, any LangChain history in tests).
        user_message (HumanMessage): The user's input for this turn.
        ai_message (AIMessage): The assistant's final reply for this turn.

    :returns
        None. The two messages are appended to the history in order.
    """
    history.add_message(user_message)
    history.add_message(ai_message)


def build_trimmer(
    token_counter: Union[BaseLanguageModel, Callable[[Sequence[BaseMessage]], int]],
    max_tokens: int = CONTEXT_MAX_TOKENS,
) -> Runnable[Any, list[BaseMessage]]:
    """
    Build a runnable that trims chat history to a token budget.

    The trimmer keeps the system message and the most recent messages that fit
    in max_tokens. The kept history always starts on a human message, so a
    reply or tool result is never separated from the message that triggered it.

    :param
        token_counter: The chat model (its tokenizer is used to count tokens)
            or a function that returns the token count of a list of messages.
        max_tokens (int): Maximum tokens to keep, including the system
            message [tokens].

    :returns
        Runnable: Takes a prompt value or list of messages and returns the
        trimmed list of messages. Can be placed in a chain between the prompt
        and the model.
    """
    return trim_messages(
        strategy="last",  # keep the most recent messages
        token_counter=token_counter,
        max_tokens=max_tokens,
        start_on="human",  # the kept history must begin with a user message
        end_on=("human", "tool"),  # and end on the user's question or a tool result
        include_system=True,  # always keep the system prompt
    )
