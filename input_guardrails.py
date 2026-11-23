"""
Input guardrails for the smartphone assistant, using NVIDIA NeMo Guardrails.

Every user message is checked by the "self check input" rail before any chain
runs. The rail asks the model whether the message breaks the guidelines in
config/prompts.yml (customer support requests, abuse, impersonation, prompt
extraction, sharing personal information, and so on). Blocked messages get a
refusal, cost one short classification call, and are never sent to the
context or review chains or saved to chat history.

Validation runs on its own instead of being chained in front of the other
chains. That allows an early exit before the expensive calls, and avoids
passing LangChain message objects through the rails.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from langchain_core.language_models import BaseChatModel
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import GenerationOptions

# Folder holding config.yml and prompts.yml, next to this file so the app can
# be started from any working directory
GUARDRAILS_CONFIG_PATH = Path(__file__).resolve().parent / "config"

# Shown to the user if a rail blocks a message without supplying its own reply
DEFAULT_BLOCKED_MESSAGE = "I'm sorry, I can't respond to that."


@dataclass
class InputCheck:
    """
    Outcome of checking one user message against the input rails.

    Attributes:
        blocked (bool): True if a rail rejected the message.
        bot_message (str): Refusal to show the user when blocked; empty otherwise.
        triggered_rail (Optional[str]): Name of the rail that blocked the
            message, e.g. "self check input"; None when allowed.
    """

    blocked: bool
    bot_message: str = ""
    triggered_rail: Optional[str] = None


def load_input_rails(
    llm: BaseChatModel,
    config_path: Union[str, Path] = GUARDRAILS_CONFIG_PATH,
) -> LLMRails:
    """
    Load the guardrails configuration and attach the model that runs the checks.

    :param
        llm (BaseChatModel): Chat model used for the self-check classification.
            Passing it in keeps guardrails on the same model and endpoint as the
            rest of the app.
        config_path (str | Path): Folder containing config.yml and prompts.yml.

    :returns
        LLMRails: Guardrails engine ready to check user messages.
    """
    rails_config = RailsConfig.from_path(str(config_path))
    return LLMRails(rails_config, llm=llm)


def check_input(rails: Any, user_input: str) -> InputCheck:
    """
    Run only the input rails on one user message.

    :param
        rails (LLMRails): Guardrails engine from load_input_rails().
        user_input (str): The user's raw message.

    :returns
        InputCheck: Whether the message was blocked, the refusal to show, and
        which rail fired.
    """
    # rails=["input"] runs only the input rails, so no answer is generated
    # here. output_vars asks NeMo to return the rail's decision variables.
    result = rails.generate(
        messages=[{"role": "user", "content": user_input}],
        options=GenerationOptions(
            rails=["input"],
            output_vars=["allowed", "triggered_input_rail", "bot_message"],
        ),
    )

    decision = result.output_data or {}
    triggered_rail = decision.get("triggered_input_rail")
    # A message is blocked if the rail explicitly disallowed it or named
    # itself as triggered (either signal alone is enough)
    blocked = decision.get("allowed") is False or bool(triggered_rail)

    if not blocked:
        return InputCheck(blocked=False)

    return InputCheck(
        blocked=True,
        bot_message=_refusal_text(result, decision),
        triggered_rail=triggered_rail,
    )


def _refusal_text(result: Any, decision: dict) -> str:
    """
    Pick the refusal message to show for a blocked input.

    :param
        result (GenerationResponse): Response from LLMRails.generate().
        decision (dict): The rail's output variables (result.output_data).

    :returns
        str: The bot message set by the rail, else the response content,
        else DEFAULT_BLOCKED_MESSAGE.
    """
    if decision.get("bot_message"):
        return decision["bot_message"]

    # Fall back to the last message in the response, if NeMo returned one
    response_messages = result.response if isinstance(result.response, list) else []
    if response_messages and response_messages[-1].get("content"):
        return response_messages[-1]["content"]

    return DEFAULT_BLOCKED_MESSAGE
