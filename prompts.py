"""
Helpers for loading the chatbot's prompts from Langfuse prompt management.

The prompts used to be hard-coded in main.py. They now live in Langfuse, where
they are versioned and labelled, so they can be edited, compared and rolled back
without changing or redeploying the application code.

Each helper fetches one prompt by name and converts it into the LangChain
template type that the chains in main.py expect. The Langfuse prompt object is
attached to the template's metadata so that the Langfuse CallbackHandler links
every generation to the prompt version that produced it. This link is what
populates the prompt's "Metrics" and "Linked generations" views in Langfuse.
"""

from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

# Langfuse label that selects which prompt version the application uses.
# Moving the "production" label to another version in the Langfuse UI changes
# the prompt the app uses (after the client cache expires) with no code change.
PROMPT_LABEL = "production"

# Names of the prompts in Langfuse. These names are required by the task spec.
CONTEXT_PROMPT_NAME = "context_system_prompt"
REVIEW_PROMPT_NAME = "review_system_prompt"
GOODBYE_PROMPT_NAME = "goodbye_system_prompt"


def load_chat_prompt(
    langfuse_client: Any,
    prompt_name: str,
    history_variable: str = "conversation",
) -> ChatPromptTemplate:
    """
    Fetch a Langfuse chat prompt and build a ChatPromptTemplate with chat history.

    The Langfuse prompt holds only the system message. The chat history is added
    here as a MessagesPlaceholder after the system message, which matches the
    message order the hard-coded prompts in main.py used before.

    :param
        langfuse_client: Langfuse client (from langfuse.get_client()).
        prompt_name (str): Name of the chat prompt in Langfuse.
        history_variable (str): Name of the chain input that holds the list of
            chat messages (default "conversation", as used in main.py).

    :returns
        ChatPromptTemplate: The system message followed by the chat history
        placeholder, with the Langfuse prompt stored in its metadata.
    """
    # Fetch the version of the prompt that currently carries the production label.
    # The client caches prompts locally, so this does not hit the API on every call.
    langfuse_prompt = langfuse_client.get_prompt(prompt_name, label=PROMPT_LABEL, type="chat")

    # get_langchain_prompt() converts Langfuse's {{variable}} syntax to LangChain's
    # {variable} syntax and returns (role, content) tuples. Index 0 is the system message.
    system_message = langfuse_prompt.get_langchain_prompt()[0]

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            system_message,
            MessagesPlaceholder(variable_name=history_variable),
        ]
    )

    # Link this template to the Langfuse prompt version for tracing and metrics
    chat_prompt.metadata = {"langfuse_prompt": langfuse_prompt}
    return chat_prompt


def load_text_prompt(langfuse_client: Any, prompt_name: str) -> PromptTemplate:
    """
    Fetch a Langfuse text prompt and build a PromptTemplate from it.

    :param
        langfuse_client: Langfuse client (from langfuse.get_client()).
        prompt_name (str): Name of the text prompt in Langfuse.

    :returns
        PromptTemplate: Template built from the prompt text, with the Langfuse
        prompt stored in its metadata.
    """
    # Fetch the production version of the text prompt
    langfuse_prompt = langfuse_client.get_prompt(prompt_name, label=PROMPT_LABEL, type="text")

    # For text prompts, get_langchain_prompt() returns a single string with
    # {{variable}} already converted to {variable}
    text_prompt = PromptTemplate.from_template(langfuse_prompt.get_langchain_prompt())

    # Link this template to the Langfuse prompt version for tracing and metrics
    text_prompt.metadata = {"langfuse_prompt": langfuse_prompt}
    return text_prompt
