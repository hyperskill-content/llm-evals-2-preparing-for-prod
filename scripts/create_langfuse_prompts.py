"""
Create (or add a new version of) the chatbot's prompts in Langfuse.

Run this once per Langfuse project before running main.py:

    uv run python scripts/create_langfuse_prompts.py

Each run creates a NEW version of every prompt and moves the "production" label
to it. After the first run, edit prompts in the Langfuse UI (or edit this file
and rerun it) rather than changing main.py.

The prompt text below is the text that was previously hard-coded in main.py,
with two changes:
  - variables use Langfuse's {{variable}} syntax instead of {variable}
  - the leading indentation from the old triple-quoted strings is removed
"""

import sys
from inspect import cleandoc
from pathlib import Path

import dotenv
from langfuse import get_client

# Make the project root importable so the prompt names and label are defined in
# one place (prompts.py) and shared with main.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts import (  # noqa: E402
    CONTEXT_PROMPT_NAME,
    GOODBYE_PROMPT_NAME,
    PROMPT_LABEL,
    REVIEW_PROMPT_NAME,
)

# Decides which smartphone models to look up with the SmartphoneInfo tool
CONTEXT_SYSTEM_PROMPT = cleandoc("""
    You're part of a smartphone recommendation system. Your work is to use the SmartphoneInfo tool to retrieve information about smartphones based on user queries.
      - If the user requests specs/comparisons/recommendations for a model explicitly mentioned in chat or can be inferred from the conversation history, call SmartphoneInfo(model)
      - If multiple models are mentioned or from the conversation history, you must call SmartphoneInfo for each model separately.
      - If the user asks a general question, do nothing.
    Do not guess or recommend a model from internal knowledge; the model name must be clear from the chat history or user input.

    Current question: {{user_input}}
""")

# Writes the final review or comparison shown to the user
REVIEW_SYSTEM_PROMPT = cleandoc("""
    You are an expert AI assistant helping customers pick the best smartphone from our catalog. Follow these rules strictly:

    1. Focus solely on concise (under 100 words), human-like, personalized reviews/comparisons of models named in the user's query or provided context.
    2. Think step by step before answering.
    3. Never guess or recommend any model not explicitly mentioned in the context or query.
    4. If no model is given, ask the user to check our online catalog for the exact model name.
    5. DO NOT assist with ordering, returns, tracking, or other general support.
    6. If asked about anything outside smartphone features/comparisons, respond that you can't help.
    7. If the user only wants to chat, engage briefly, but always steer back to smartphone comparisons.
    8. Never list smartphone specifications, but instead explain how they translate to real-world benefits.

    When recommending, evaluate performance, display, battery, camera, and special functions (e.g., 5G, fast charging, expandable storage), and how they translate to real-world benefits.
    Always confirm the user's needs before finalizing.

    Current user: {{user_id}}
    Current question: {{user_input}}
""")

# Writes the goodbye message when the user ends the session
GOODBYE_SYSTEM_PROMPT = cleandoc("""
    You have been helping the user: {{user_id}} with smartphone features and comparisons.
    Generate a short friendly goodbye message for the user and also thank them for their feedback.
    (note that you've already been helping the user with smartphone features and comparisons, so do not repeat that or greet them again)
""")


def create_prompts(langfuse_client) -> None:
    """
    Create a new production version of each chatbot prompt in Langfuse.

    The context and review prompts are chat prompts holding only the system
    message (main.py adds the chat history placeholder). The goodbye prompt is a
    text prompt because main.py uses it with a plain PromptTemplate.

    :param
        langfuse_client: Langfuse client (from langfuse.get_client()).

    :returns
        None. Prints the name and version of each prompt created.
    """
    # (name, prompt body, Langfuse prompt type) for each prompt
    prompt_specs = [
        (CONTEXT_PROMPT_NAME, [{"role": "system", "content": CONTEXT_SYSTEM_PROMPT}], "chat"),
        (REVIEW_PROMPT_NAME, [{"role": "system", "content": REVIEW_SYSTEM_PROMPT}], "chat"),
        (GOODBYE_PROMPT_NAME, GOODBYE_SYSTEM_PROMPT, "text"),
    ]

    for prompt_name, prompt_body, prompt_type in prompt_specs:
        created_prompt = langfuse_client.create_prompt(
            name=prompt_name,
            prompt=prompt_body,
            type=prompt_type,
            labels=[PROMPT_LABEL],
            commit_message="Move prompt from main.py into Langfuse",
        )
        print(f"Created {prompt_type} prompt '{prompt_name}' version {created_prompt.version}")


if __name__ == "__main__":
    # Langfuse credentials come from .env (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST)
    dotenv.load_dotenv()
    create_prompts(get_client())
