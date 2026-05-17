import os
import json

from dotenv import load_dotenv
from langfuse import Langfuse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_redis import RedisChatMessageHistory
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

load_dotenv()

# Initialize Langfuse client
langfuse_client = Langfuse()

# Redis configuration
REDIS_URL = os.getenv("REDIS_CONNECTION_STRING", "redis://localhost:6380/0")
CHAT_HISTORY_TTL = int(os.getenv("CHAT_HISTORY_TTL", "3600"))  # 1 hour default

# Load smartphone data
with open(os.path.join(os.path.dirname(__file__), "datasets", "smartphones.json")) as f:
    smartphones_data = json.load(f)

smartphones_context = json.dumps(smartphones_data[:20], indent=2)  # limit context size

# Guardrails configuration
guardrails_config = RailsConfig.from_path(
    os.path.join(os.path.dirname(__file__), "config")
)

BLOCKED_RESPONSE = "I'm sorry, I can't respond to that."


def get_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def get_redis_history(session_id: str) -> BaseChatMessageHistory:
    """Return a Redis-backed chat history for the given session."""
    return RedisChatMessageHistory(session_id, redis_url=REDIS_URL, ttl=CHAT_HISTORY_TTL)


def get_chain_with_history():
    """Build the chain with Redis message history and trimming."""
    smartphone_prompt = langfuse_client.get_prompt("smartphone-assistant")

    prompt = ChatPromptTemplate.from_messages(
        [
            smartphone_prompt.get_langchain_prompt()[0],  # system message
            MessagesPlaceholder(variable_name="chat_history"),
            smartphone_prompt.get_langchain_prompt()[1],  # user message
        ]
    )

    prompt.metadata = {"langfuse_prompt": smartphone_prompt}

    llm = get_llm()

    # Trimmer to prevent chat history from growing too long
    trimmer = trim_messages(
        strategy="last",
        token_counter=len,  # approximate: count by number of messages
        max_tokens=20,  # keep last 20 messages max
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
    )

    chain = prompt | trimmer | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_redis_history,
        input_messages_key="user_input",
        history_messages_key="chat_history",
    )

    return chain_with_history


def chat(user_input: str, session_id: str = "default_session"):
    """Process a user message and return the assistant's response."""
    # Apply guardrails to check input via context chain
    llm = get_llm()
    rails = RunnableRails(guardrails_config, input_key="user_input", output_key="output")
    context_chain_with_rails = rails | llm

    rail_response = context_chain_with_rails.invoke({"user_input": user_input})

    # Check if the input rail was triggered
    if BLOCKED_RESPONSE in rail_response.content:
        return BLOCKED_RESPONSE

    # Input passed guardrails, proceed with the full chain
    chain = get_chain_with_history()
    response = chain.invoke(
        {"user_input": user_input, "context": smartphones_context},
        config={"configurable": {"session_id": session_id}},
    )
    return response.content


def main():
    """Interactive chat loop."""
    print("Smartphone Info Bot (type 'quit' to exit)")
    print("-" * 40)

    session_id = "hyperskill_user"
    total_usage = 0.0

    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break

        response = chat(user_input, session_id)
        print(f"System: {response}")
        print(f"Your usage so far: {total_usage}")


if __name__ == "__main__":
    main()
