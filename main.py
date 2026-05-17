import os
import json

from dotenv import load_dotenv
from langfuse import Langfuse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# Initialize Langfuse client
langfuse_client = Langfuse()

# Load smartphone data
with open(os.path.join(os.path.dirname(__file__), "datasets", "smartphones.json")) as f:
    smartphones_data = json.load(f)

smartphones_context = json.dumps(smartphones_data[:20], indent=2)  # limit context size


def get_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def get_chain():
    """Build the chain using prompts fetched from Langfuse."""
    # Fetch prompt from Langfuse instead of hard-coding
    smartphone_prompt = langfuse_client.get_prompt("smartphone-assistant")

    prompt = ChatPromptTemplate.from_messages(
        [
            smartphone_prompt.get_langchain_prompt()[0],  # system message
            MessagesPlaceholder(variable_name="chat_history"),
            smartphone_prompt.get_langchain_prompt()[1],  # user message
        ]
    )

    # Link prompt to Langfuse observations for metrics tracking
    prompt.metadata = {"langfuse_prompt": smartphone_prompt}

    llm = get_llm()
    chain = prompt | llm
    return chain


def chat(user_input: str, chat_history: list = None):
    """Process a user message and return the assistant's response."""
    if chat_history is None:
        chat_history = []

    chain = get_chain()
    response = chain.invoke({
        "user_input": user_input,
        "chat_history": chat_history,
        "context": smartphones_context,
    })
    return response.content


def main():
    """Interactive chat loop."""
    print("Smartphone Info Bot (type 'quit' to exit)")
    print("-" * 40)

    chat_history = []

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break

        response = chat(user_input, chat_history)
        print(f"\nAssistant: {response}")

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))


if __name__ == "__main__":
    main()
