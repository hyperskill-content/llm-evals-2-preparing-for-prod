import json
import os
import sys
import traceback
import uuid

import dotenv
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.docstore.document import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langfuse import get_client, observe, propagate_attributes
from langfuse.langchain import CallbackHandler

# Load environment variables from .env file
dotenv.load_dotenv()

# Generate unique session_id and user_id once
session_id = f"session-{uuid.uuid4().hex[:8]}"
users = ["James", "George", "Mike", "Sherlock"]
user_id = users[uuid.uuid4().int % len(users)]

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")
COLLECTION_NAME = "smartphones"

# Initialize the LLM with OpenAI API credentials (substitute for other models)
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Initialize the embeddings model with OpenAI API credentials
embeddings_model = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDINGS_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    show_progress_bar=False,
)

# Initialize Redis-backed conversation history (with 1-hour TTL)
chat_history = RedisChatMessageHistory(
    session_id=session_id,
    url=REDIS_URL,
    ttl=3600,
)

# Initialize Langfuse client
langfuse = get_client()


# ---------------------------
# Load JSON Data and Connect to Qdrant Cloud Vector Store
# ---------------------------

@observe(name="embed_documents")
def embed_documents(json_path: str) -> QdrantVectorStore | list:
    """
    Validate the smartphone JSON file and connect to the existing Qdrant Cloud collection.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {json_path} was not found.")
        return []
    except json.JSONDecodeError as jde:
        print(f"Error decoding JSON from file {json_path}: {jde}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while reading {json_path}: {e}")
        return []

    documents = []
    for entry in data:
        content = (
            f"Model: {entry.get('model', '')}\n"
            f"Price: {entry.get('price', '')}\n"
            f"Rating: {entry.get('rating', '')}\n"
            f"SIM: {entry.get('sim', '')}\n"
            f"Processor: {entry.get('processor', '')}\n"
            f"RAM: {entry.get('ram', '')}\n"
            f"Battery: {entry.get('battery', '')}\n"
            f"Display: {entry.get('display', '')}\n"
            f"Camera: {entry.get('camera', '')}\n"
            f"Card: {entry.get('card', '')}\n"
            f"OS: {entry.get('os', '')}\n"
            f"In Stock: {entry.get('in_stock', '')}"
        )
        documents.append(Document(page_content=content))

    try:
        return QdrantVectorStore.from_existing_collection(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            embedding=embeddings_model,
            collection_name=COLLECTION_NAME,
        )
    except Exception as e:
        print(f"Error connecting to the existing vector store: {e}")
        traceback.print_exc()
        return []


# Initialize the vector store
product_db = embed_documents("datasets/smartphones.json")


# ---------------------------
# Tool Definitions
# ---------------------------
@tool("SmartphoneInfo")
def smartphone_info_tool(model: str) -> str:
    """
    Retrieves information about a smartphone model from the product database.
    """
    if not isinstance(product_db, QdrantVectorStore):
        return "Product database is unavailable."
    try:
        results = product_db.similarity_search(model, k=1)
        if not results:
            print(f"Info: No results found for model: {model}")
            return "Could not find information for the specified model."
        return results[0].page_content
    except Exception as e:
        return f"Error during smartphone information retrieval for model {model}: {e}"


# ---------------------------
# Tool Call Handling and Response Generation
# ---------------------------
@observe(name="generate_context")
def generate_context(ai_message: AIMessage) -> None:
    """
    Process tool calls from the language model and append the AI message and
    each tool's response to Redis conversation history.
    """
    chat_history.add_message(ai_message)

    if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
        chat_history.add_message(
            AIMessage(
                content="No tool calls found. Please ensure the model is configured to use tools."
            )
        )

    try:
        for tool_call in ai_message.tool_calls:
            if tool_call["name"] == "SmartphoneInfo":
                tool_output = smartphone_info_tool.invoke(tool_call)
                chat_history.add_message(tool_output)

    except Exception as e:
        print(f"An error occurred while processing tool calls: {e}")
        chat_history.add_message(
            AIMessage(
                content=f"An error occurred while processing tool calls: {e}"
            )
        )


# ---------------------------
# Main Conversation Loop
# ---------------------------
def main():
    tools = [smartphone_info_tool]
    llm_with_tools = llm.bind_tools(tools)

    # Fetch production prompts from Langfuse
    context_langfuse_prompt = langfuse.get_prompt("context")
    review_langfuse_prompt = langfuse.get_prompt("review")
    goodbye_langfuse_prompt = langfuse.get_prompt("goodbye")

    context_prompt = ChatPromptTemplate.from_messages(
        [
            *context_langfuse_prompt.get_langchain_prompt(),
            MessagesPlaceholder(variable_name="conversation"),
        ]
    )

    review_prompt = ChatPromptTemplate.from_messages(
        [
            *review_langfuse_prompt.get_langchain_prompt(),
            MessagesPlaceholder(variable_name="conversation"),
        ]
    )

    goodbye_prompt = PromptTemplate.from_template(
        goodbye_langfuse_prompt.get_langchain_prompt()[0][1]
    )

    context_chain = context_prompt | llm_with_tools | generate_context
    review_chain = review_prompt | llm
    goodbye_chain = goodbye_prompt | llm

    langfuse_handler = CallbackHandler()

    try:
        print("Welcome to the Smartphone Assistant! I can help you with smartphone features and comparisons.")
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() in ["exit", "quit", "bye", "end"]:
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="user-query",
                    input={"user_input": user_input},
                ) as span:
                    with propagate_attributes(
                        session_id=session_id,
                        user_id=user_id,
                    ):
                        goodbye_message = goodbye_chain.invoke(
                            {"user_id": user_id},
                            config={
                                "run_name": "goodbye-message",
                                "callbacks": [langfuse_handler],
                                "metadata": {"langfuse_prompt": goodbye_langfuse_prompt},
                            },
                        )

                        span.update(output={"response": goodbye_message.content})

                print(f"System: {goodbye_message.content}")

                feedback = input("\nWas this conversation helpful? (Yes/No): ").strip()
                user_comment = input("Please give us a reason for your answer. This will help us improve: ").strip()

                langfuse.create_score(
                    session_id=session_id,
                    name="conversation_usefulness",
                    value=feedback,
                    data_type="CATEGORICAL",
                    comment=user_comment,
                )

                langfuse.flush()
                print("\nThank you for your feedback!")
                break

            chat_history.add_user_message(user_input)

            with langfuse.start_as_current_observation(
                as_type="span",
                name="user-query",
                input={"user_input": user_input},
            ) as span:
                with propagate_attributes(
                    session_id=session_id,
                    user_id=user_id,
                ):
                    context_chain.invoke(
                        {"user_input": user_input, "conversation": chat_history.messages},
                        config={
                            "run_name": "context",
                            "callbacks": [langfuse_handler],
                            "metadata": {"langfuse_prompt": context_langfuse_prompt},
                        },
                    )

                    response = review_chain.invoke(
                        {"user_id": user_id, "user_input": user_input, "conversation": chat_history.messages},
                        config={
                            "run_name": "final-response",
                            "callbacks": [langfuse_handler],
                            "metadata": {"langfuse_prompt": review_langfuse_prompt},
                        },
                    )

                span.update(output={"response": response.content})

            print(f"System: {response.content}")
            chat_history.add_ai_message(response.content)

    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
        langfuse.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()