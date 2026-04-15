import json
import logging
import os
import sys
import uuid

import dotenv
from langchain_community.docstore.document import Document
from langchain_core.messages import AIMessage, trim_messages
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisChatMessageHistory
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from langfuse import observe, Langfuse
from langfuse.langchain import CallbackHandler
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Load environment variables from .env file
dotenv.load_dotenv()
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(name)s %(levelname)s   %(message)s', datefmt='%H:%M:%S')
# Ensure only guardrails actions are logged at INFO level to match task_3.md requirements
logging.getLogger("actions.py").setLevel(logging.INFO)

# Initialize Langfuse Callback Handler
langfuse_handler = CallbackHandler()
langfuse_client = Langfuse()

total_cost = 0.0


def update_usage(response):
    global total_cost
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        usage = response.usage_metadata
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        # gpt-4o-mini prices: $0.15 / 1M input, $0.60 / 1M output
        cost = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000
        total_cost += cost
    return total_cost

users = ["James", "George", "Mike", "Sherlock"]
user_id = users[uuid.uuid4().int % len(users)]

# Initialize the LLM with OpenAI API credentials (substitute for other models)
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize the embeddings model with OpenAI API credentials
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    show_progress_bar=True,
)

# Initialize Redis connection for chat history
REDIS_URL = "redis://localhost:6380/0"


def get_redis_history(session_id: str):
    """
    Retrieves or creates a RedisChatMessageHistory instance for the given session ID.
    :param session_id: Unique identifier for the chat session.
    :returns: A RedisChatMessageHistory object.
    """
    return RedisChatMessageHistory(session_id, redis_url=REDIS_URL, ttl=600)  # Setting TTL to 10 minutes


def get_history_messages(session_id: str):
    """
    Retrieves and repairs chat history from Redis, restoring lost tool_calls in AIMessages.
    :param session_id: Unique identifier for the chat session.
    :returns: A list of Message objects.
    """
    history = get_redis_history(session_id)
    messages = history.messages
    for m in messages:
        if isinstance(m, AIMessage) and "_tool_calls" in m.additional_kwargs:
            try:
                # Restore tool_calls from serialized additional_kwargs
                m.tool_calls = json.loads(m.additional_kwargs["_tool_calls"])
            except Exception:
                pass
    return messages


def embed_documents(json_path: str):
    """
    Load JSON data from the smartphones.json file and convert each entry to a Document.
    :param
        json_path (str): Path to the JSON file containing smartphone data.

    :returns
        Qdrant vector store A Qdrant vector store built from the smartphone documents,
                or an empty list if an error occurs.
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
        # Build a readable content string from the JSON entry
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
        collection_name = "smartphones"
        qdrant_client = QdrantClient("http://localhost:6333")

        collection_exists = qdrant_client.collection_exists(collection_name=collection_name)
        if not collection_exists:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1536,
                    distance=Distance.COSINE,
                ),
            )

            qdrant_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=collection_name,
                embedding=embeddings_model
            )

            qdrant_store.add_documents(documents=documents)

            return qdrant_store

        # no need to create a vector store every time
        else:
            qdrant_store = QdrantVectorStore.from_existing_collection(
                embedding=embeddings_model,
                collection_name=collection_name,
            )

            return qdrant_store

    except Exception as e:
        print(f"Error initializing the vector store: {e}")
        return []


# ---------------------------
# Tool Definitions
# ---------------------------
@tool("SmartphoneInfo")
def smartphone_info_tool(model: str) -> str:
    """
    Retrieves information about a smartphone model from the product database.

    :param
        model (str): The smartphone model to search for.

    :returns
        str: The smartphone's specifications, price, and availability,
             or an error message if not found or if an error occurs.
    """
    product_db = embed_documents("datasets/smartphones.json")
    try:
        results = product_db.similarity_search(model, k=1)
        if not results:
            print(f"Info: No results found for model: {model}")
            return "Could not find information for the specified model."
        info = results[0].page_content
        return info
    except Exception as e:
        return f"Error during smartphone information retrieval for model {model}: {e}"


# ---------------------------
# Tool Call Handling and Response Generation
# ---------------------------
@observe(name="generate-context")
def generate_context(ai_message: AIMessage, session_id: str) -> None:
    """
    Process tool calls from the language model and collect their responses as ToolMessage objects
    directly into Redis chat history.

    :param
        ai_message (AIMessage): The language model's output message containing tool_calls.
        session_id (str): The current chat session ID.
    """
    history = get_redis_history(session_id)

    # Workaround for tool_calls serialization bug in langchain_redis
    if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
        ai_message.additional_kwargs["_tool_calls"] = json.dumps(ai_message.tool_calls)

    history.add_message(ai_message)

    # Check if the AI message has any tool calls
    if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
        return

    try:
        # Process each tool call, invoke the appropriate tool, and append the result to Redis
        for tool_call in ai_message.tool_calls:
            if tool_call["name"] == "SmartphoneInfo":
                tool_output = smartphone_info_tool.invoke(tool_call)
                from langchain_core.messages import ToolMessage
                history.add_message(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))

    except Exception as e:
        print(f"An error occurred while processing tool calls: {e}")
        history.add_ai_message(f"An error occurred while processing tool calls: {e}")


# ---------------------------
# Main Conversation Loop
# ---------------------------
@observe(name="ai-response")
def main():
    session_id = f"session-{uuid.uuid4().hex[:8]}"
    user_id = users[uuid.uuid4().int % len(users)]
    history = get_redis_history(session_id)

    # Define a trimmer to manage message history length
    trimmer = trim_messages(
        strategy="last",
        token_counter=llm,
        max_tokens=600,
        start_on="human",
        include_system=True,
    )

    # List of available tools
    tools = [smartphone_info_tool]

    # Bind the tools to the language model instance
    llm_with_tools = llm.bind_tools(tools)

    # Get the prompts from Langfuse
    context_system_prompt = langfuse_client.get_prompt("context_system_prompt")
    review_system_prompt = langfuse_client.get_prompt("review_system_prompt")
    goodbye_system_prompt = langfuse_client.get_prompt("goodbye_system_prompt")

    context_prompt = ChatPromptTemplate.from_messages(
        [
            context_system_prompt.get_langchain_prompt()[0],
            MessagesPlaceholder(variable_name="conversation"),
            context_system_prompt.get_langchain_prompt()[1],
        ]
    )
    context_prompt.metadata = {"langfuse_prompt": context_system_prompt}

    review_prompt = ChatPromptTemplate.from_messages(
        [
            review_system_prompt.get_langchain_prompt()[0],
            MessagesPlaceholder(variable_name="conversation"),
            review_system_prompt.get_langchain_prompt()[1]
        ]
    )
    review_prompt.metadata = {"langfuse_prompt": review_system_prompt}

    goodbye_prompt = ChatPromptTemplate.from_messages(
        [
            goodbye_system_prompt.get_langchain_prompt()[0]
        ]
    )
    goodbye_prompt.metadata = {"langfuse_prompt": goodbye_system_prompt}

    context_chain = context_prompt | llm_with_tools
    review_chain = review_prompt | llm
    goodbye_chain = goodbye_prompt | llm

    # Load rails config
    rails_config = RailsConfig.from_path("config")
    # create an instance of the guardrails
    rails = RunnableRails(rails_config, input_key="user_input")

    # wrap the context chain with rails
    def context_chain_with_rails_invoke(input_data, config=None):
        # We call rails with only user_input to avoid serialization error
        # We need to preserve the result if triggered
        res = rails.invoke({"user_input": input_data["user_input"]}, config=config)
        if isinstance(res, dict) and res.get("output") == "I'm sorry, I can't respond to that.":
            return res
        # If not triggered, call the real chain
        return context_chain.invoke(input_data, config=config)

    context_chain_with_rails = RunnableLambda(context_chain_with_rails_invoke)

    try:
        print("Welcome to the Smartphone Assistant! I can help you with smartphone features and comparisons.")
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() in ["exit", "quit", "bye", "end"]:
                goodbye_message = goodbye_chain.invoke(
                    {"user_id": user_id},
                    config={
                        "run_name": "goodbye-message",
                        "callbacks": [langfuse_handler],
                        "metadata": {
                            "langfuse_session_id": session_id,
                            "langfuse_user_id": user_id,
                        },
                    }
                )
                # collect feedback
                feedback = input("Was this answer helpful? (Yes/No): ")
                user_comment = input("Please give us a reason for your answer. This will help us improve: ")

                # associate the score with that trace and push scores and comments to Langfuse
                langfuse_client.score_current_trace(
                    name="usefulness",
                    value=feedback.upper(),
                    data_type="CATEGORICAL",
                    comment=user_comment
                )

                print(f"System: {goodbye_message.content}")
                break

            # Add human message to Redis history
            history.add_user_message(user_input)

            # Retrieve and trim messages for context
            conversation = trimmer.invoke(get_history_messages(session_id))

            ai_msg_with_tools = context_chain_with_rails.invoke(
                {"user_input": user_input, "conversation": conversation},
                config={
                    "run_name": "context",
                    "callbacks": [langfuse_handler],
                    "metadata": {
                        "langfuse_session_id": session_id,
                        "langfuse_user_id": user_id,
                    },
                }
            )

            # Check if input rail was triggered
            if isinstance(ai_msg_with_tools, dict) and ai_msg_with_tools.get("output") == "I'm sorry, I can't respond to that.":
                print(f"System: {ai_msg_with_tools['output']}")
                print(f"Your usage so far: {total_cost}")
                continue

            update_usage(ai_msg_with_tools)

            # Process tool calls and update Redis history
            generate_context(ai_msg_with_tools, session_id)

            # Update conversation after tools for the final response
            conversation = trimmer.invoke(get_history_messages(session_id))

            response = review_chain.invoke(
                {"user_id": user_id, "user_input": user_input, "conversation": conversation},
                config={
                    "run_name": "final-response",
                    "callbacks": [langfuse_handler],
                    "metadata": {
                        "langfuse_session_id": session_id,
                        "langfuse_user_id": user_id,
                    },
                }
            )

            print(f"System: {response.content}")
            update_usage(response)
            print(f"Your usage so far: {total_cost}")
            # Add final AI response to Redis history
            history.add_ai_message(response.content)

    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
# start where you left off in part one of this series
