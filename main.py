# start where you left off in part one of this series
# I will use my code from LLM-eval Step 3 (no code was changed on following steps)
import json
import os
import sys
import uuid

import dotenv
from langchain_community.docstore.document import Document
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import trim_messages
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisChatMessageHistory
from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

dotenv.load_dotenv()

# Intention: Your code should now collect user feedback and send it to Langfuse.
# When you run the code and send some queries, you should see the scores in the web UI
# showing the user feedback and comments for that trace.

users = ["James", "George", "Mike", "Sherlock"]
user_id = users[uuid.uuid4().int % len(users)]
session_name = f"session-{uuid.uuid4().hex[:8]}"
servicename = os.getenv("OTEL_SERVICE_NAME")
os.environ["OTEL_SERVICE_NAME"] = os.getenv("OTEL_SERVICE_NAME")
basic_tag = "LLM-eval2"

REDIS_URL = "redis://localhost:6380/0"
HISTORY_TTL_SECONDS = 3600

langfuse_handler = CallbackHandler()

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


# ---------------------------
# Load JSON Data and Build Qdrant Vector Store
# ---------------------------

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
def generate_context(ai_message: AIMessage, config: dict):
    """
    Processes tool calls and stores tool responses in Redis history.
    """

    # session_id = config["configurable"]["session_id"]
    # history = get_redis_history(session_id)
    #
    # # if no tool calls
    # if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
    #     return ai_message
    #
    # for tool_call in ai_message.tool_calls:
    #     if tool_call["name"] == "SmartphoneInfo":
    #         # execute tool
    #         tool_output = smartphone_info_tool.invoke(tool_call["args"]["model"])
    #         # save Tool-Response as a message
    #         tool_message = ToolMessage(
    #             content=tool_output,
    #             tool_call_id=tool_call["id"],
    #         )
    #         # IMPORTANT: Save message in Redis history
    #         history.add_message(tool_message)

    return ai_message

def _update_trace(run_name: str):
    lf = get_client()
    lf.update_current_trace(
        name=run_name,
        session_id=session_name,
        user_id=user_id,
        input={
            "user_id": user_id,
            "session": session_name
        }
    )

def get_redis_history(session_id: str) -> BaseChatMessageHistory:
    return RedisChatMessageHistory(
        session_id=session_id,
        redis_url=REDIS_URL,
        ttl=HISTORY_TTL_SECONDS,
    )

trimmer = trim_messages(
    strategy="last",
    token_counter=llm,
    max_tokens=500,
    start_on="human",
    end_on=("human", "tool"),
    include_system=True,
)

# ---------------------------
# Main Conversation Loop
# ---------------------------
@observe(name="main")
def main():
    _update_trace('ai-response')
    # List of available tools
    tools = [smartphone_info_tool]

    # Bind the tools to the language model instance
    llm_with_tools = llm.bind_tools(tools)

    langfuse = get_client()


    lf_context_prompt = langfuse.get_prompt(name="context-prompt", label="production")
    context_messages = lf_context_prompt.get_langchain_prompt()

    context_prompt = ChatPromptTemplate.from_messages(
        [
            context_messages[0],  # system message from langfuse
            MessagesPlaceholder(variable_name="conversation"),
            ("human", "{user_input}"),
        ],
    )

    lf_review_prompt = langfuse.get_prompt(name="review-prompt", label="production")
    review_messages = lf_review_prompt.get_langchain_prompt()

    review_prompt = ChatPromptTemplate.from_messages(
        [
            review_messages[0],
            MessagesPlaceholder(variable_name="conversation"),
        ]
    )

    lg_goodbye_prompt = langfuse.get_prompt(name="goodbye-prompt", label="production")
    goodbye_prompt = ChatPromptTemplate.from_messages(
        [
            lg_goodbye_prompt.get_langchain_prompt()[0],
            MessagesPlaceholder(variable_name="conversation"),
            ("human", "{user_input}"),
        ]
    )

    context_chain = context_prompt | trimmer | llm_with_tools | RunnableLambda(generate_context)

    review_chain = review_prompt | llm

    goodbye_chain = goodbye_prompt | llm

    context_chain_with_history = RunnableWithMessageHistory(
        context_chain,
        get_redis_history,
        input_messages_key="user_input",
        history_messages_key="conversation",
    )

    goodbye_chain_with_history = RunnableWithMessageHistory(
        goodbye_chain,
        get_redis_history,
        input_messages_key="user_input",
        history_messages_key="conversation",
    )

    try:
        print(f"Welcome to the Smartphone Assistant! I can help you with smartphone features and comparisons. SessionID:", session_name)
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() in ["exit", "quit", "bye", "end"]:
                goodbye_message = goodbye_chain_with_history.invoke(
                    {"user_id": user_id, "user_input": user_input},
                    config={
                        "run_name": "goodbye-message",
                        "callbacks": [langfuse_handler],
                        "configurable": {"session_id": session_name},
                        "metadata": {
                            "langfuse_session_id": session_name,
                            "langfuse_user_id": user_id,
                            "langfuse_tags": [basic_tag + "-goodbye"],
                            "langfuse_prompt": lg_goodbye_prompt,
                        },
                    },
                )
                lf = get_client()
                lf.update_current_trace(
                    metadata={"last_user_input": user_input}
                )
                print(f"System: {goodbye_message.content}")
                feedback = input("Was this answer helpful? (Yes/No): ")
                feedback = feedback.strip().lower()
                valid_yes = ["yes", "y"]
                feedback = "Yes" if feedback in valid_yes else "No"
                user_comment = input("Please give us a reason ...: ")
                langfuse_client = get_client()
                langfuse_client.score_current_trace(
                    name="usefulness",
                    value=feedback,
                    data_type="CATEGORICAL",
                    comment=user_comment
                )
                break

            context_result = context_chain_with_history.invoke(
                {"user_input": user_input, "user_id": user_id},
                config={
                    "run_name": "context",
                    "callbacks": [langfuse_handler],
                    "configurable": {"session_id": session_name},
                    "metadata": {
                        "langfuse_session_id": session_name,
                        "langfuse_user_id": user_id,
                        "langfuse_tags": [basic_tag + "-context"],
                        "langfuse_prompt": lf_context_prompt,
                    },
                },
            )

            history = get_redis_history(session_name)

            if hasattr(context_result, "tool_calls") and context_result.tool_calls:
                for tool_call in context_result.tool_calls:
                    if tool_call["name"] == "SmartphoneInfo":
                        tool_output = smartphone_info_tool.invoke(tool_call["args"]["model"])

                        tool_message = ToolMessage(
                            content=tool_output,
                            tool_call_id=tool_call["id"],
                        )

                        history.add_message(tool_message)

            response = review_chain.invoke(
                {
                    "user_id": user_id,
                    "conversation": history.messages,
                },
                config={
                    "run_name": "final-response",
                    "callbacks": [langfuse_handler],
                    "metadata": {
                        "langfuse_session_id": session_name,
                        "langfuse_user_id": user_id,
                        "langfuse_tags": [basic_tag + "-final-response"],
                        "langfuse_prompt": lf_review_prompt,
                    },
                },
            )

            history.add_message(response)

            lf = get_client()
            lf.update_current_trace(
                output=response.content
            )
            print(f"System: {response.content}")

    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
