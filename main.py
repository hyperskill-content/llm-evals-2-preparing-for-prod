import json
import os
import sys
import uuid

import dotenv
from langchain_community.docstore.document import Document
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage, trim_messages
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisChatMessageHistory
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

# Load environment variables from .env file
dotenv.load_dotenv()

users = ["James", "George", "Mike", "Sherlock"]
user_id = users[uuid.uuid4().int % len(users)]
session_name = f"session-{uuid.uuid4().hex[:8]}"

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

REDIS_URL = os.getenv("REDIS_CONNECTION_STRING", "redis://localhost:6380/0")
chat_history = RedisChatMessageHistory(session_id=session_name, redis_url=REDIS_URL, ttl=3600)


trimmer = trim_messages(
    strategy="last",
    token_counter=llm,
    max_tokens=500,
    start_on="human",
    end_on=("human", "tool"),
    include_system=True,
)


# ---------------------------
# Load JSON Data and Build Qdrant Vector Store
# ---------------------------

@observe(name="embed-documents")
def embed_documents(json_path: str):
    """
    Load JSON data from the smartphones.json file and convert each entry to a Document.
    :param
        json_path (str): Path to the JSON file containing smartphone data.

    :returns
        Qdrant vector store A Qdrant vector store built from the smartphone documents,
                or an empty list if an error occurs.
    """
    langfuse_client = get_client()
    langfuse_client.update_current_trace(
        session_id=session_name,
        user_id=user_id,
    )

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
def generate_context(ai_message: AIMessage) -> dict:
    """
    Process tool calls from the language model and collect their responses as ToolMessage objects.

    :param
        ai_message (AIMessage): The language model's output message containing tool_calls.

    :returns
        A dictionary containing a list of ToolMessage objects under the key "tool_responses".
    """
    langfuse_client = get_client()
    langfuse_client.update_current_trace(
        session_id=session_name,
        user_id=user_id,
    )

    # construct the conversation history with the AI message containing tool calls
    chat_history.add_message(ai_message)

    # Check if the AI message has any tool calls
    if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
        chat_history.add_message(
            AIMessage(
                content="No tool calls found. Please ensure the model is configured to use tools."
            )
        )

    try:
        # Process each tool call, invoke the appropriate tool, and append the result to the conversation
        # a message with tool calls is expected to be followed by tool responses
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
@observe(name="main")
def main():
    langfuse_client = get_client()
    langfuse_client.update_current_trace(
        session_id=session_name,
        user_id=user_id,
    )

    langfuse_handler = CallbackHandler()

    # List of available tools
    tools = [smartphone_info_tool]

    # Bind the tools to the language model instance
    llm_with_tools = llm.bind_tools(tools)

    context_system_prompt = langfuse_client.get_prompt("context-system-prompt")
    review_system_prompt =  langfuse_client.get_prompt("review-system-prompt")
    goodbye_system_prompt = langfuse_client.get_prompt("goodbye-system-prompt")

    context_prompt = ChatPromptTemplate.from_messages(
        context_system_prompt.get_langchain_prompt()
    )
    context_prompt.metadata = {"langfuse_prompt": context_system_prompt}

    review_prompt = ChatPromptTemplate.from_messages(
        review_system_prompt.get_langchain_prompt()
    )
    review_prompt.metadata = {"langfuse_prompt": review_system_prompt}

    goodbye_prompt = PromptTemplate.from_template(
        goodbye_system_prompt.get_langchain_prompt()
    )
    goodbye_prompt.metadata = {"langfuse_prompt": goodbye_system_prompt}

    rails_config = RailsConfig.from_path("config")
    rails = RunnableRails(rails_config, input_key="user_input")

    context_chain = context_prompt | llm_with_tools | generate_context
    review_chain = review_prompt | llm

    goodbye_chain = goodbye_prompt | llm

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
                            "langfuse_session_id": session_name,
                            "langfuse_user_id": user_id,
                        },
                    }
                )

                feedback = input("Was this answer helpful? (Yes/No): ").strip()
                user_comment = input("Please give us a reason for your answer. This will help us improve: ").strip()

                langfuse_client.score_current_trace(
                    name="usefulness",
                    value=feedback,
                    data_type="CATEGORICAL",
                    comment=user_comment
                )

                langfuse_client.score_current_trace(
                    name="Context recall",
                    value=0.3,
                    data_type="NUMERIC",
                    comment="Context recall, so and so"
                )

                langfuse_client.score_current_trace(
                    name="Answer relevancy",
                    value="1",
                    data_type="CATEGORICAL",
                    comment="Relevancy, good enough"
                )

                langfuse_client.score_current_trace(
                    name="Faithfulness",
                    value="0.9",
                    data_type="CATEGORICAL",
                    comment="Mostly correct"
                )

                print(f"System: {goodbye_message.content}")
                break

            chat_history.add_message(HumanMessage(content=user_input))

            guard_result = rails.invoke({"user_input": user_input})
            if guard_result.get("output", "").strip() == "I'm sorry, I can't respond to that.":
                print(f"System: {guard_result['output']}")
                chat_history.add_message(AIMessage(content=guard_result["output"]))
                continue

            trimmed_conversation = trimmer.invoke(chat_history.messages)

            context_chain.invoke(
                {"user_input": user_input, "conversation": trimmed_conversation},
                config={
                    "run_name": "context",
                    "callbacks": [langfuse_handler],
                    "metadata": {
                        "langfuse_session_id": session_name,
                        "langfuse_user_id": user_id,
                    },
                }
            )

            response = review_chain.invoke(
                {"user_id": user_id, "user_input": user_input, "conversation": chat_history.messages},
                config={
                    "run_name": "final-response",
                    "callbacks": [langfuse_handler],
                    "metadata": {
                        "langfuse_session_id": session_name,
                        "langfuse_user_id": user_id,
                    },
                }
            )

            print(f"System: {response.content}")
            chat_history.add_message(response)

    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
