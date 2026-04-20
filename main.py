import json
import os
import sys
import uuid

import dotenv
from langchain_community.docstore.document import Document
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisChatMessageHistory
from langfuse import observe, get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from openai import AuthenticationError, BadRequestError
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Load environment variables from .env file
dotenv.load_dotenv()

REDIS_URL = "redis://localhost:6380/0"

langfuse_client = get_client()
langfuse_handler = CallbackHandler()

session_id = f"session-{uuid.uuid4().hex[:8]}"
user_id = f"user-{uuid.uuid4().hex[:8]}"

# Initialize the LLM with OpenAI API credentials (substitute for other models)
llm = ChatOpenAI(
    model=os.getenv("LITELLM_MODEL"),
    base_url=os.getenv("LITELLM_BASE_URL"),
    api_key=os.getenv("LITELLM_API_KEY"),
    model_kwargs={"user": "Nush"}
)

trimmer = trim_messages(
    strategy="last",
    token_counter=llm,
    max_tokens=500,
    start_on="human",
    end_on=("human", "tool"),
    include_system=True
)

rails_config = RailsConfig.from_path("config")
my_rails = RunnableRails(
    config=rails_config,
    input_key="user_input"
)

NEGATIVE_RESPONSE = "I'm sorry, I can't respond to that."

# Initialize the embeddings model with OpenAI API credentials
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    show_progress_bar=True
)


def get_redis_history(sid: str) -> BaseChatMessageHistory:
    return RedisChatMessageHistory(
        session_id=sid,
        redis_url=REDIS_URL,
        ttl=3600
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
    with propagate_attributes(
        trace_name="store-documents",
        session_id=session_id,
        user_id=user_id
    ):
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


# Initialize the vector store
product_db = embed_documents("datasets/smartphones.json")


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
def generate_context(ai_message: AIMessage) -> BaseMessage:
    """
    Process tool calls from the language model and collect their responses as ToolMessage objects.

    :param
        ai_message (AIMessage): The language model's output message containing tool_calls.

    :returns
        A BaseMessage containing the combined tool outputs.
    """
    with propagate_attributes(
        trace_name="ai-response",
        session_id=session_id,
        user_id=user_id
    ):
        tool_outputs = []
        try:
            # Process each tool call, invoke the appropriate tool, and append the result to the conversation
            # a message with tool calls is expected to be followed by tool responses
            for tool_call in ai_message.tool_calls:
                if tool_call["name"] == "SmartphoneInfo":
                    tool_message = smartphone_info_tool.invoke(
                        input=tool_call,
                        config={
                            "run_name": "smartphone-info-tool",
                            "callbacks": [langfuse_handler],
                            "metadata": {
                                "langfuse_session_id": session_id,
                                "langfuse_user_id": user_id
                            }
                        }
                    )
                    tool_outputs.append(tool_message.content)
        except Exception as e:
            print(f"An error occurred while processing tool calls: {e}")

        # Writing out tool_outputs to an AIMessage, as RunnableWithMessageHistory is not preserving the 'tool_calls'
        # attribute, and according to the OpenAI API spec, messages with role 'tool' must be preceded by a message
        # with 'tool_calls'. Ideally, we would want to return List[ToolMessage] here.
        combined_content = "\n".join(tool_outputs)
        return AIMessage(content=combined_content)


# ---------------------------
# Main Conversation Loop
# ---------------------------
@observe(name="main")
def main():
    with propagate_attributes(
        trace_name="ai-response",
        session_id=session_id,
        user_id=user_id
    ):
        # List of available tools
        tools = [smartphone_info_tool]

        # Bind the tools to the language model instance
        llm_with_tools = llm.bind_tools(tools)

        context_prompt = langfuse_client.get_prompt("context_prompt")
        context_prompt_template = ChatPromptTemplate.from_messages(context_prompt.get_langchain_prompt())
        context_prompt_template.metadata = {"langfuse_prompt": context_prompt}

        review_prompt = langfuse_client.get_prompt("review_prompt")
        review_prompt_template = ChatPromptTemplate.from_messages(review_prompt.get_langchain_prompt())
        review_prompt_template.metadata = {"langfuse_prompt": review_prompt}

        goodbye_prompt = langfuse_client.get_prompt("goodbye_prompt")
        goodbye_prompt_template = ChatPromptTemplate.from_messages(goodbye_prompt.get_langchain_prompt())
        goodbye_prompt_template.metadata = {"langfuse_prompt": goodbye_prompt}

        context_chain = context_prompt_template | trimmer | llm_with_tools | generate_context
        context_chain_with_message_history = RunnableWithMessageHistory(
            runnable=context_chain,
            get_session_history=get_redis_history,
            input_messages_key="user_input",
            history_messages_key="conversation"
        )
        context_chain_with_rails = my_rails | context_chain_with_message_history

        review_chain = review_prompt_template | llm
        review_chain_with_message_history = RunnableWithMessageHistory(
            runnable=review_chain,
            get_session_history=get_redis_history,
            input_messages_key="user_input",
            history_messages_key="conversation"
        )

        goodbye_chain = goodbye_prompt_template | llm

        try:
            print("Welcome to the Smartphone Assistant! I can help you with smartphone features and comparisons.")
            while True:
                user_input = input("User: ").strip()
                if user_input.lower() in ["exit", "quit", "bye", "end"]:
                    goodbye_message = goodbye_chain.invoke(
                        input={"user_id": user_id},
                        config={
                            "run_name": "goodbye-message",
                            "callbacks": [langfuse_handler],
                            "metadata": {
                                "langfuse_session_id": session_id,
                                "langfuse_user_id": user_id
                            }
                        }
                    )
                    feedback = input("Was this answer helpful? (Yes/No): ")
                    user_comment = input("Please give us a reason for your answer. This will help us improve: ")
                    langfuse_client.score_current_trace(
                        name="usefulness",
                        value=feedback,
                        data_type="CATEGORICAL",
                        comment=user_comment
                    )
                    print(f"System: {goodbye_message.content}")
                    break

                context_chain_response = context_chain_with_rails.invoke(
                    input={
                        "user_input": user_input
                    },
                    config={
                        "run_name": "context",
                        "configurable": {
                            "session_id": session_id
                        },
                        "callbacks": [langfuse_handler],
                        "metadata": {
                            "langfuse_session_id": session_id,
                            "langfuse_user_id": user_id
                        }
                    }
                )

                if (isinstance(context_chain_response, dict) and
                        context_chain_response.get("output") == NEGATIVE_RESPONSE):
                    print(f"System: {NEGATIVE_RESPONSE}")
                    continue

                review_chain_response = review_chain_with_message_history.invoke(
                    input={
                        "user_id": user_id,
                        "user_input": user_input
                    },
                    config={
                        "run_name": "final-response",
                        "configurable": {
                            "session_id": session_id
                        },
                        "callbacks": [langfuse_handler],
                        "metadata": {
                            "langfuse_session_id": session_id,
                            "langfuse_user_id": user_id
                        }
                    }
                )

                print(f"System: {review_chain_response.content}")
        except AuthenticationError as ae:
            print(f"An authentication error occurred: {ae}")
        except BadRequestError as bre:
            print(f"A client-side error occurred: {bre}")
        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
