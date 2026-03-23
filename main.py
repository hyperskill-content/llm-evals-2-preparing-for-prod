import json
import os
import sys
import uuid
import dotenv
from langchain_community.docstore.document import Document
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_redis import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnableLambda

# Load environment variables from .env file
dotenv.load_dotenv()

session_id = f"session-{uuid.uuid4().hex[:8]}"
user_id = f"user-{uuid.uuid4().hex[:8]}"

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

langfuse_handler = CallbackHandler()
lf = get_client()
# Initialize conversation history
conversation = []
REDIS_URL = "redis://localhost:6380/0"
chat_history = RedisChatMessageHistory(session_id=session_id, redis_url=REDIS_URL)

def get_redis_history(session_id: str) -> BaseChatMessageHistory:
    return RedisChatMessageHistory(session_id = session_id, redis_url=REDIS_URL, ttl=120)

def get_clean_messages():
    return [
        msg for msg in chat_history.messages
        if not isinstance(msg, ToolMessage)
        and not (isinstance(msg, AIMessage) and msg.tool_calls)
    ]

def get_config(run_name: str, metadata: dict) -> RunnableConfig:
    # "goodbye" ["config", "goodbye"]
    config: RunnableConfig = {
        "run_name": run_name,
        "callbacks": [langfuse_handler],
        "configurable": {"session_id": session_id},
        "metadata": {
            "langfuse_session_id": session_id,
            "langfuse_user_id": user_id,
            "langfuse_tags": metadata
        },
    }
    return config


def update_trace(run_name: str):
    lf.update_current_trace(
        name=run_name,
        session_id=session_id,
        user_id=user_id,
    )


# ---------------------------
# Load JSON Data and Build Qdrant Vector Store
# ---------------------------
@observe(name="load-data-observer")
def embed_documents(json_path: str):
    """
    Load JSON data from the smartphones.json file and convert each entry to a Document.
    :param
        json_path (str): Path to the JSON file containing smartphone data.

    :returns
        Qdrant vector store A Qdrant vector store built from the smartphone documents,
                or an empty list if an error occurs.
    """
    update_trace("load-data-trace")
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

def get_trimmer():
    return trim_messages(
        strategy="last",  # keep either the last or first messages
        token_counter=llm,  # use your LLM to count tokens or create a special function
        max_tokens=500,  # the maximum number of tokens
        start_on="human",  # the first message type in the trimmed history
        end_on=("human", "tool"),  # the last message type in the trimmed history
        include_system=True,  # always include the system message
    )

# ---------------------------
# Tool Call Handling and Response Generation
# ---------------------------
@observe(name="generate_context-observer")
def generate_context(ai_message: AIMessage):
    """
    Process tool calls from the language model and collect their responses as ToolMessage objects.

    :param
        ai_message (AIMessage): The language model's output message containing tool_calls.

    :returns
        A dictionary containing a list of ToolMessage objects under the key "tool_responses".
    """
    # construct the conversation history with the AI message containing tool calls
    chat_history.add_message(ai_message)

    if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
        chat_history.add_message(
            AIMessage(
                content="No tool calls found. Please ensure the model is configured to use tools."
            )
        )
        return {"tool_responses": []}

    tool_responses = []
    available_tools = {"SmartphoneInfo": smartphone_info_tool}

    for tool_call in ai_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        selected_tool = available_tools.get(tool_name)
        if selected_tool:
            try:
                result = selected_tool.invoke(tool_args)
                tool_msg = ToolMessage(content=result, tool_call_id=tool_call["id"])
                chat_history.add_message(tool_msg)
                tool_responses.append(tool_msg)
            except Exception as e:
                error_msg = ToolMessage(
                    content=f"Error executing tool {tool_name}: {e}",
                    tool_call_id=tool_call["id"],
                )
                chat_history.add_message(error_msg)
                tool_responses.append(error_msg)
        else:
            not_found_msg = ToolMessage(
                content=f"Tool {tool_name} not found.",
                tool_call_id=tool_call["id"],
            )
            chat_history.add_message(not_found_msg)
            tool_responses.append(not_found_msg)

    return {"tool_responses": tool_responses}


# ---------------------------
# Main Conversation Loop
# ---------------------------
@observe(name="main-observer")
def main():
    # List of available tools
    tools = [smartphone_info_tool]

    # Bind the tools to the language model instance
    llm_with_tools = llm.bind_tools(tools)

    lf_prompt_context = lf.get_prompt(name="smartphone/context", label="latest")
    lf_prompt_review = lf.get_prompt(name="smartphone/review", label="latest")
    lf_prompt_goodbye = lf.get_prompt(name="smartphone/goodbye", label="latest")

    context_prompt = ChatPromptTemplate.from_messages(
        [
            lf_prompt_context.get_langchain_prompt()[0],
            MessagesPlaceholder(variable_name="conversation"),
            lf_prompt_context.get_langchain_prompt()[1],
        ]
    )
    review_prompt = ChatPromptTemplate.from_messages(
        [
            lf_prompt_review.get_langchain_prompt()[0],
            MessagesPlaceholder(variable_name="conversation"),
            lf_prompt_review.get_langchain_prompt()[1],
        ]
    )
    goodbye_prompt = ChatPromptTemplate.from_messages(
        lf_prompt_goodbye.get_langchain_prompt()
    )

    review_prompt.metadata = {"langfuse_prompt": lf_prompt_review}
    context_prompt.metadata = {"langfuse_prompt": lf_prompt_context}
    goodbye_prompt.metadata = {"langfuse_prompt": lf_prompt_goodbye}

    trimmer = get_trimmer()
    context_chain = context_prompt | trimmer | llm_with_tools | RunnableLambda(generate_context)
    review_chain = review_prompt | trimmer | llm
    goodbye_chain = goodbye_prompt | llm

    # chain_context_with_message_history = RunnableWithMessageHistory(
    #     runnable=context_chain, get_session_history=get_redis_history, input_messages_key="user_input", history_messages_key="conversation"
    # )
    #
    # chain_review_with_message_history = RunnableWithMessageHistory(
    #     runnable=review_chain, get_session_history=get_redis_history, input_messages_key="user_input", history_messages_key="conversation"
    # )

    update_trace("main-trace")
    try:
        print("Welcome to the Smartphone Assistant! I can help you with smartphone features and comparisons.")
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() in ["exit", "quit", "bye", "end"]:
                goodbye_message = goodbye_chain.invoke(
                    {"user_id": user_id},
                    get_config("goodbye", {"goodbye": "invoke"})
                )

                while True:
                    feedback = input("Was this answer helpful? (Yes/No): ").lower()
                    if feedback in ["yes", "y", "no", "n"]:
                        break  # Exit the loop for valid input
                    else:
                        print("Invalid input. Please try again.")
                user_comment = input("Please give us a reason for your answer. This will help us improve: ")

                lf.score_current_trace(
                    name="usefulness",
                    value=feedback,
                    data_type="CATEGORICAL",
                    comment=user_comment
                )
                chat_history.add_message(AIMessage(content=goodbye_message.content))
                print(f"System: {goodbye_message.content}")
                break


            chat_history.add_message(HumanMessage(content=user_input))
            trimmed_messages = trimmer.invoke(get_clean_messages())
            context_chain.invoke(
                {"user_id": user_id, "user_input": user_input, "conversation": trimmed_messages},
                get_config("context", {"context": "invoke"})
            )

            response = review_chain.invoke(
                {"user_id": user_id, "user_input": user_input, "conversation": get_clean_messages()},
                get_config("review", {"review": "invoke"})
            )

            chat_history.add_message(AIMessage(content=response.content))

            print(f"System: {response.content}")

    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
