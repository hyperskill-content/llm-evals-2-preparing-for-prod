import json
import os
import sys
import uuid
import dotenv
from langchain_community.docstore.document import Document
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisChatMessageHistory
from langfuse import observe, get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

dotenv.load_dotenv()

users = ["James", "George", "Mike", "Sherlock"]
user_id = users[uuid.uuid4().int % len(users)]

langfuse_client = get_client()

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    show_progress_bar=True
)


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

        else:
            qdrant_store = QdrantVectorStore.from_existing_collection(
                embedding=embeddings_model,
                collection_name=collection_name,
            )

            return qdrant_store

    except Exception as e:
        print(f"Error initializing the vector store: {e}")
        return []


product_db = embed_documents("datasets/smartphones.json")


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


@observe(name="generate-context")
def generate_context(ai_message: AIMessage, session_id: str, user_input: str):
    redis_history = get_redis_history(session_id)
    redis_history.add_user_message(user_input)
    redis_history.add_message(ai_message)

    if not ai_message.tool_calls:
        return ai_message

    for tool_call in ai_message.tool_calls:
        if tool_call["name"] == "SmartphoneInfo":
            tool_output = smartphone_info_tool.invoke(tool_call["args"])
            redis_history.add_message(AIMessage(content=str(tool_output)))

    return ai_message


langfuse_handler = CallbackHandler()

REDIS_URL = "redis://localhost:6380/0"
BLOCKED_MESSAGE = "I'm sorry, I can't respond to that."


def get_redis_history(session_id: str) -> BaseChatMessageHistory:
    return RedisChatMessageHistory(
        session_id,
        redis_url=REDIS_URL,
        ttl=3600
    )

def print_redis_history(session_id: str):
    history = get_redis_history(session_id)
    messages = history.messages
    print(f"\n--- Redis History ({len(messages)} messages) ---")
    for i, msg in enumerate(messages):
        print(f"[{i}] {type(msg).__name__}: {msg.content[:80] if msg.content else '(tool_calls only)'}")
    print("---\n")

def get_filtered_trimmed_history(session_id: str, max_messages: int = 20):
    messages = get_redis_history(session_id).messages
    if len(messages) > max_messages:
        print(f"Info: Trimming history from {len(messages)} to {max_messages} messages.")
        messages = messages[-max_messages:]

    filtered = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            continue
        if isinstance(msg, AIMessage) and msg.tool_calls and not msg.content:
            continue
        filtered.append(msg)
    return filtered

@observe(name="main")
@propagate_attributes()
def main():
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    langfuse_client = get_client()
    langfuse_client.update_current_span(
        name="ai-response",
        metadata={
            "session_id": session_id,
            "user_id": user_id
        }
    )
    context_lf_prompt = langfuse_client.get_prompt("context-prompt")
    review_lf_prompt = langfuse_client.get_prompt("review_system_prompt")
    goodbye_lf_prompt = langfuse_client.get_prompt("good_bye")

    context_prompt = ChatPromptTemplate.from_messages(
        [
            context_lf_prompt.get_langchain_prompt()[0],
            MessagesPlaceholder(variable_name="conversation"),
        ]
    )

    review_prompt = ChatPromptTemplate.from_messages(
        [
            review_lf_prompt.get_langchain_prompt()[0],
            MessagesPlaceholder(variable_name="conversation"),
        ]
    )

    goodbye_prompt = PromptTemplate.from_template(
        goodbye_lf_prompt.get_langchain_prompt()
    )

    tools = [smartphone_info_tool]

    llm_with_tools = llm.bind_tools(tools)

    def generate_context_with_session(ai_message: AIMessage):
        generate_context(ai_message, session_id, user_input)
        return {"output": ai_message.content or ""}

    rails_config = RailsConfig.from_path("config")
    rails = RunnableRails(rails_config, input_key="user_input")

    goodbye_chain = goodbye_prompt | llm
    context_chain = context_prompt | llm_with_tools | generate_context_with_session
    review_chain = review_prompt | llm

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
                            "langfuse_user_id": user_id
                        }
                    }
                )
                print(f"System: {goodbye_message.content}")

                feedback = input("Was this answer helpful? (Yes/No): ").strip()
                user_comment = input("Please give us a reason for your answer. This will help us improve: ").strip()

                langfuse_client = get_client()
                langfuse_client.score_current_trace(
                    name="usefulness",
                    value=feedback,
                    data_type="CATEGORICAL",
                    comment=user_comment
                )
                print("Thank you for your feedback!")
                break

            rails_check = rails.invoke({"input": user_input, "conversation": []})
            if rails_check.get("output", "") == BLOCKED_MESSAGE:
                print(f"System: {BLOCKED_MESSAGE}")
                continue

            context_chain.invoke(
                {
                    "user_input": user_input,
                    "conversation": get_filtered_trimmed_history(session_id)
                },
                config={
                    "run_name": "context",
                    "callbacks": [langfuse_handler],
                    "metadata": {
                        "langfuse_session_id": session_id,
                        "langfuse_user_id": user_id
                    }
                }
            )

            response = review_chain.invoke(
                {
                    "user_id": user_id,
                    "user_input": user_input,
                    "conversation": get_filtered_trimmed_history(session_id)
                },
                config={
                    "run_name": "final-response",
                    "callbacks": [langfuse_handler],
                    "metadata": {
                        "langfuse_session_id": session_id,
                        "langfuse_user_id": user_id
                    }
                }
            )
            get_redis_history(session_id).add_message(response)
            print_redis_history(session_id)
            print(f"System: {response.content}")
    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
