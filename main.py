# start where you left off in part one of this series
# I will use my code from LLM-eval2 Step 2 (redis integration is missing here)
import json
import os
import sys
import uuid

import dotenv
from langchain_community.docstore.document import Document
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

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

langfuse_handler = CallbackHandler()

# --------------------------------------------------------------------
# Guardrails AI Configuration
# --------------------------------------------------------------------
config = RailsConfig.from_path("config")
llm_rails = LLMRails(config)
# runnable_rails = RunnableRails(config)


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

# Initialize conversation history
conversation = []


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
def generate_context(ai_message: AIMessage) -> dict:
    """
    Process tool calls from the language model and collect their responses as ToolMessage objects.

    :param
        ai_message (AIMessage): The language model's output message containing tool_calls.

    :returns
        A dictionary containing a list of ToolMessage objects under the key "tool_responses".
    """
    # construct the conversation history with the AI message containing tool calls
    conversation.append(ai_message)

    # Check if the AI message has any tool calls
    if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
        conversation.append(
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
                conversation.append(
                    ToolMessage(
                        content=tool_output,
                        tool_call_id=tool_call["id"]
                    )
                )

    except Exception as e:
        print(f"An error occurred while processing tool calls: {e}")
        conversation.append(
            AIMessage(
                content=f"An error occurred while processing tool calls: {e}"
            )
        )
    return {"tool_responses": conversation}


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
    goodbye_prompt = ChatPromptTemplate.from_messages(lg_goodbye_prompt.get_langchain_prompt())

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

            # 1. Guardrails checks only input
            guardrails_result = llm_rails.generate(
                messages=[{"role": "user", "content": user_input}]
            )
            # If Guardrails blocks, answer directly
            if guardrails_result and guardrails_result.get("content"):
                blocked_answer = guardrails_result["content"]

                # If Guardrails returns refusal, output directly
                if "can't help" in blocked_answer.lower() or "cannot help" in blocked_answer.lower():
                    print(f"System: {blocked_answer}")
                    continue

            # if user input is not (exit, bye, etc)
            conversation.append(HumanMessage(user_input))

            context_chain.invoke(
                {"user_input": user_input, "conversation": conversation},
                config={
                    "run_name": "context",
                    "callbacks": [langfuse_handler],
                    "metadata": {
                        "langfuse_session_id": session_name,
                        "langfuse_user_id": user_id,
                        "langfuse_tags": [basic_tag + "-context"],
                        "langfuse_prompt": lf_context_prompt,
                    },
                },
            )

            response = review_chain.invoke(
                {"user_id": user_id, "user_input": user_input, "conversation": conversation},
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
            lf = get_client()
            lf.update_current_trace(
                output=response.content
            )

            print(f"System: {response.content}")
            conversation.append(response)

    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()