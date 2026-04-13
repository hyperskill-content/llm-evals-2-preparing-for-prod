import json
import os
import sys
import uuid

import dotenv
from langchain_community.docstore.document import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langfuse import observe, get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Load environment variables from .env file
dotenv.load_dotenv()

langfuse_client = get_client()
langfuse_handler = CallbackHandler()

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
    show_progress_bar=True
)

# Initialize conversation history
conversation = []


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
def generate_context(ai_message: AIMessage) -> dict:
    """
    Process tool calls from the language model and collect their responses as ToolMessage objects.

    :param
        ai_message (AIMessage): The language model's output message containing tool_calls.

    :returns
        A dictionary containing a list of ToolMessage objects under the key "tool_responses".
    """
    with propagate_attributes(
        trace_name="ai-response",
        session_id=session_id,
        user_id=user_id
    ):
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
                    tool_output = smartphone_info_tool.invoke(
                        input=tool_call,
                        config={
                            "run-name": "smartphone-info-tool",
                            "callbacks": [langfuse_handler],
                            "metadata": {
                                "langfuse_session_id": session_id,
                                "langfuse_user_id": user_id
                            }
                        }
                    )
                    conversation.append(tool_output)

        except Exception as e:
            print(f"An error occurred while processing tool calls: {e}")
            conversation.append(
                AIMessage(
                    content=f"An error occurred while processing tool calls: {e}"
                )
            )


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

        context_system_prompt = """
            You're part of a smartphone recommendation system. Your work is to use the SmartphoneInfo tool to retrieve information about smartphones based on user queries.
              - If the user requests specs/comparisons/recommendations for a model explicitly mentioned in chat or can be inferred from the conversation history, call SmartphoneInfo(model)
              - If multiple models are mentioned or from the conversation history, you must call SmartphoneInfo for each model separately.
              - If the user asks a general question, do nothing. 
            Do not guess or recommend a model from internal knowledge; the model name must be clear from the chat history or user input. 
            
            Current question: {user_input}
        """

        review_system_prompt = """
            You are an expert AI assistant helping customers pick the best smartphone from our catalog. Follow these rules strictly:
            
            1. Focus solely on concise (under 100 words), human-like, personalized reviews/comparisons of models named in the user’s query or provided context.
            2. Think step by step before answering.
            3. Never guess or recommend any model not explicitly mentioned in the context or query.
            4. If no model is given, ask the user to check our online catalog for the exact model name.
            5. DO NOT assist with ordering, returns, tracking, or other general support.
            6. If asked about anything outside smartphone features/comparisons, respond that you can’t help.
            7. If the user only wants to chat, engage briefly, but always steer back to smartphone comparisons.
            8. Never list smartphone specifications, but instead explain how they translate to real-world benefits.
            
            When recommending, evaluate performance, display, battery, camera, and special functions (e.g., 5G, fast charging, expandable storage), and how they translate to real-world benefits. 
            Always confirm the user’s needs before finalizing. 
            
            Current user: {user_id}
            Current question: {user_input}
        """

        goodbye_system_prompt = """
            You have been helping the user: {user_id} with smartphone features and comparisons. 
            Generate a short friendly goodbye message for the user and also thank them for their feedback.
            (note that you've already been helping the user with smartphone features and comparisons, so do not repeat that or greet them again)
        """

        context_prompt = ChatPromptTemplate.from_messages(
            [
                (SystemMessage(context_system_prompt)),
                MessagesPlaceholder(variable_name="conversation")
            ]
        )

        review_prompt = ChatPromptTemplate.from_messages(
            [
                (SystemMessage(review_system_prompt)),
                MessagesPlaceholder(variable_name="conversation")
            ]
        )

        goodbye_prompt = PromptTemplate.from_template(
            goodbye_system_prompt
        )

        context_chain = context_prompt | llm_with_tools | generate_context
        review_chain = review_prompt | llm

        goodbye_chain = goodbye_prompt | llm

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

                conversation.append(HumanMessage(user_input))

                context_chain.invoke(
                    input={
                        "user_input": user_input,
                        "conversation": conversation
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
                    input={
                        "user_id": user_id,
                        "user_input": user_input,
                        "conversation": conversation
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

                print(f"System: {response.content}")
                conversation.append(response)

        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
