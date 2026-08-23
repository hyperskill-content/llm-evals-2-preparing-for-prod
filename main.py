# Start from the last coding stage of the previous LLM evals project
import json
import os
import sys
import uuid

import dotenv
from langchain_community.docstore.document import Document
from langchain_core.globals import set_debug
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisChatMessageHistory
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langfuse import observe, propagate_attributes, get_client
from langfuse.langchain import CallbackHandler
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from nemoguardrails.rails.llm.options import GenerationOptions

# Load environment variables from .env file
dotenv.load_dotenv()

# Load guardrails configuration
config = RailsConfig.from_path("config/")

# Create guardrails instance for input validation only
input_rails = RunnableRails(config, input_key="user_input")
output_rails = RunnableRails(config, input_key="response")

# Generate unique session_id and user_id once
session_id = f"session-{uuid.uuid4().hex[:8]}"
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
    model=os.getenv("OPENAI_EMBEDDINGS_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    show_progress_bar=True
)

# Initialize Redis for conversation history
REDIS_URL=os.getenv("REDIS_CONNECTION_STRING")
redis_history = RedisChatMessageHistory(
    session_id=session_id,
    redis_url=REDIS_URL,
    ttl=3600 # 1 hour
)

# Initialize Langfuse client
langfuse = get_client()


# ---------------------------
# Load JSON Data and Build Qdrant Vector Store
# ---------------------------

@observe(name="embed_documents")
def embed_documents(json_path: str) -> QdrantVectorStore | list:
    """
    Load JSON data from the smartphones.json file and convert each entry to a Document.
    :param
        json_path (str): Path to the JSON file containing smartphone data.

    :returns
        A Qdrant vector store built from the smartphone documents,
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
@observe(name="generate_context")
def generate_context(ai_message: AIMessage, conversation: list) -> None:
    """
    Process tool calls from the language model and append the AI message and
    each tool's response as ToolMessage objects to the conversation history.

    :param
        ai_message (AIMessage): The language model's output message containing tool_calls.
        conversation (list): In memory conversation history of the current turn
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
                conversation.append(tool_output)

    except Exception as e:
        print(f"An error occurred while processing tool calls: {e}")
        conversation.append(
            AIMessage(
                content=f"An error occurred while processing tool calls: {e}"
            )
        )


# ---------------------------
# Message and Context Handling
# ---------------------------
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
def main():
    # List of available tools
    tools = [smartphone_info_tool]

    # Bind the tools to the language model instance
    llm_with_tools = llm.bind_tools(tools)

    # Langfuse prompts
    context_lf_prompt = langfuse.get_prompt("context_system_prompt")
    review_lf_prompt = langfuse.get_prompt("review_system_prompt")
    goodbye_lf_prompt = langfuse.get_prompt("goodbye_system_prompt")

    context_prompt = ChatPromptTemplate.from_messages(
        [
            context_lf_prompt.get_langchain_prompt()[0],
            MessagesPlaceholder(variable_name="conversation")
        ]
    )
    context_prompt.metadata = {"langfuse_prompt": context_lf_prompt}

    review_prompt = ChatPromptTemplate.from_messages(
        [
            review_lf_prompt.get_langchain_prompt()[0],
            MessagesPlaceholder(variable_name="conversation")
        ]
    )
    review_prompt.metadata = {"langfuse_prompt": review_lf_prompt}

    goodbye_prompt = PromptTemplate.from_template(
        goodbye_lf_prompt.get_langchain_prompt()[0][1]
    )
    goodbye_prompt.metadata = {"langfuse_prompt": goodbye_lf_prompt}

    context_chain = context_prompt | trimmer | llm_with_tools
    review_chain = review_prompt | llm

    goodbye_chain = goodbye_prompt | llm

    # Initialize the Langfuse handler once for the entire conversation
    langfuse_handler = CallbackHandler()

    try:
        print("Welcome to the Smartphone Assistant! I can help you with smartphone features and comparisons.")
        while True:
            # Load messages from Redis
            conversation = list(redis_history.messages)

            user_input = input("User: ").strip()
            if user_input.lower() in ["exit", "quit", "bye", "end"]:
                # Create a parent span for the goodbye message
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="user-query",
                    input={"user_input": user_input}
                ) as span:
                    with propagate_attributes(
                        session_id=session_id,
                        user_id=user_id
                    ):
                        goodbye_message = goodbye_chain.invoke(
                            {"user_id": user_id},
                            config={
                                "run_name": "goodbye-message",
                                "callbacks": [langfuse_handler]
                            }
                        )

                        # Set the output on the parent span
                        span.update(output={"response": goodbye_message.content})

                print(f"System: {goodbye_message.content}")

                # Collect user feedback about the entire conversation
                feedback = input("\nWas this conversation helpful? (Yes/No): ").strip()
                user_comment = input("Please give us a reason for your answer. This will help us improve: ").strip()

                # Score at the session level (not individual trace)
                langfuse.create_score(
                    session_id=session_id,  # Use the session_id from the start of the conversation
                    name="conversation_usefulness",
                    value=feedback,
                    data_type="CATEGORICAL",
                    comment=user_comment
                )

                print("\nThank you for your feedback!")
                break

            # Create user message
            user_message = HumanMessage(user_input)
            conversation.append(user_message)

            # Create a parent span for this user query to group all chain invocations
            with langfuse.start_as_current_observation(
                as_type="span",
                name="user-query",
                input={"user_input": user_input}
            ) as span:
                # Propagate trace attributes to all child observations
                with propagate_attributes(
                    session_id=session_id,
                    user_id=user_id
                ):

                    validation_result = input_rails.rails.generate(
                        messages=[{"role": "user", "content": user_input}],
                        options=GenerationOptions(
                            rails=["input"],
                            output_vars=["allowed", "triggered_input_rail", "bot_message"],
                        ),
                    )

                    validation_context = validation_result.output_data or {}
                    input_allowed = validation_context.get("allowed")
                    triggered_input_rail = validation_context.get("triggered_input_rail")
                    rail_triggered = input_allowed is False or bool(triggered_input_rail)

                    span.update(
                        metadata={
                            "input_guardrail_allowed": input_allowed,
                            "input_guardrail_triggered_rail": triggered_input_rail
                        }
                    )

                    if rail_triggered:
                        # Rail triggered - skip further processing
                        span.update(
                            output={"response": validation_result.response[0]["content"]},
                            level="WARNING",
                            status_message="Input blocked by a guardrail"
                        )
                        span.score_trace(name="guardrail_blocked", value=1, data_type="BOOLEAN")
                        print(f"System: {validation_result.response[0]["content"]}")
                        continue  # Skip saving to Redis and proceed to next input

                    # Rail not triggered - continue with normal processing
                    ai_message = context_chain.invoke(
                        {"user_input": user_input, "conversation": conversation},
                        config={
                            "run_name": "context",
                            "callbacks": [langfuse_handler]
                        }
                    )
                    generate_context(ai_message, conversation)

                    # Final response chain invocation
                    response = review_chain.invoke(
                        {"user_id": user_id, "user_input": user_input, "conversation": conversation},
                        config={
                            "run_name": "final-response",
                            "callbacks": [langfuse_handler]
                        }
                    )

                output_validation_result = output_rails.rails.generate(
                    messages=[
                        {"role": "user", "content": ""},
                        {"role": "assistant", "content": response.content}
                    ],
                    options=GenerationOptions(
                        rails=["output"],
                        output_vars=["allowed", "triggered_output_rail", "bot_message"],
                    ),
                )

                output_validation_context = output_validation_result.output_data or {}
                output_allowed = output_validation_context.get("allowed")
                triggered_output_rail = output_validation_context.get("triggered_output_rail")
                output_rail_triggered = output_allowed is False or bool(triggered_output_rail)

                span.update(
                    metadata={
                        "output_guardrail_allowed": output_allowed,
                        "output_guardrail_triggered_rail": triggered_output_rail,
                    }
                )

                if output_rail_triggered:
                    # Rail triggered - skip further processing
                    span.update(
                        output={"response": output_validation_result.response[0]["content"]},
                        level="WARNING",
                        status_message="Output blocked by guardrail",
                    )
                    span.score_trace(name="guardrail_blocked", value=1, data_type="BOOLEAN")
                    print(f"System: {output_validation_result.response[0]["content"]}")
                    continue

                span.update(output={"response": response.content})
                span.score_trace(name="guardrail_blocked", value=0, data_type="BOOLEAN")

            print(f"System: {response.content}")

            # Save only user input and final response (only when rail not triggered)
            redis_history.add_message(user_message)
            redis_history.add_message(response)

    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
