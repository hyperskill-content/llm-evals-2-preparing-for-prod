# Start from the last coding stage of the previous LLM evals project
import json
import os
import re
import sys
import uuid
from difflib import SequenceMatcher

import dotenv
from langchain_community.docstore.document import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisChatMessageHistory
from openai import RateLimitError
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langfuse import observe, propagate_attributes, get_client
from langfuse.langchain import CallbackHandler
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from nemoguardrails.rails.llm.options import GenerationOptions

# Load environment variables from .env file
dotenv.load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6380/0")

# Generate unique session_id and user_id once
session_id = f"session-{uuid.uuid4().hex[:8]}"
users = ["James", "George", "Mike", "Sherlock"]
user_id = users[uuid.uuid4().int % len(users)]

llm = ChatOpenAI(
    model=os.getenv("LITELLM_CHAT_MODEL"),
    base_url=os.getenv("LITELLM_BASE_URL"),
    api_key=os.getenv("LITELLM_API_KEY"),
    model_kwargs={"user": user_id},
    tiktoken_model_name="gpt-4o-mini",
)

embeddings_model = OpenAIEmbeddings(
    model=os.getenv("LITELLM_EMBEDDINGS_MODEL"),
    base_url=os.getenv("LITELLM_BASE_URL"),
    api_key=os.getenv("LITELLM_API_KEY"),
    show_progress_bar=True
)

# loaded fresh into a local list at the start of each turn.

# Initialize Langfuse client
langfuse = get_client()


# ---------------------------
# Load JSON Data and Build Qdrant Vector Store
# ---------------------------

def load_smartphone_data(json_path: str) -> list:
    """
    Load raw smartphone JSON data from disk. Used both for building the vector
    store (embed_documents) and for fuzzy model-name matching (known_models).

    :param
        json_path (str): Path to the JSON file containing smartphone data.

    :returns
        A list of smartphone dicts, or an empty list if an error occurs.
    """
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {json_path} was not found.")
        return []
    except json.JSONDecodeError as jde:
        print(f"Error decoding JSON from file {json_path}: {jde}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while reading {json_path}: {e}")
        return []


@observe(name="embed_documents")
def embed_documents(data: list) -> QdrantVectorStore | list:
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

        else:
            qdrant_store = QdrantVectorStore.from_existing_collection(
                embedding=embeddings_model,
                collection_name=collection_name,
            )

            return qdrant_store

    except Exception as e:
        print(f"Error initializing the vector store: {e}")
        return []


# Load the raw JSON data once, then reuse it both for the vector store and for
# fuzzy model-name matching in the tool below.
smartphone_data = load_smartphone_data("datasets/smartphones.json")
product_db = embed_documents(smartphone_data)
known_models = [entry["model"] for entry in smartphone_data]


# ---------------------------
# Tool Definitions
# ---------------------------

def token_sort_ratio(a: str, b: str) -> float:
    a_sorted = " ".join(sorted(a.lower().split()))
    b_sorted = " ".join(sorted(b.lower().split()))
    return SequenceMatcher(None, a_sorted, b_sorted).ratio() * 100


def extract_numbers(s: str) -> set:
    return set(re.findall(r"\d+", s))


@tool("SmartphoneInfo")
def smartphone_info_tool(model: str) -> str:
    """
    Look up a smartphone model in the catalog. Uses substring matching first,
    then falls back to fuzzy matching with a number-mismatch guard to avoid
    hallucinating specs for models that are not actually in the catalog.
    """
    try:
        if not known_models:
            return "Smartphone catalog is currently unavailable."

        # 1. Exact substring match first (handles e.g. "Galaxy S23 Ultra" ->
        #    "Samsung Galaxy S23 Ultra 5G")
        lower_query = model.lower()
        substring_matches = [m for m in known_models if lower_query in m.lower()]
        if substring_matches:
            best = min(substring_matches, key=len)  # shortest containing match = closest
        else:
            # 2. Fuzzy fallback with number-mismatch guard
            scored = sorted(known_models, key=lambda c: -token_sort_ratio(model, c))
            top_match = scored[0]
            q_nums, m_nums = extract_numbers(model), extract_numbers(top_match)
            score = token_sort_ratio(model, top_match)
            if (q_nums and not (q_nums & m_nums)) or score < 60:
                return f"No information found for \'{model}\' in our catalog."
            best = top_match

        results = product_db.similarity_search(best, k=1)
        return results[0].page_content
    except Exception as e:
        return f"Error during smartphone information retrieval for model {model}: {e}"


# ---------------------------
# Tool Call Handling and Response Generation
# ---------------------------
@observe(name="generate_context")
def generate_context(ai_message: AIMessage, conversation: list) -> None:
    """
    Process tool calls from the language model and append the AI message and
    each tool's response as ToolMessage objects to the passed-in conversation list.

    :param
        ai_message (AIMessage): The language model's output message containing tool_calls.
        conversation (list): The current turn's in-memory conversation history
                              (loaded from Redis + current user input).
    """
    conversation.append(ai_message)

    if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
        conversation.append(
            AIMessage(
                content="No tool calls found. Please ensure the model is configured to use tools."
            )
        )

    try:
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
# Main Conversation Loop
# ---------------------------
def main():
    tools = [smartphone_info_tool]

    llm_with_tools = llm.bind_tools(tools)

    # ---------------------------
    # Fetch prompts from Langfuse (Task 1: Prompt Management and Versioning)
    # ---------------------------
    context_lf_prompt = langfuse.get_prompt("context_system_prompt")
    context_prompt = ChatPromptTemplate.from_messages([
        context_lf_prompt.get_langchain_prompt()[0],
        MessagesPlaceholder(variable_name="conversation"),
    ])
    context_prompt.metadata = {"langfuse_prompt": context_lf_prompt}

    review_lf_prompt = langfuse.get_prompt("review_system_prompt")
    review_prompt = ChatPromptTemplate.from_messages([
        review_lf_prompt.get_langchain_prompt()[0],
        MessagesPlaceholder(variable_name="conversation"),
    ])
    review_prompt.metadata = {"langfuse_prompt": review_lf_prompt}

    goodbye_lf_prompt = langfuse.get_prompt("goodbye_system_prompt")
    goodbye_prompt = PromptTemplate.from_template(
        goodbye_lf_prompt.get_langchain_prompt()
    )
    goodbye_prompt.metadata = {"langfuse_prompt": goodbye_lf_prompt}

    # ---------------------------
    # Task 2: Message trimming to bound token usage
    # ---------------------------
    trimmer = trim_messages(
        strategy="last",
        token_counter=llm,
        max_tokens=500,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
    )

    context_chain = context_prompt | trimmer | llm_with_tools
    review_chain = review_prompt | llm

    goodbye_chain = goodbye_prompt | llm

    # Initialize the Langfuse handler once for the entire conversation
    langfuse_handler = CallbackHandler()

    # ---------------------------
    # Task 2: Redis-backed chat history (replaces global `conversation = []`)
    # ---------------------------
    redis_history = RedisChatMessageHistory(
        session_id=session_id,
        redis_url=REDIS_URL,
        ttl=3600  # expire after 1 hour of inactivity
    )

    # ---------------------------
    # Task 3: Guardrails input validation via NeMo Guardrails
    # ---------------------------
    with open("config/config.yml", "r") as f:
        config_yaml_content = f.read().replace(
            "OPENAI_BASE_URL_PLACEHOLDER", os.getenv("LITELLM_BASE_URL")
        )

    with open("config/prompts.yml", "r") as f:
        prompts_yaml_content = f.read()

    guardrails_config = RailsConfig.from_content(
        colang_content="",
        yaml_content=config_yaml_content + "\n" + prompts_yaml_content,
    )

    input_rails = RunnableRails(guardrails_config, input_key="user_input")

    try:
        print("Welcome to the Smartphone Assistant! I can help you with smartphone features and comparisons.")
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() in ["exit", "quit", "bye", "end"]:
                # Create a parent span for the goodbye message
                try:
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

                except RateLimitError as budget_error:
                    # ---------------------------
                    # Task 4: Graceful handling of LiteLLM budget-exceeded errors
                    # ---------------------------
                    print(
                        f"System: Sorry {user_id}, you have reached your usage limit for today "
                        "and I can't process any more requests right now. Please try again later."
                    )
                    print(f"[debug] Budget error detail: {budget_error}")

                # Collect user feedback about the entire conversation
                feedback = input("\nWas this conversation helpful? (Yes/No): ").strip()
                user_comment = input("Please give us a reason for your answer. This will help us improve: ").strip()

                langfuse.create_score(
                    session_id=session_id,  # Use the session_id from the start of the conversation
                    name="conversation_usefulness",
                    value=feedback,
                    data_type="CATEGORICAL",
                    comment=user_comment
                )
                langfuse.flush()

                print("\nThank you for your feedback!")
                break

            # Load history fresh from Redis each turn, then append current input in-memory
            conversation = list(redis_history.messages)
            user_message = HumanMessage(user_input)
            conversation.append(user_message)

            # ---------------------------
            # Task 3: Validate input BEFORE invoking any chain or making an LLM call
            # ---------------------------
            try:
                validation_result = input_rails.rails.generate(
                    messages=[{"role": "user", "content": user_input}],
                    options=GenerationOptions(
                        rails=["input"],
                        output_vars=["allowed", "triggered_input_rail", "bot_message"],
                    ),
                )
            except RateLimitError as budget_error:
                # ---------------------------
                # Task 4: Graceful handling of LiteLLM budget-exceeded errors
                # raised during the guardrails self-check-input LLM call.
                # ---------------------------
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="user-query-budget-exceeded",
                    input={"user_input": user_input},
                ) as span:
                    with propagate_attributes(
                        session_id=session_id,
                        user_id=user_id
                    ):
                        span.update(
                            output={"response": str(budget_error)},
                            level="WARNING",
                        )

                print(
                    f"System: Sorry {user_id}, you have reached your usage limit for today "
                    "and I can't process any more requests right now. Please try again later."
                )
                print(f"[debug] Budget error detail: {budget_error}")
                continue

            validation_context = validation_result.output_data or {}
            rail_triggered = validation_context.get("allowed") is False or bool(
                validation_context.get("triggered_input_rail")
            )

            if rail_triggered:
                blocked_message = validation_result.response[0]["content"]

                # Trace the blocked attempt in Langfuse for later guardrail tuning
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="user-query-blocked",
                    input={"user_input": user_input},
                ) as span:
                    with propagate_attributes(
                        session_id=session_id,
                        user_id=user_id
                    ):
                        span.update(output={"response": blocked_message}, level="WARNING")

                print(f"System: {blocked_message}")
                continue  # Skip chains and Redis persistence entirely

            # Create a parent span for this user query to group all chain invocations
            try:
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
                        # Context chain invocation (tool calls happen in-memory only)
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

                    span.update(output={"response": response.content})

            except RateLimitError as budget_error:
                print(
                    f"System: Sorry {user_id}, you have reached your usage limit for today "
                    "and I can't process any more requests right now. Please try again later."
                )
                print(f"[debug] Budget error detail: {budget_error}")
                continue  # Skip printing/persisting a response, go back for next input

            print(f"System: {response.content}")

            # Persist only the user's message and the final AI response to Redis
            # (tool calls and intermediate messages stay in-memory for this turn only,
            # and blocked turns are never saved)
            redis_history.add_message(user_message)
            redis_history.add_message(response)

    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()