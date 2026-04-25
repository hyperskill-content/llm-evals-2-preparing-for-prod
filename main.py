import json
import os
import sys
import uuid
import dotenv
from langchain_community.docstore.document import Document
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe

dotenv.load_dotenv()

# redis runs on 6380 because 6379 is taken by langfuse
REDIS_URL = os.getenv("REDIS_CONNECTION_STRING", "redis://localhost:6380/0")
session_id = str(uuid.uuid4())

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

# route through litellm proxy instead of calling openai directly
# model_kwargs passes the user so litellm can track budgets per user
llm = ChatOpenAI(
    model=os.getenv("LITELLM_MODEL"),
    base_url=os.getenv("LITELLM_BASE_URL"),
    api_key=os.getenv("LITELLM_API_KEY"),
    model_kwargs={"user": "HyperUser"},
)

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    base_url=os.getenv("LITELLM_BASE_URL"),
    api_key=os.getenv("LITELLM_API_KEY"),
    show_progress_bar=True,
)

session_ending = False
blocked_message = "I'm sorry, I can't respond to that."


# returns a fresh redis chat history for this session with a 1 hour ttl
def get_chat_history():
    return RedisChatMessageHistory(session_id, url=REDIS_URL, ttl=3600)


# pulls the prompts from langfuse instead of hardcoding them
def load_prompts():
    assistant_langfuse_prompt = langfuse.get_prompt("smartphone-assistant-prompt")
    final_langfuse_prompt = langfuse.get_prompt("smartphone-final-response-prompt")

    # build the assistant prompt template with a placeholder for chat history
    assistant_prompt = ChatPromptTemplate.from_messages([
        assistant_langfuse_prompt.get_langchain_prompt()[0],  # system message
        MessagesPlaceholder(variable_name="chat_history"),
        assistant_langfuse_prompt.get_langchain_prompt()[1],  # user message
    ])
    # link the prompt to langfuse so we can see metrics per prompt
    assistant_prompt.metadata = {"langfuse_prompt": assistant_langfuse_prompt}

    final_prompt = ChatPromptTemplate.from_messages([
        final_langfuse_prompt.get_langchain_prompt()[0],
        MessagesPlaceholder(variable_name="chat_history"),
        final_langfuse_prompt.get_langchain_prompt()[1],
    ])
    final_prompt.metadata = {"langfuse_prompt": final_langfuse_prompt}

    return assistant_prompt, final_prompt


# load smartphones from json and store them in qdrant
def embed_documents(json_path):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
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
        if not qdrant_client.collection_exists(collection_name=collection_name):
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            qdrant_store = QdrantVectorStore(
                client=qdrant_client, collection_name=collection_name, embedding=embeddings_model
            )
            qdrant_store.add_documents(documents=documents)
            return qdrant_store
        else:
            return QdrantVectorStore.from_existing_collection(
                embedding=embeddings_model, collection_name=collection_name
            )
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return []


@observe(name="SmartphoneInfo")
def _get_smartphone_info(model):
    results = product_db.similarity_search(model, k=1)
    if not results:
        return "Could not find information for the specified model."
    info = results[0].page_content
    langfuse_context.update_current_observation(
        input=model, output=info, metadata={"collection": "smartphones", "k": 1}
    )
    return info


@tool("SmartphoneInfo")
def smartphone_info_tool(model: str) -> str:
    """Retrieves information about a smartphone model from the product database."""
    try:
        return _get_smartphone_info(model)
    except Exception as e:
        return f"Error retrieving smartphone info: {e}"


@tool("EndSession")
def end_session_tool(session_status: str):
    """Ends the current session when the user is done."""
    global session_ending
    session_ending = True
    try:
        goodbye = llm.invoke(
            "You are an AI assistant for smartphones. Write a short polite goodbye message."
        ).content
        print(goodbye)
    except Exception:
        print("Thank you for visiting. Goodbye!")
    print("\nRate the model's responses from 1 to 5 (3 being average):")
    return "Session ended."


@observe(name="generate_context")
def generate_context(llm_response):
    context = []
    for tool_call in llm_response.tool_calls:
        if tool_call["name"] == "SmartphoneInfo":
            result = smartphone_info_tool.invoke(tool_call)
            context.append(result)
        elif tool_call["name"] == "EndSession":
            end_session_tool.invoke(tool_call)
        else:
            context.append("No tool found for this query.")
            sys.exit(0)
    langfuse_context.update_current_observation(
        input=str(llm_response.tool_calls),
        output=str(context),
        metadata={"tool_calls": len(llm_response.tool_calls)},
    )
    return context


@observe(name="context")
def get_context(context_chain, chat_history, user_input):
    result = context_chain.invoke({"chat_history": chat_history, "user_input": user_input})
    langfuse_context.update_current_observation(
        input=user_input, output=str(result),
        metadata={"chat_history_length": len(chat_history)}
    )
    return result


@observe(name="final_response", as_type="generation")
def get_final_response(final_prompt, chat_history, user_input, context):
    response = llm.invoke(
        final_prompt.invoke({
            "chat_history": chat_history,
            "user_input": user_input,
            "context": str(context),
        })
    )
    langfuse_context.update_current_observation(
        input=user_input, output=response.content,
        metadata={"model": os.getenv("LITELLM_MODEL")}
    )
    return response


@observe(name="main")
def main():
    global session_ending
    session_ending = False

    assistant_prompt, final_prompt = load_prompts()

    tools = [smartphone_info_tool, end_session_tool]
    llm_with_tools = llm.bind_tools(tools)

    # trim old messages so we don't send too many tokens each time
    trimmer = trim_messages(
        strategy="last",
        token_counter=len,
        max_tokens=10,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
    )

    # set up guardrails to check input before running the chain
    rails_config = RailsConfig.from_path("./config")
    input_checker = RunnableRails(
        rails_config,
        runnable=RunnableLambda(lambda x: {"output": "allowed"}),
        input_key="user_input",
    )

    context_chain = assistant_prompt | llm_with_tools | generate_context

    # set up redis chat history and clear it for a fresh session
    chat_history = get_chat_history()
    chat_history.clear()

    print("Welcome to the Smartphone Assistant! I can help you with smartphone features and comparisons.")

    while True:
        user_input = input("User: ").strip()

        # trim history to avoid token limit issues
        history = chat_history.messages
        trimmed_history = trimmer.invoke(history) if history else []

        # check with guardrails first - if blocked skip the llm call entirely
        guard_result = input_checker.invoke({"user_input": user_input})
        if isinstance(guard_result, dict) and guard_result.get("output") == blocked_message:
            print(f"System: {blocked_message}")
            continue

        context = get_context(context_chain, trimmed_history, user_input)

        if session_ending:
            trace_id = langfuse_context.get_current_trace_id()
            rating_str = input("User: ").strip()
            try:
                rating = float(rating_str)
            except ValueError:
                rating = 3.0
            print("Please give us a reason for your answer:")
            comment = input("User: ").strip()
            langfuse.score(
                trace_id=trace_id, name="usefulness", value=rating,
                comment=comment, data_type="NUMERIC",
            )
            langfuse.flush()
            sys.exit(0)

        response = get_final_response(final_prompt, trimmed_history, user_input, context)

        # save the messages to redis
        chat_history.add_message(HumanMessage(content=user_input))
        chat_history.add_message(AIMessage(content=response.content))

        print(f"System: {response.content}")


if __name__ == "__main__":
    product_db = embed_documents("datasets/smartphones.json")
    main()
