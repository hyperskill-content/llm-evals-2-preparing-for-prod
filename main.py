import json
import os
import sys

import dotenv
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.docstore.document import Document
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage, trim_messages
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_redis import RedisChatMessageHistory
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langfuse.langchain import CallbackHandler
from langfuse import observe, get_client, propagate_attributes, Langfuse

dotenv.load_dotenv()

REDIS_URL = "redis://localhost:6380/0"

app = FastAPI()


class QueryRequest(BaseModel):
    user_input: str
    user_id: str
    session_id: str


llm = ChatOpenAI(
    model=os.getenv("LITELLM_MODEL"),
    base_url="http://localhost:4000/",
    api_key=os.getenv("LITELLM_API_KEY"),
    model_kwargs={"user": "HyperUser"},
)

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    base_url="http://localhost:4000/",
    api_key=os.getenv("LITELLM_API_KEY"),
    show_progress_bar=True,
)


def get_clean_messages(chat_history):
    return [
        msg for msg in chat_history.messages
        if not isinstance(msg, ToolMessage)
        and not (isinstance(msg, AIMessage) and msg.tool_calls)
    ]


@observe(name="embed-documents")
def embed_documents(json_path: str):
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


@tool("SmartphoneInfo")
def smartphone_info_tool(model: str) -> str:
    """Retrieves information about a smartphone model from the product database."""
    product_db = embed_documents("datasets/smartphones.json")
    try:
        results = product_db.similarity_search(model, k=1)
        if not results:
            return "Could not find information for the specified model."
        info = results[0].page_content
        return info
    except Exception as e:
        return f"Error during smartphone information retrieval for model {model}: {e}"


def generate_context(ai_message: AIMessage, chat_history) -> dict:
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


@app.post("/ask")
def ask(request: QueryRequest):
    user_input = request.user_input
    user_id = request.user_id
    session_id = request.session_id

    langfuse_client = get_client()

    chat_history = RedisChatMessageHistory(session_id=session_id, redis_url=REDIS_URL, ttl=300)

    trimmer = trim_messages(
        strategy="last",
        token_counter=llm,
        max_tokens=500,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
    )

    rails_config = RailsConfig.from_path("config")
    rails = RunnableRails(rails_config, input_key="user_input")

    with propagate_attributes(
        trace_name="ai-response",
        session_id=session_id,
        user_id=user_id,
    ):
        langfuse_handler = CallbackHandler()

        tools = [smartphone_info_tool]
        llm_with_tools = llm.bind_tools(tools)

        langfuse_prompts = Langfuse()

        context_langfuse_prompt = langfuse_prompts.get_prompt("context-prompt")
        review_langfuse_prompt = langfuse_prompts.get_prompt("review-prompt")

        context_prompt = ChatPromptTemplate.from_messages(
            [
                context_langfuse_prompt.get_langchain_prompt()[0],
                MessagesPlaceholder(variable_name="conversation"),
                context_langfuse_prompt.get_langchain_prompt()[1],
            ]
        )
        context_prompt.metadata = {"langfuse_prompt": context_langfuse_prompt}

        review_prompt = ChatPromptTemplate.from_messages(
            [
                review_langfuse_prompt.get_langchain_prompt()[0],
                MessagesPlaceholder(variable_name="conversation"),
                review_langfuse_prompt.get_langchain_prompt()[1],
            ]
        )
        review_prompt.metadata = {"langfuse_prompt": review_langfuse_prompt}

        # Check guardrails
        rail_result = rails.invoke({"user_input": user_input})
        if isinstance(rail_result, dict) and "I'm sorry, I can't respond to that" in rail_result.get("output", ""):
            return {"response": rail_result["output"]}

        chat_history.add_message(HumanMessage(content=user_input))

        trimmed_messages = trimmer.invoke(get_clean_messages(chat_history))

        # Context chain — get tool results
        context_result = (context_prompt | llm_with_tools).invoke(
            {"user_input": user_input, "conversation": trimmed_messages},
            config={
                "run_name": "context",
                "callbacks": [langfuse_handler],
                "metadata": {
                    "langfuse_user_id": user_id,
                    "langfuse_session_id": session_id,
                },
            },
        )
        generate_context(context_result, chat_history)

        # Review chain — generate final response
        response = (review_prompt | llm).invoke(
            {"user_id": user_id, "user_input": user_input, "conversation": get_clean_messages(chat_history)},
            config={
                "run_name": "final-response",
                "callbacks": [langfuse_handler],
                "metadata": {
                    "langfuse_user_id": user_id,
                    "langfuse_session_id": session_id,
                },
            },
        )

        chat_history.add_message(response)

        return {"response": response.content}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)