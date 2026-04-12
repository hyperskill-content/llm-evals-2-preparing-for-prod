import asyncio
import json
import os
import sys
import uuid

import dotenv
from langchain_community.docstore.document import Document
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langfuse import observe, propagate_attributes, get_client
from langfuse._client.span import LangfuseSpan
from langfuse.langchain import CallbackHandler
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from ragas_eval import score_observation

# Load environment variables from .env file
dotenv.load_dotenv()

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
    model="text-embedding-ada-002",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    show_progress_bar=True
)

langfuse_client = get_client()
if langfuse_client.auth_check():
    print("Langfuse client is authenticated and ready for use.")
else:
    print("Langfuse client authentication failed. Proceeding without Langfuse integration.")
langfuse_handler = CallbackHandler()

# Initialize conversation history
conversation = []


# ---------------------------
# Load JSON Data and Build Qdrant Vector Store
# ---------------------------
def read_docs(json_path):
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

    documents = [
        Document(
            page_content=(
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
        ) for entry in data]
    return documents


@observe(name="embed-documents", as_type="embedding")
def embed_documents(json_path: str) -> QdrantVectorStore:
    """
    Load JSON data from the smartphones.json file and convert each entry to a Document.
    :param
        json_path (str): Path to the JSON file containing smartphone data.

    :returns
        Qdrant vector store A Qdrant vector store built from the smartphone documents,
                or an empty list if an error occurs.
    """
    try:
        collection_name = "smartphones"
        qdrant_client = QdrantClient("http://localhost:6333")

        collection_exists = qdrant_client.collection_exists(collection_name=collection_name)
        if not collection_exists:
            documents = read_docs(json_path)

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
        raise Exception(f"Error initializing the vector store: {e}")


# Initialize the vector store
product_db = embed_documents("datasets/smartphones.json")


# ---------------------------
# Tool Definitions
# ---------------------------
@tool("SmartphoneInfo")
@observe(name="retrieval", as_type="retriever")
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
def get_context(user_input):
    llm_with_tools = llm.bind_tools([smartphone_info_tool])

    context_system_prompt = langfuse_client.get_prompt("context")
    langfuse_client.update_current_generation(prompt=context_system_prompt)

    context_prompt = ChatPromptTemplate.from_messages(
        [
            context_system_prompt.get_langchain_prompt()[0],  # system
            MessagesPlaceholder(variable_name="conversation"),
            context_system_prompt.get_langchain_prompt()[1]  # user
        ]
    )
    context_prompt.metadata = {"langfuse_prompt": context_system_prompt}

    context_chain = context_prompt | llm_with_tools | generate_context

    return context_chain.invoke(
        {"user_input": user_input, "conversation": conversation},
        config={
            "run_name": "context",
            "callbacks": [langfuse_handler],
            "metadata": {"langfuse_tags": ["dev", "test"]}
        }
    )

@observe(name="generate-context", as_type="retrieval")
def generate_context(ai_message: AIMessage) -> list[str]:
    """
    Process tool calls from the language model and collect their responses as ToolMessage objects.
    ToolMessage objects are appended to the conversation history.

    :param
        ai_message (AIMessage): The language model's output message containing tool_calls.

    :returns
        A list containing the content of the tool messages.
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
        context = []
        for tool_call in ai_message.tool_calls:
            print(f"Invoking tool: {tool_call}")
            if tool_call["name"] == "SmartphoneInfo":
                tool_output: ToolMessage = smartphone_info_tool.invoke(tool_call)
                context.append(tool_output.content)
                conversation.append(tool_output)
        return context

    except Exception as e:
        print(f"An error occurred while processing tool calls: {e}")
        conversation.append(
            AIMessage(
                content=f"An error occurred while processing tool calls: {e}"
            )
        )
        return []


# ---------------------------
# User Feedback Handling
# ---------------------------
def get_user_feedback() -> None:
    feedback = input("Was this answer helpful? (Yes/No): ")
    user_comment = input("Please give us a reason for your answer. This will help us improve: ")
    langfuse_client.score_current_trace(
        name="usefulness",
        value=feedback,
        data_type="CATEGORICAL",
        comment=user_comment
    )


# ---------------------------
# AI Message Generation
# ---------------------------
def generate_review(user_input: str, chunks: list[str]) :
    with langfuse_client.start_as_current_observation(name="generate-review",
                                                      as_type="generation",
                                                      input=[user_input, *chunks]) as obs:
        review_system_prompt = langfuse_client.get_prompt("review")
        langfuse_client.update_current_generation(prompt=review_system_prompt)

        review_prompt = ChatPromptTemplate.from_messages(
            [
                review_system_prompt.get_langchain_prompt()[0],
                MessagesPlaceholder(variable_name="conversation"),
                review_system_prompt.get_langchain_prompt()[1]
            ]
        )
        review_prompt.metadata = {"langfuse_prompt": review_system_prompt}
        review_chain = review_prompt | llm

        response = review_chain.invoke(
            {"user_id": user_id, "user_input": user_input, "conversation": conversation},
            config=RunnableConfig(
                run_name="ai-response",
                callbacks=[langfuse_handler],
                metadata={"langfuse_tags": ["dev", "test"]}
            )
        )

        scoring = asyncio.create_task(score_observation(obs, user_input, chunks, response))
    return response, scoring


@observe(name="goodbye-message", as_type="generation")
def goodbye(span: LangfuseSpan, user_input: str) -> None:
    goodbye_system_prompt = langfuse_client.get_prompt("goodbye")
    langfuse_client.update_current_generation(prompt=goodbye_system_prompt)

    goodbye_prompt = PromptTemplate.from_template(
        goodbye_system_prompt.get_langchain_prompt()[0][1] # [0] -> system message tuple, [1]-> its pure text
    )
    goodbye_prompt.metadata={"langfuse_prompt": goodbye_system_prompt}

    goodbye_chain = goodbye_prompt | llm
    goodbye_message = goodbye_chain.invoke(
        {"user_id": user_id},
        config=RunnableConfig(
            run_name="goodbye-message",
            callbacks=[langfuse_handler],
            metadata={"langfuse_tags": ["dev", "test", "final-response"]}
        )
    )
    span.update(name="goodbye-message", input=user_input, output=goodbye_message.content)
    print(f"System: {goodbye_message.content}")


# ---------------------------
# Main Conversation Loop
# ---------------------------
@observe(name="main-loop", as_type="span")
async def main():
    session_name = f"session-{uuid.uuid4().hex[:8]}"

    try:
        print("Welcome to the Smartphone Assistant! I can help you with smartphone features and comparisons.")

        with (propagate_attributes(session_id=session_name, user_id=user_id)):
            awaitables = set()
            while True:
                user_input = input("User: ").strip()

                with langfuse_client.start_as_current_observation(name="turn" ,input=user_input) as span:
                    conversation.append(HumanMessage(user_input))

                    # termination condition
                    if user_input.lower() in ["exit", "quit", "bye", "end"]:
                        goodbye(span, user_input)
                        get_user_feedback()

                        # wait for scoring tasks and flush before exiting
                        await asyncio.gather(*awaitables)
                        langfuse_client.flush()
                        break

                    # gather context and regenerate review
                    chunks: list[str] = get_context(user_input)
                    response, scoring_task = generate_review(user_input, chunks)
                    print(f"System: {response.content}")

                    # add scoring task to awaitables
                    awaitables.add(scoring_task)
                    scoring_task.add_done_callback(lambda _: awaitables.remove(scoring_task) if scoring_task in awaitables else None)


                    # update span and conversation history
                    span.update(output=response.content, input=[user_input, *chunks])
                    conversation.append(response)

    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
