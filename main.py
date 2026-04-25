import json
import os
import sys
import dotenv
from langchain_community.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe

dotenv.load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

# Initialize the LLM
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Initialize the embeddings model
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    show_progress_bar=True,
)

# Global flag to signal session end
_session_ending = False

# ---------------------------
# Load prompts from Langfuse
# ---------------------------
def load_prompts():
    """Fetch managed prompts from Langfuse and build LangChain templates."""
    lf_assistant = langfuse.get_prompt("smartphone-assistant-prompt")
    lf_final = langfuse.get_prompt("smartphone-final-response-prompt")

    # Context/tool-routing prompt: system + chat history + user message
    assistant_prompt = ChatPromptTemplate.from_messages(
        [
            lf_assistant.get_langchain_prompt()[0],          # system
            MessagesPlaceholder(variable_name="chat_history"),
            lf_assistant.get_langchain_prompt()[1],          # user: "You have been asked: {{user_input}}"
        ]
    )
    # Link the Langfuse prompt so metrics appear in the UI
    assistant_prompt.metadata = {"langfuse_prompt": lf_assistant}

    # Final response prompt: system + context + user query
    final_prompt = ChatPromptTemplate.from_messages(
        [
            lf_final.get_langchain_prompt()[0],              # system
            MessagesPlaceholder(variable_name="chat_history"),
            lf_final.get_langchain_prompt()[1],              # user: "User query / Context"
        ]
    )
    final_prompt.metadata = {"langfuse_prompt": lf_final}

    return assistant_prompt, final_prompt, lf_assistant, lf_final

# ---------------------------
# Load JSON Data and Build Qdrant Vector Store
# ---------------------------
def embed_documents(json_path: str):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {json_path} was not found.")
        return []
    except json.JSONDecodeError as jde:
        print(f"Error decoding JSON: {jde}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
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

# ---------------------------
# Tool Definitions
# ---------------------------

@observe(name="SmartphoneInfo")
def _get_smartphone_info(model: str) -> str:
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
        return f"Error during smartphone information retrieval: {e}"


@tool("EndSession")
def end_session_tool(session_status: str):
    """Ends the current session when the user is done."""
    global _session_ending
    _session_ending = True
    prompt = (
        "You are an AI assistant specialized in answering questions about smartphone features. "
        "Provide a polite goodbye message and thank the user for their feedback."
    )
    try:
        goodbye_message = llm.invoke(prompt).content
        print(goodbye_message)
    except Exception:
        print("Thank you for visiting. Goodbye!")
    print("\nRate the model's responses from 1 to 5 (3 being average):")
    return "Session ended."

# ---------------------------
# Observed helper functions
# ---------------------------

@observe(name="generate_context")
def generate_context(llm_tools):
    generated_context = []
    for tool_call in llm_tools.tool_calls:
        if tool_call["name"] == "SmartphoneInfo":
            tool_response = smartphone_info_tool.invoke(tool_call)
            generated_context.append(tool_response)
        elif tool_call["name"] == "EndSession":
            end_session_tool.invoke(tool_call)
        else:
            generated_context.append("No tool found for this query.")
            sys.exit(0)
    langfuse_context.update_current_observation(
        input=str(llm_tools.tool_calls),
        output=str(generated_context),
        metadata={"tool_calls": len(llm_tools.tool_calls)},
    )
    return generated_context


@observe(name="context")
def get_context(prompt, llm_with_tools, chat_history, user_input):
    chain = prompt | llm_with_tools | generate_context
    result = chain.invoke({"chat_history": chat_history, "user_input": user_input})
    langfuse_context.update_current_observation(
        input=user_input, output=str(result), metadata={"chat_history_length": len(chat_history)}
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
        input=user_input, output=response.content, metadata={"model": os.getenv("OPENAI_MODEL")}
    )
    return response

# ---------------------------
# Main Conversation Loop
# ---------------------------

@observe(name="main")
def main():
    global _session_ending
    _session_ending = False

    # Load managed prompts from Langfuse
    assistant_prompt, final_prompt, lf_assistant, lf_final = load_prompts()

    tools = [smartphone_info_tool, end_session_tool]
    llm_with_tools = llm.bind_tools(tools)

    chat_history = []
    last_question = ""
    last_answer = ""
    last_contexts = []

    print("Welcome to the Smartphone Assistant! I can help you with smartphone features and comparisons.")

    while True:
        user_input = input("User: ").strip()

        context = get_context(assistant_prompt, llm_with_tools, chat_history, user_input)

        if _session_ending:
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

        response = get_final_response(final_prompt, chat_history, user_input, context)
        last_question = user_input
        last_answer = response.content

        if isinstance(context, list):
            last_contexts = [c.content if hasattr(c, "content") else str(c) for c in context]
        else:
            last_contexts = [str(context)]

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response.content))

        print(f"System: {response.content}")


if __name__ == "__main__":
    product_db = embed_documents("datasets/smartphones.json")
    main()
