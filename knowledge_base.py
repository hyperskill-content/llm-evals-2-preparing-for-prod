import json
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langfuse import observe
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

load_dotenv()


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


embeddings_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    show_progress_bar=True
)
product_db = embed_documents(json_path="datasets/smartphones.json")


# ---------------------------
# Tool Definition
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
