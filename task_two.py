import os
import uuid

import dotenv
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_redis import RedisChatMessageHistory

dotenv.load_dotenv()

users = ["James", "George", "Mike", "Sherlock"]
user_id = users[uuid.uuid4().int % len(users)]
session_name = f"session-{uuid.uuid4().hex[:8]}"
servicename = os.getenv("OTEL_SERVICE_NAME")
os.environ["OTEL_SERVICE_NAME"] = os.getenv("OTEL_SERVICE_NAME")
basic_tag = "LLM-eval2"

REDIS_URL = "redis://localhost:6380/0"
HISTORY_TTL_SECONDS = 3600

DEFAULT_SESSION_ID = "hyper_1"



llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

chat_history = RedisChatMessageHistory(
    session_id=DEFAULT_SESSION_ID,
    redis_url=REDIS_URL,
    ttl=HISTORY_TTL_SECONDS
)


# ---------------------
# Basic process
# ---------------------

system_prompt = "You are a helpful assistant."

user_input = input("User: >").strip()

chat_history.add_message(SystemMessage(content=system_prompt)) # to add a system message (you may not need to add this to the chat history)
chat_history.add_message(HumanMessage(content=user_input)) # to add a user's input
response = llm.invoke(chat_history.messages)
chat_history.add_ai_message(AIMessage(content=response.content)) # to add an AI response

print(f"System: {response.content}")

print("Chat History:")
for message in chat_history.messages:
    print(f"{message.type}: {message.content}")

# ---------------------
# Search for messages
# ---------------------
search_results = chat_history.search_messages("smartphone")
for result in search_results:
    print("")
    print(f"{result['type']}: {result['content'][:100]}")