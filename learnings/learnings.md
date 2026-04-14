### Task 1 : Langfuse Prompt Management and Versioning

To manage and version prompts using Langfuse and LangChain:

1. **Retrieve Prompts:** Use the Langfuse client to fetch defined prompts by their names.
   ```python
   langfuse_prompt = langfuse_client.get_prompt("prompt_name")
   ```

2. **Convert to LangChain Format:** Use `get_langchain_prompt()` to convert the Langfuse prompt into a format compatible with LangChain's `ChatPromptTemplate`.
   - For simple prompts:
     ```python
     prompt = ChatPromptTemplate.from_messages([langfuse_prompt.get_langchain_prompt()[0]])
     ```
   - For prompts with conversation history:
     ```python
     prompt = ChatPromptTemplate.from_messages([
         langfuse_prompt.get_langchain_prompt()[0],
         MessagesPlaceholder(variable_name="conversation"),
         langfuse_prompt.get_langchain_prompt()[1]
     ])
     ```

3. **Link for Tracking:** Associate the original Langfuse prompt object with the LangChain prompt's metadata. This enables versioning, metrics, and cost tracking in the Langfuse UI.
   ```python
   prompt.metadata = {"langfuse_prompt": langfuse_prompt}
   ```

---

### Task 2: Redis-based Chat History Management

To replace in-memory conversation lists with persistent, TTL-aware Redis storage using LangChain:

#### 1. Redis Setup via Docker
Map container port `6379` to host port `6380` (avoids collision with Langfuse's default port):
```bash
docker run --restart always --name hyper-redis -d -p 6380:6379 redis redis-server --save 60 1
```
Install required packages:
```bash
pip install langchain-redis redis
```

#### 2. Initialize Chat History with TTL
Use `RedisChatMessageHistory` with a `ttl` (in seconds) to automatically expire old sessions:
```python
from langchain_redis import RedisChatMessageHistory

REDIS_URL = "redis://localhost:6380/0"

def get_redis_history(session_id: str):
    return RedisChatMessageHistory(session_id, redis_url=REDIS_URL, ttl=600)  # 10 minutes
```

#### 3. Workaround: tool_calls Serialization Bug in `langchain_redis`
`AIMessage.tool_calls` are not correctly serialized/deserialized by `langchain_redis`. The fix is to manually serialize them into `additional_kwargs` before saving, and restore them after loading:
```python
# Before saving to Redis:
if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
    ai_message.additional_kwargs["_tool_calls"] = json.dumps(ai_message.tool_calls)

# After loading from Redis:
def get_history_messages(session_id: str):
    history = get_redis_history(session_id)
    messages = history.messages
    for m in messages:
        if isinstance(m, AIMessage) and "_tool_calls" in m.additional_kwargs:
            try:
                m.tool_calls = json.loads(m.additional_kwargs["_tool_calls"])
            except Exception:
                pass
    return messages
```

#### 4. Trim Messages to Control Token Usage
Use `trim_messages()` to keep only the most recent messages within a token budget before passing history to the LLM:
```python
from langchain_core.messages import trim_messages

trimmer = trim_messages(
    strategy="last",        # keep the most recent messages
    token_counter=llm,      # use the LLM to count tokens
    max_tokens=600,
    start_on="human",       # trimmed history must start with a human message
    include_system=True,    # always keep the system message
)

# Apply before each chain invocation:
conversation = trimmer.invoke(get_history_messages(session_id))
```

#### 5. Integration Pattern with Chains
The general pattern for a turn in the conversation loop:
```python
# 1. Save human message
history.add_user_message(user_input)

# 2. Retrieve and trim history
conversation = trimmer.invoke(get_history_messages(session_id))

# 3. Run context chain (tool calls)
ai_msg_with_tools = context_chain.invoke({"user_input": user_input, "conversation": conversation}, ...)

# 4. Save tool call results to Redis
generate_context(ai_msg_with_tools, session_id)

# 5. Re-trim for final response chain
conversation = trimmer.invoke(get_history_messages(session_id))

# 6. Generate and save final response
response = review_chain.invoke({"conversation": conversation, ...}, ...)
history.add_ai_message(response.content)
```

#### Key Design Decisions
- **TTL over `clear()`:** Setting a TTL on the Redis key is preferable to manually calling `clear()`, as it automatically expires sessions without extra logic.
- **Two trims per turn:** Trimming is applied twice — once before the tool-call chain and once before the final response chain — to ensure the token budget is respected at both stages.
- **Session ID via UUID:** Each conversation gets a unique `session_id` (e.g. `session-{uuid4().hex[:8]}`), allowing multiple concurrent users without history collisions.
