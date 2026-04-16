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

---

### Task 3: NeMo Guardrails for Input Validation

#### 1. Configuration Files

Two files are required under `config/`:

**`config/config.yml`** — defines the model, general instructions, and which rails to activate:
```yaml
models:
  - type: main
    engine: openai
    model: gpt-4o-mini

instructions:
  - type: general
    content: |
      You verify inputs for a bot called the Smartphones info Bot.
      ...

rails:
  input:
    flows:
      - self check input
```

**`config/prompts.yml`** — defines the self-check prompt for the `self_check_input` task:
```yaml
prompts:
  - task: self_check_input
    content: |
      Your task is to check if the user message below follows guidelines...
      User message: "{{ user_input }}"
      Question: Should the user message be blocked (Yes or No)?
      Answer:
```

#### 2. Integrating RunnableRails into a LangChain Chain

NeMo Guardrails wraps chains via `RunnableRails`. The `input_key` parameter must match the variable name used for user input in the chain (default is `"input"`):
```python
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

rails_config = RailsConfig.from_path("config")
chain_with_rails = RunnableRails(rails_config, runnable=my_chain, input_key="user_input")
```

When the input is blocked, the chain returns a dict with `{"output": "I'm sorry, I can't respond to that."}`. This string is defined as `REFUSAL_MESSAGE` in `nemoguardrails.guardrails.iorails` and can be imported to avoid hardcoding:
```python
from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE

if isinstance(result, dict) and result.get("output") == REFUSAL_MESSAGE:
    print(f"System: {result['output']}")
```

#### 3. Serialization Problem with LangChain Message Objects

`RunnableRails` cannot serialize complex LangChain message objects (`HumanMessage`, `AIMessage`, `ToolMessage`) that are passed alongside the user input. The workaround is to serialize messages to dicts before passing to `RunnableRails` and deserialize them inside a wrapper chain:

```python
from langchain_core.messages import messages_from_dict, message_to_dict
from langchain_core.runnables import RunnableLambda

# Wrapper: deserializes conversation dicts → LangChain message objects
context_chain_deserializer = RunnableLambda(
    lambda x: context_chain.invoke({
        "user_input": x["user_input"],
        "conversation": messages_from_dict(x.get("conversation", [])),
    })
)

chain_with_rails = RunnableRails(rails_config, runnable=context_chain_deserializer, input_key="user_input")

# When invoking: serialize first
serialized_conversation = [message_to_dict(m) for m in conversation]
result = chain_with_rails.invoke({"user_input": user_input, "conversation": serialized_conversation})
```

#### 4. Avoid Double LLM Calls

When using `RunnableRails` **without** a `runnable`, NeMo generates its own LLM response after the input check passes — which gets discarded if you then call the real chain separately. This wastes tokens:

```python
# Schlecht: 2 LLM-Aufrufe wenn Rail nicht triggert
rails.invoke({"user_input": ...})   # NeMo generiert unnötige Antwort
context_chain.invoke(...)           # eigentliche Antwort

# Gut: 1 LLM-Aufruf — NeMo ruft context_chain direkt auf
RunnableRails(config, runnable=context_chain_deserializer, input_key="user_input")
```

#### 5. Logging

To see the self-check result (`Yes`/`No`) in the console without flooding logs, set only the NeMo actions logger to `INFO`:
```python
import logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("actions.py").setLevel(logging.INFO)
```
Output example:
```
00:34:05 actions.py INFO   Input self-checking result is: `Yes`.

---

### Task 4: LiteLLM Proxy and Session-aware Logging

#### 1. LiteLLM Proxy Setup & Virtual Keys
LiteLLM provides a unified interface for multiple LLM providers and adds management features like budgets and rate limits via **Virtual Keys**.

**Generate a Virtual Key via API:**
```bash
curl 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer <master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{"models": ["gpt-4o-mini"], "max_budget": 1.0, "budget_duration": "1d"}'
```

**Client Configuration:**
Set the `OPENAI_BASE_URL` to your proxy and use the virtual key as `OPENAI_API_KEY`.

#### 2. Session & User Attribution in LiteLLM
To group logs and track costs per user/session in the LiteLLM Dashboard, metadata must be passed with every request.

**Binding Metadata to LangChain LLMs:**
Use `.bind()` to attach the `user` and `extra_body` (for custom metadata) to the LLM instance.
```python
# Bind user and session metadata
llm_session = llm.bind(
    user=user_id,
    extra_body={"metadata": {"session_id": session_id}}
)

# Use the bound LLM in your chains
chain = prompt | llm_session
```

#### 3. Tracking RAG (Embeddings) in Sessions
Embedding calls are often logged as separate, unrelated entries. To link them to a chat session, you must pass the same metadata to the embedding model.

**Session-aware Embeddings in Tools:**
```python
def smartphone_info_tool(model: str, user_id: str, session_id: str):
    # Create transient embeddings instance with metadata
    tool_embeddings = OpenAIEmbeddings(
        ...,
        model_kwargs={
            "user": user_id,
            "extra_body": {"metadata": {"session_id": session_id, "tool": "SmartphoneInfo"}}
        }
    )
    # Use this instance for vector store retrieval
    product_db = QdrantVectorStore.from_existing_collection(embedding=tool_embeddings, ...)
    ...
```

#### 4. Preserving Metadata through Guardrails
When using `RunnableRails` with a `runnable`, metadata passed in the `config` of the outer `invoke()` call can get lost. To fix this, inject the metadata explicitly inside the `RunnableLambda` wrapper:

```python
context_chain_deserializer = RunnableLambda(
    lambda x: context_chain.invoke(
        {...},
        config={
            "metadata": {
                "langfuse_session_id": session_id,
                "langfuse_user_id": user_id,
            }
        }
    )
)
```

#### Key Benefits
- **Consolidated Logs:** Filtering by `session_id` in LiteLLM shows the entire trace (LLM calls, Guardrails checks, and Embeddings).
- **Accurate Budgeting:** Costs for both Chat and Embeddings are aggregated under the correct Virtual Key and User ID.
- **Improved Debugging:** Metadata like `"tool": "SmartphoneInfo"` helps identify which component triggered a specific API call.

```
