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
