import asyncio
import copy
import os
import sys
import uuid

import dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, trim_messages
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.globals import set_debug
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langfuse import Langfuse
from langfuse import LangfuseSpan
from langfuse import observe, propagate_attributes, get_client
from langfuse.langchain import CallbackHandler

from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

from ragas_eval import score_observation
from user_feedback import get_user_feedback
from knowledge_base import smartphone_info_tool

dotenv.load_dotenv()
# Uncomment for debugging
#set_debug(True)

users = ["James", "George", "Mike", "Sherlock"]
REDIS_URL = "redis://localhost:6380/0"
guardrails_config_in = RailsConfig.from_path("config/input/")
guardrails_config_out = RailsConfig.from_path("config/output/")
BLOCKED_QUERY_MSG = "I'm sorry, I can't respond to that."
BLOCKED_OUTPUT_MSG = "Sorry, something went wrong. Please try again."

def init_langfuse() -> Langfuse:
    """Initialize and return the Langfuse client, exiting if authentication fails."""
    client = get_client()
    if client.auth_check():
        return client
    else:
        print("Langfuse client authentication failed: Is the container running?")
        sys.exit(1)

def get_redis_history(session_id: str) -> BaseChatMessageHistory:
    """Return a Redis-backed chat message history for the given session, with a 120-second TTL."""
    return RedisChatMessageHistory(session_id, url=REDIS_URL, ttl=120)

class Session:
    """Holds per-conversation state: user identity, Langfuse observability handles, and Redis chat history."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.user_id = users[uuid.uuid4().int % len(users)]
        self.langfuse_client = init_langfuse()
        self.langfuse_handler = CallbackHandler()

        self.history = get_redis_history(self.session_id)

class ConfigWithSession(RunnableConfig):
    session_id: str


class Agent:
    """Orchestrates input/output guardrails, tool-based retrieval, and LLM response generation for a session."""

    def __init__(self, session: Session):
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        self.tools = [smartphone_info_tool]
        self.session = session
        self.config = ConfigWithSession(
            callbacks=[self.session.langfuse_handler],
            metadata={"langfuse_tags": ["dev", "test"]},
            session_id=self.session.session_id
        )
        self.trimmer = trim_messages(
            strategy="last", # keep either the last or first messages
            token_counter=self.llm, # use your LLM to count tokens or create a special function
            max_tokens=500, # the maximum number of tokens
            start_on="human", # the first message type in the trimmed history
            end_on=("human", "tool"), # the last message type in the trimmed history
            include_system=True, # always include the system message
        )

    # ---------------------------
    # Entry point for each turn in the conversation
    # ---------------------------
    async def query(self, user_input) -> tuple[BaseMessage, asyncio.Task|None]:
        """Run input guardrails, retrieve context, and generate a response for one conversation turn."""
        # Check user input first, since chaining does not work with RedisChatHistory.
        # We need to invoke guardrails asynchronously, since NeMo expects that when invoked within an async context.
        input_guardrails = RunnableRails(
            guardrails_config_in,
            llm=self.llm,
            runnable=RunnableLambda(lambda x: x["user_input"]),  # return string so output rails see a valid response
            input_key="user_input",
            output_key="output",
            input_blocked_message=BLOCKED_QUERY_MSG
        )
        rails_result = await input_guardrails.ainvoke({"user_input": user_input})
        output = rails_result.get("output") if isinstance(rails_result, dict) else rails_result
        if output == BLOCKED_QUERY_MSG:
            return AIMessage(content=BLOCKED_QUERY_MSG), None

        # Input ok -> gather context and regenerate review
        context_responses: list[AIMessage] = await self.get_context(user_input)
        chunks: list[str] = [response.content for response in context_responses]

        response, task = await self.generate_review(user_input, chunks)
        return response, task

    # ---------------------------
    # Tool Calling / Retrieval
    # ---------------------------
    async def get_context(self, user_input: str) -> list[AIMessage]:
        """Invoke tool-calling chain to retrieve relevant context for the user's query."""
        llm_with_tools = self.llm.bind_tools(self.tools)
        chat_history = get_redis_history(self.session.session_id)

        context_system_prompt = self.session.langfuse_client.get_prompt("context")
        self.session.langfuse_client.update_current_generation(prompt=context_system_prompt)
        system_part = context_system_prompt.get_langchain_prompt()[0]
        context_prompt = ChatPromptTemplate.from_messages(
            [ system_part, MessagesPlaceholder(variable_name="conversation") ]
        )
        context_prompt.metadata = {"langfuse_prompt": context_system_prompt}

        # We manually add the user message to the chat history before invoking the chain.
        # When passed as input_messages, it violates the AI message with tool calls followed by ToolMessages rule.
        chat_history.add_user_message(HumanMessage(content=user_input))

        # input_messages_key seems necessary, though, so we pass an empty list
        chain_with_history = RunnableWithMessageHistory(
            context_prompt | self.trimmer | llm_with_tools | self.generate_context,
            get_redis_history,
            input_messages_key="empty_list",
            history_messages_key="conversation",
        )

        config = copy.copy(self.config)
        config["run_name"] = "get_context"

        return chain_with_history.invoke(
            input={"user_input": user_input, "conversation": chat_history, "empty_list": []},
            config=config
        )

    @observe(name="generate-context", as_type="retriever")
    def generate_context(self, ai_message: AIMessage) -> list[AIMessage]:
        """
        Process tool calls from the language model and collect their responses as AIMessage objects.
        AIMessage objects are used as a workaround for the current limitations of the current
        Redis-based chat history management.

        :param
            ai_message (AIMessage): The language model's output message containing tool_calls.

        :returns
            A list containing AIMessages with the content of the tool messages.
        """
        # Check if the AI message has any tool calls
        if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
            return [AIMessage(content="Use the chat history to answer the user's question.")]

        try:
            # Process each tool call, invoke the appropriate tool, and append the result to the conversation
            # a message with tool calls is expected to be followed by tool responses
            chat_history = get_redis_history(self.session.session_id)
            chat_history.add_message(ai_message)
            results = []
            for tool_call in ai_message.tool_calls:
                if tool_call["name"] == "SmartphoneInfo":
                    tool_output: ToolMessage = smartphone_info_tool.invoke(tool_call)
                    results.append(tool_output)
            return results

        except Exception as e:
            print(f"An error occurred while processing tool calls: {e}")
            return [AIMessage(content=f"An error occurred while processing tool calls: {e}")]

    # ---------------------------
    # Message Generation
    # ---------------------------
    async def generate_review(self, user_input: str, chunks: list[str]) -> tuple[BaseMessage, asyncio.Task]:
        """Generate and output-guard a review response, then kick off async RAGAS scoring."""
        chat_history = get_redis_history(self.session.session_id)

        with (self.session.langfuse_client.start_as_current_observation(
                name="generate-review",
                as_type="generation",
                input={"user_input": user_input, "chunks": chunks}
        ) as obs):
            review_system_prompt = self.session.langfuse_client.get_prompt("review")
            self.session.langfuse_client.update_current_generation(prompt=review_system_prompt)

            review_prompt = ChatPromptTemplate.from_messages(
                [
                    review_system_prompt.get_langchain_prompt()[0],
                    MessagesPlaceholder(variable_name="conversation"),
                    review_system_prompt.get_langchain_prompt()[1]
                ]
            )
            review_prompt.metadata = {"langfuse_prompt": review_system_prompt}

            review_chain = review_prompt | self.llm
            chain_with_history = RunnableWithMessageHistory(
                review_chain, get_redis_history, input_messages_key="empty_list", history_messages_key="conversation",
            )

            config = copy.copy(self.config)
            config["run_name"] = "ai_response"
            response = await chain_with_history.ainvoke(
                input={"user_id": self.session.user_id, "user_input": user_input, "conversation": chat_history,
                       "empty_list": []},
                config=config
            )

            # Check the generated response through output rails (after history is written, avoiding serialization issues)
            output_rails = RunnableRails(
                guardrails_config_out,
                llm=self.llm,
                runnable=RunnableLambda(lambda x: {"output": x["ai_response"]}),
                input_key="ai_response",
                output_key="output",
                output_blocked_message=BLOCKED_OUTPUT_MSG,
            )

            rails_result = await output_rails.ainvoke({"ai_response": response.content})
            content = rails_result.get("output", response.content) if isinstance(rails_result, dict) else rails_result
            response = AIMessage(content=content)
            self.session.langfuse_client.update_current_generation(output=response.content)

            scoring = asyncio.create_task(score_observation(obs, user_input, chunks, response))
        return response, scoring

    @observe(name="goodbye-message", as_type="generation")
    async def goodbye(self, span: LangfuseSpan, user_input: str) -> None:
        """Generate and print a personalized goodbye message, then update the Langfuse span."""
        goodbye_system_prompt = self.session.langfuse_client.get_prompt("goodbye")
        self.session.langfuse_client.update_current_generation(prompt=goodbye_system_prompt)

        goodbye_prompt = PromptTemplate.from_template(
            goodbye_system_prompt.get_langchain_prompt()[0][1]  # [0] -> system message tuple, [1]-> its pure text
        )
        goodbye_prompt.metadata = {"langfuse_prompt": goodbye_system_prompt}

        goodbye_chain = goodbye_prompt | self.llm
        goodbye_message = await goodbye_chain.ainvoke(
            {"user_id": self.session.user_id},
            config=RunnableConfig(
                run_name="goodbye-message",
                callbacks=[self.session.langfuse_handler],
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
    """Run the interactive smartphone assistant conversation loop until the user exits."""
    session_id = f"session-{uuid.uuid4().hex[:8]}"
    session = Session(session_id)
    agent = Agent(session)

    try:
        print("Welcome to the Smartphone Assistant! I can help you with smartphone features and comparisons.")

        with (propagate_attributes(session_id=session_id, user_id=session.user_id)):
            awaitables = set()
            while True:
                user_input = input("User: ").strip()

                with session.langfuse_client.start_as_current_observation(name="turn" ,input=user_input) as span:

                    # termination condition
                    if user_input.lower() in ["exit", "quit", "bye", "end"]:
                        await agent.goodbye(span, user_input)
                        get_user_feedback(session.langfuse_client)

                        # wait for scoring tasks and flush before exiting
                        print("Shutting down, please be patient.", end="", flush=True)
                        await asyncio.gather(*awaitables)

                        session.langfuse_client.flush()
                        break

                    # guard input
                    response, scoring_task = await agent.query(user_input)
                    print(f"System: {response.content}")

                    # add scoring task for this turn to awaitables
                    if scoring_task:
                        awaitables.add(scoring_task)
                        scoring_task.add_done_callback(lambda _: awaitables.remove(scoring_task) if scoring_task in awaitables else None)

                    # update span and conversation history
                    span.update(output=response.content)

    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
        raise e


if __name__ == "__main__":
    asyncio.run(main())
