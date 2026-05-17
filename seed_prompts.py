"""
Seed prompts into Langfuse for the smartphone info bot.
Run this once to create the initial prompt versions.
"""

from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

langfuse_client = Langfuse()

# Create the smartphone-assistant chat prompt in Langfuse
langfuse_client.create_prompt(
    name="smartphone-assistant",
    type="chat",
    prompt=[
        {
            "role": "system",
            "content": (
                "You are a helpful smartphone information assistant. "
                "Answer questions about smartphones using only the provided context. "
                "If the answer is not in the context, say you don't have that information. "
                "Be concise and accurate.\n\n"
                "Context:\n{{context}}"
            ),
        },
        {
            "role": "user",
            "content": "{{user_input}}",
        },
    ],
    labels=["production"],
)

print("Prompt 'smartphone-assistant' created successfully in Langfuse.")
