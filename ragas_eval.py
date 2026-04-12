import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
# wrappers
from ragas.llms import llm_factory
from ragas.metrics import MetricResult
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
)

load_dotenv()


# Initialize the LLM with OpenAI API credentials
openai_client=client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
llm = llm_factory(
    model='gpt-4o-mini',
    client=openai_client
)

# Initialize the embeddings model with OpenAI API credentials
embeddings_model = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING"),
    client=openai_client
)

class ReviewScore:
    def __init__(self, faithfulness, relevancy, precision):
        self.scores = {
            'faithfulness': faithfulness,
            'relevancy': relevancy,
            'precision': precision
        }

async def score_faithfulness(query, chunks, answer) -> MetricResult:
    if not chunks:
        return MetricResult(value=None, reason="No retrieved contexts")
    result = await Faithfulness(llm).ascore(
        user_input=query,
        retrieved_contexts=chunks,
        response=answer
    )
    return result

async def score_relevancy(query, answer) -> MetricResult:
    result = await AnswerRelevancy(llm, embeddings_model).ascore(
        user_input=query,
        response=answer
    )
    return result

async def score_context_precision(query, chunks, answer) -> MetricResult:
    if not chunks:
        return MetricResult(value=None, reason="No retrieved contexts")
    result = await ContextPrecisionWithoutReference(llm).ascore(
        user_input=query,
        retrieved_contexts=chunks,
        response=answer
    )
    return result

async def score_review(query, chunks, answer) -> ReviewScore:
    (faithfulness, relevancy, context_precision) = await asyncio.gather(
        score_faithfulness(query, chunks, answer),
        score_relevancy(query, answer),
        score_context_precision(query, chunks, answer)
    )
    return ReviewScore(faithfulness, relevancy, context_precision)


async def score_observation(observation, user_input, chunks, response) -> None:
    score: ReviewScore = await score_review(
        query=user_input,
        chunks=chunks,
        answer=response.content
    )
    for key, entry in score.scores.items():
        if entry.value is not None:
            print(f"Adding metric {key}: {entry}")
            observation.score(
                name=key,
                value=entry.value,
                comment=f"{entry.reason if entry.reason else 'No reason provided'}",
            )
