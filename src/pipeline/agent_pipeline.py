import os
import sys

from groq import Groq
from dotenv import load_dotenv

from src.components.agent_tools import search_products_tool, compare_price_to_average_tool
from src.logger import logging
from src.exception import CustomException

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a helpful shopping assistant. You are given a list of
products found via visual similarity search, along with a price comparison
against the category average. Your job is to analyze this data and answer
the user's question clearly and concisely, referencing specific product
names and prices from the data. Do not invent products or prices that are
not in the given data.

When evaluating what counts as "best" or "good value", consider these
factors together, not just one in isolation:
1. Similarity score: how closely the product matches what the user is
   looking for (higher is better).
2. Price relative to the category average (lower is generally better
   value, but don't ignore similarity to save money on an unrelated item).
3. If the user mentions a budget, prioritize options within that budget,
   but explicitly mention if a cheaper option (even if further from the
   budget ceiling) offers better overall value.

Briefly explain your reasoning and the trade-off you made, not just the
final recommendation."""


def run_shopping_agent(query_image, user_question: str, top_k: int = 5) -> str:
    """
    Runs the full agent pipeline: search -> price comparison -> LLM reasoning.
    """
    try:
        logging.info("Running search tool...")
        search_text, search_results = search_products_tool(query_image, top_k=top_k)

        logging.info("Running price comparison tool...")
        price_text = compare_price_to_average_tool(search_results)

        context = (
            f"Search Results:\n{search_text}\n\n"
            f"Price Comparison:\n{price_text}"
        )

        user_prompt = f"{context}\n\nUser question: {user_question}"

        logging.info("Sending request to Groq...")

        response = client.chat.completions.create(
            model="groq/compound-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )

        answer = response.choices[0].message.content

        logging.info("Agent responded successfully")

        return answer

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import load_balanced_subset

    balanced_df, full_dataset = load_balanced_subset(samples_per_category=20)
    test_image = full_dataset[0]["image"]

    answer = run_shopping_agent(
        test_image,
        user_question="What's the best value option here, and why?"
    )

    print("=== Agent Answer ===")
    print(answer)