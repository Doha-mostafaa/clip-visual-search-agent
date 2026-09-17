import sys
import pandas as pd

from src.pipeline.search_pipeline import search_similar_products
from src.logger import logging
from src.exception import CustomException


def search_products_tool(query_image, top_k: int = 5) -> str:
    """
    Runs the CV search pipeline and formats the results as a
    readable text block the agent (LLM) can reason over.
    """
    try:
        logging.info("Running search_products_tool...")

        results = search_similar_products(query_image, top_k=top_k)

        formatted_lines = []
        for i, row in results.iterrows():
            formatted_lines.append(
                f"{i + 1}. {row['productDisplayName']} "
                f"(Category: {row['masterCategory']}, "
                f"Price: {row['price']} EGP, "
                f"Similarity: {row['similarity_score']:.2f})"
            )

        formatted_text = "\n".join(formatted_lines)

        logging.info("search_products_tool completed successfully")

        return formatted_text, results

    except Exception as e:
        raise CustomException(e, sys)


def compare_price_to_average_tool(results: pd.DataFrame) -> str:
    """
    Takes the search results DataFrame and compares each product's
    price to the average price within the same category.
    """
    try:
        logging.info("Running compare_price_to_average_tool...")

        category_avg = results.groupby("masterCategory")["price"].transform("mean")
        results = results.copy()
        results["category_avg_price"] = category_avg.round(2)
        results["price_vs_avg"] = (results["price"] - results["category_avg_price"]).round(2)

        formatted_lines = []
        for i, row in results.iterrows():
            direction = "above" if row["price_vs_avg"] > 0 else "below"
            formatted_lines.append(
                f"{row['productDisplayName']}: {row['price']} EGP "
                f"({abs(row['price_vs_avg']):.2f} EGP {direction} the "
                f"category average of {row['category_avg_price']} EGP)"
            )

        formatted_text = "\n".join(formatted_lines)

        logging.info("compare_price_to_average_tool completed successfully")

        return formatted_text

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import load_balanced_subset

    balanced_df, full_dataset = load_balanced_subset(samples_per_category=20)
    test_image = full_dataset[0]["image"]

    search_text, search_results = search_products_tool(test_image, top_k=5)
    print("=== Search Results ===")
    print(search_text)

    print("\n=== Price Comparison ===")
    price_text = compare_price_to_average_tool(search_results)
    print(price_text)