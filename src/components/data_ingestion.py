import os
import sys
import random
import pandas as pd
from datasets import load_dataset

from src.logger import logging
from src.exception import CustomException


def load_balanced_subset(samples_per_category: int = 100, seed: int = 42):
    """
    Downloads the dataset from Hugging Face and returns a balanced
    subset with an equal number of samples from each masterCategory.
    """
    try:
        logging.info("Starting dataset download from Hugging Face...")

        dataset = load_dataset("ashraq/fashion-product-images-small", split="train")
        df = dataset.to_pandas()

        logging.info(f"Full dataset loaded, total rows: {len(df)}")

        categories = df["masterCategory"].unique()
        logging.info(f"Categories found: {categories}")

        balanced_rows = []
        random.seed(seed)

        for category in categories:
            category_df = df[df["masterCategory"] == category]
            n = min(samples_per_category, len(category_df))
            sampled = category_df.sample(n=n, random_state=seed)
            balanced_rows.append(sampled)

        balanced_df = pd.concat(balanced_rows)
        balanced_df["orig_idx"] = balanced_df.index  # keep original dataset index before resetting
        balanced_df = balanced_df.reset_index(drop=True)

        logging.info(f"Balanced subset created, total rows: {len(balanced_df)}")

        return balanced_df, dataset

    except Exception as e:
        raise CustomException(e, sys)


def add_fake_prices(df: pd.DataFrame):
    """
    Adds a synthetic but category-aware price column, since the
    original dataset has no real price data.
    """
    try:
        price_ranges = {
            "Apparel": (150, 500),
            "Footwear": (400, 1500),
            "Accessories": (100, 800),
            "Personal Care": (50, 300),
        }

        def generate_price(category):
            low, high = price_ranges.get(category, (100, 500))
            return round(random.uniform(low, high), 2)

        df["price"] = df["masterCategory"].apply(generate_price)

        logging.info("Fake prices added successfully")
        return df

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    balanced_df, full_dataset = load_balanced_subset(samples_per_category=100)
    balanced_df = add_fake_prices(balanced_df)
    print(balanced_df[["productDisplayName", "masterCategory", "price"]].head(10))