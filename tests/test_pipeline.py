import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytest

from src.components.data_ingestion import load_balanced_subset, add_fake_prices
from src.components.embedding_generator import load_clip_model, generate_image_embeddings
from src.pipeline.search_pipeline import search_similar_products


@pytest.fixture(scope="module")
def small_dataset():
    """
    Loads a very small subset once and reuses it across all tests
    in this file, to keep the test suite fast.
    """
    balanced_df, full_dataset = load_balanced_subset(samples_per_category=5)
    balanced_df = add_fake_prices(balanced_df)
    return balanced_df, full_dataset


def test_fake_prices_are_valid(small_dataset):
    balanced_df, _ = small_dataset

    assert "price" in balanced_df.columns
    assert (balanced_df["price"] > 0).all()


def test_embedding_dimension(small_dataset):
    balanced_df, full_dataset = small_dataset

    subset_dataset = full_dataset.select(balanced_df["orig_idx"].tolist()[:3])
    images = subset_dataset["image"]

    model, processor, device = load_clip_model()
    embeddings = generate_image_embeddings(images, model, processor, device)

    assert embeddings.shape[0] == 3
    assert embeddings.shape[1] == 512


def test_search_returns_correct_top_k(small_dataset):
    balanced_df, full_dataset = small_dataset
    test_image = full_dataset[int(balanced_df["orig_idx"].iloc[0])]["image"]

    results = search_similar_products(test_image, top_k=3)

    assert len(results) == 3
    assert "similarity_score" in results.columns