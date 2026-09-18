import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from datasets import load_dataset

from src.components.index_builder import load_index_and_metadata
from src.components.embedding_generator import load_clip_model, generate_image_embeddings
from src.logger import logging
from src.exception import CustomException


def search_similar_products(query_image, top_k: int = 5):
    """
    Takes a single PIL image and returns the top_k most similar
    products from the catalog, including their actual images.
    """
    try:
        logging.info("Loading FAISS index and metadata...")
        index, metadata_df = load_index_and_metadata()

        logging.info("Loading CLIP model...")
        model, processor, device = load_clip_model()

        logging.info("Generating embedding for query image...")
        query_embedding = generate_image_embeddings([query_image], model, processor, device)

        logging.info(f"Searching for top {top_k} similar products...")
        distances, indices = index.search(query_embedding.astype("float32"), top_k)

        results = metadata_df.iloc[indices[0]].copy()
        results["similarity_score"] = distances[0]
        results = results.reset_index(drop=True)

        logging.info("Fetching actual product images...")
        full_dataset = load_dataset("ashraq/fashion-product-images-small", split="train")
        images = [full_dataset[int(idx)]["image"] for idx in results["orig_idx"]]
        results["image"] = images

        logging.info("Search completed successfully")

        return results

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import load_balanced_subset

    balanced_df, full_dataset = load_balanced_subset(samples_per_category=20)
    test_image = full_dataset[0]["image"]

    results = search_similar_products(test_image, top_k=5)
    print(results[["productDisplayName", "price", "similarity_score"]])
    print(f"\nFirst result image type: {type(results['image'].iloc[0])}")