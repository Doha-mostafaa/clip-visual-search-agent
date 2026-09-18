import os
import sys
import pickle

import faiss
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException


PROCESSED_DIR = os.path.join("data", "processed")
INDEX_PATH = os.path.join(PROCESSED_DIR, "faiss.index")
METADATA_PATH = os.path.join(PROCESSED_DIR, "metadata.pkl")


def build_faiss_index(embeddings: np.ndarray):
    """
    Builds a FAISS index using exact search (Inner Product),
    which is equivalent to cosine similarity since our embeddings
    are already normalized.
    """
    try:
        logging.info("Building FAISS index...")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)

        # FAISS requires float32 specifically
        embeddings = embeddings.astype("float32")
        index.add(embeddings)

        logging.info(f"FAISS index built successfully. Total vectors: {index.ntotal}")

        return index

    except Exception as e:
        raise CustomException(e, sys)


def save_index_and_metadata(index, metadata_df: pd.DataFrame):
    """
    Saves the FAISS index and the matching metadata (product info)
    to disk, so they don't need to be rebuilt every run.
    """
    try:
        os.makedirs(PROCESSED_DIR, exist_ok=True)

        faiss.write_index(index, INDEX_PATH)
        logging.info(f"FAISS index saved to: {INDEX_PATH}")

        with open(METADATA_PATH, "wb") as f:
            pickle.dump(metadata_df, f)
        logging.info(f"Metadata saved to: {METADATA_PATH}")

    except Exception as e:
        raise CustomException(e, sys)


def load_index_and_metadata():
    """
    Loads a previously saved FAISS index and metadata, so we skip
    re-downloading data and re-computing embeddings.
    """
    try:
        index = faiss.read_index(INDEX_PATH)

        with open(METADATA_PATH, "rb") as f:
            metadata_df = pickle.load(f)

        logging.info(f"Loaded existing FAISS index with {index.ntotal} vectors")

        return index, metadata_df

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import load_balanced_subset, add_fake_prices
    from src.components.embedding_generator import load_clip_model, generate_image_embeddings

    balanced_df, full_dataset = load_balanced_subset(samples_per_category=150)
    balanced_df = add_fake_prices(balanced_df)

    subset_dataset = full_dataset.select(balanced_df["orig_idx"].tolist())
    images = subset_dataset["image"]

    model, processor, device = load_clip_model()
    embeddings = generate_image_embeddings(images, model, processor, device)

    index = build_faiss_index(embeddings)

    # Keep only the metadata columns we actually need for search results
    metadata_df = balanced_df[["orig_idx", "productDisplayName", "masterCategory", "price"]].reset_index(drop=True)

    save_index_and_metadata(index, metadata_df)

    print(f"Index built and saved with {index.ntotal} vectors")
    print(metadata_df.head())