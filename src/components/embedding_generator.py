import sys
import torch
import numpy as np
from transformers import CLIPModel, CLIPProcessor

from src.logger import logging
from src.exception import CustomException


MODEL_NAME = "openai/clip-vit-base-patch32"


def load_clip_model():
    """
    Loads the pretrained CLIP model and its processor.
    """
    try:
        logging.info(f"Loading CLIP model: {MODEL_NAME}")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)

        logging.info(f"CLIP model loaded successfully on device: {device}")

        return model, processor, device

    except Exception as e:
        raise CustomException(e, sys)


def generate_image_embeddings(images, model, processor, device, batch_size: int = 32):
    """
    Takes a list of PIL images and returns a numpy array of embeddings,
    one row per image, processed in batches for efficiency.
    """
    try:
        logging.info(f"Generating embeddings for {len(images)} images...")

        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            inputs = processor(images=batch, return_tensors="pt").to(device)

            with torch.no_grad():
                image_features = model.get_image_features(**inputs)

            # Normalize embeddings so cosine similarity works correctly later
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            all_embeddings.append(image_features.cpu().numpy())

            logging.info(f"Processed batch {i // batch_size + 1} "
                         f"({min(i + batch_size, len(images))}/{len(images)} images)")

        embeddings_array = np.vstack(all_embeddings)

        logging.info(f"Finished generating embeddings, final shape: {embeddings_array.shape}")

        return embeddings_array

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import load_balanced_subset, add_fake_prices

    balanced_df, _ = load_balanced_subset(samples_per_category=20)
    balanced_df = add_fake_prices(balanced_df)

    subset_dataset = _.select(balanced_df["orig_idx"].tolist())
    images = subset_dataset["image"]

    model, processor, device = load_clip_model()
    embeddings = generate_image_embeddings(images, model, processor, device)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"First embedding (first 5 values): {embeddings[0][:5]}")