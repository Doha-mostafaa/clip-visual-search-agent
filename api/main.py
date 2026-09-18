import sys
import io

from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image

from src.pipeline.agent_pipeline import run_shopping_agent
from src.logger import logging
from src.exception import CustomException

app = FastAPI(title="Visual Shopping Assistant API")


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/search-and-analyze")
async def search_and_analyze(
    image: UploadFile = File(...),
    question: str = Form(...),
    top_k: int = Form(5),
):
    try:
        logging.info(f"Received request with question: {question}")

        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        answer = run_shopping_agent(pil_image, question, top_k=top_k)

        return {
            "question": question,
            "answer": answer,
        }

    except Exception as e:
        raise CustomException(e, sys)