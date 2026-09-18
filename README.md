# 🛍️ Visual Shopping Assistant

An AI-powered visual product search system with an intelligent shopping
assistant agent. Upload a product image, and the system finds visually
similar products from a catalog, then uses an LLM agent to analyze
pricing and recommend the best value option.

## 🎯 What it does

1. **Visual Search**: Upload any product image, and CLIP embeddings +
   FAISS find the most visually similar products in the catalog.
2. **AI Agent Analysis**: A Groq-powered LLM agent analyzes the results,
   comparing prices against category averages and reasoning about which
   option offers the best value — not just the cheapest.

## 🏗️ Architecture

Image → CLIP Embedding → FAISS Search → Top-K Products
↓
Price Comparison + LLM Agent Reasoning
↓
Final Recommendation



## 🛠️ Tech Stack

- **CV/Embeddings**: CLIP (openai/clip-vit-base-patch32) via Hugging Face Transformers
- **Similarity Search**: FAISS (exact search, Inner Product)
- **LLM Agent**: Groq API (Llama-based models)
- **Backend API**: FastAPI
- **Frontend**: Streamlit
- **Containerization**: Docker
- **Testing**: pytest
- **CI**: GitHub Actions

## 📸 Demo

[Add a screenshot or GIF here]

**Live app**: [link once deployed]

## 🚀 Running Locally

### Option 1: With Docker (recommended)

```bash
docker build -t visual-shopping-assistant .
docker run -p 8501:8501 --env-file .env visual-shopping-assistant
```

### Option 2: Without Docker

```bash
python -m venv venv
venv\Scripts\activate  # Windows

pip install -r requirements.txt

# Create a .env file with:
# GROQ_API_KEY=your_key_here

python -m src.components.index_builder  # builds the catalog index (first time only)
streamlit run streamlit_app.py
```

Then open `http://localhost:8501`

## 🧪 Running Tests

```bash
pytest tests/ -v
```

## 📊 Dataset

Uses a balanced subset of the [Fashion Product Images dataset](https://huggingface.co/datasets/ashraq/fashion-product-images-small).
Prices are synthetically generated (category-aware ranges) since the
original dataset doesn't include pricing data.

## ⚠️ Design Decisions & Trade-offs

- **Custom catalog instead of live scraping**: Real e-commerce sites
  restrict scraping in their ToS. The architecture is designed to plug
  into a real product API/database with minimal changes.
- **Exact search (not approximate)**: With a catalog this size, FAISS
  exact search (IndexFlatIP) is fast enough and more accurate than
  approximate methods like HNSW.
- **RAG-style prompting instead of full function calling**: Since the
  search + price comparison always run together as one flow, a full
  agentic function-calling loop wasn't necessary — the LLM reasons over
  pre-fetched results instead.