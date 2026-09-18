import streamlit as st

from src.pipeline.agent_pipeline import run_shopping_agent
from src.components.agent_tools import search_products_tool, compare_price_to_average_tool

st.set_page_config(page_title="Visual Shopping Assistant", layout="wide")

st.title(" Visual Shopping Assistant")
st.write("Upload a product image and ask a question to get AI-powered recommendations.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
question = st.text_input("Your question", value="What's the best value option here?")
top_k = st.number_input("Number of results", min_value=3, max_value=10, value=5, step=1)

if st.button("Search") and uploaded_file is not None:
    from PIL import Image

    query_image = Image.open(uploaded_file).convert("RGB")

    st.image(query_image, caption="Your uploaded image", width=200)

    with st.spinner("Searching and analyzing..."):
        search_text, search_results = search_products_tool(query_image, top_k=top_k)
        answer = run_shopping_agent(query_image, question, top_k=top_k)

    st.subheader(" Agent's Analysis")
    st.write(answer)

    st.subheader(" Matching Products")
    cols = st.columns(len(search_results))

    for col, (_, row) in zip(cols, search_results.iterrows()):
        with col:
            st.image(row["image"], use_container_width=True)
            st.caption(f"**{row['productDisplayName']}**")
            st.caption(f"{row['price']} EGP")
            st.caption(f"Similarity: {row['similarity_score']:.2f}")