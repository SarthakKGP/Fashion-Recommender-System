from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.recommender import FashionRecommender
from src.feature_extractor import load_model


st.set_page_config(
    page_title="Fashion Recommender",
    page_icon="👗",
    layout="wide",
)

st.title("👗 Fashion Recommender System")
st.write(
    "Upload a clothing image to discover visually similar "
    "products from our fashion catalog."
)


@st.cache_resource
def load_engine():
    recommender = FashionRecommender()
    model, transform = load_model()
    return recommender, model, transform


with st.spinner("Loading ResNet50 and fashion catalog..."):
    recommender, model, transform = load_engine()

st.success("Recommendation engine ready!")

uploaded_file = st.file_uploader(
    "Upload a clothing image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 2])

        with left:
            st.subheader("Your image")
            st.image(image, use_container_width=True)

        # Apply the exact preprocessing used for the dataset.
        image_tensor = transform(image).unsqueeze(0)

        # Extract the uploaded image's 2048-dimensional embedding.
        with torch.inference_mode():
            query_embedding = (
                model(image_tensor).cpu().numpy().astype(np.float32)
            )

        # Find the five nearest products in the saved catalog.
        distances, indices = recommender.knn.kneighbors(
            query_embedding,
            n_neighbors=5,
        )

        st.subheader("Recommended products")

        columns = st.columns(5)

        for rank, (distance, index) in enumerate(
            zip(distances[0], indices[0]),
            start=1,
        ):
            product = recommender.metadata.iloc[int(index)]
            image_path = Path(recommender.image_paths[int(index)])

            with columns[rank - 1]:
                if image_path.exists():
                    st.image(
                        str(image_path),
                        use_container_width=True,
                    )
                else:
                    st.warning("Product image unavailable")

                st.caption(f"Recommendation #{rank}")
                st.write(
                    f"**{product.get('articleType', 'Product').title()}**"
                )
                st.write(
                    f"Colour: {product.get('baseColour', 'Unknown').title()}"
                )
                st.metric(
                    "Cosine similarity score",
                    f"{max(0.0, 1.0 - float(distance)):.3f}",
                )

    except Exception as error:
        st.error(f"Could not generate recommendations: {error}")
        st.exception(error)
