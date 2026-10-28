
from pathlib import Path
from collections import Counter

import numpy as np
import streamlit as st
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

from src.recommender import FashionRecommender
from src.feature_extractor import load_model


st.set_page_config(
    page_title="Fashion Recommender",
    page_icon="👗",
    layout="wide",
)

st.title("👗 Fashion Recommender System")
st.write(
    "Discover visually similar fashion products using "
    "ResNet50 embeddings and cosine similarity."
)


@st.cache_resource
def load_engine():
    recommender = FashionRecommender()
    model, transform = load_model()
    model.eval()
    return recommender, model, transform


with st.spinner("Loading fashion catalog and ResNet50..."):
    recommender, model, transform = load_engine()

st.success("Recommendation engine ready!")

uploaded_file = st.file_uploader(
    "Upload a clothing image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Your uploaded image")
        st.image(image, width=280)

        # Extract a 2,048-dimensional feature vector.
        image_tensor = transform(image).unsqueeze(0)

        with torch.inference_mode():
            query_embedding = (
                model(image_tensor)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        # Find the nearest catalog images.
        nearest_count = min(20, len(recommender.embeddings))

        distances, indices = recommender.knn.kneighbors(
            query_embedding,
            n_neighbors=nearest_count,
        )

        # Estimate the category from the nearest catalog matches.
        # This is a heuristic, not a trained classifier.
        category_votes = Counter()

        for index in indices[0]:
            product = recommender.metadata.iloc[int(index)]
            gender = str(product["gender"]).strip().lower()
            article = str(product["articleType"]).strip().lower()

            if gender and article:
                category_votes[(gender, article)] += 1

        if not category_votes:
            st.warning("Could not estimate a product category.")
            st.stop()

        estimated_gender, estimated_article = (
            category_votes.most_common(1)[0][0]
        )

        st.caption(
            f"Estimated category: {estimated_gender.title()} "
            f"{estimated_article.title()}"
        )

        # Restrict candidates to the estimated category.
        metadata = recommender.metadata

        eligible_mask = (
            metadata["gender"].eq(estimated_gender)
            & metadata["articleType"].eq(estimated_article)
        ).to_numpy()

        eligible_indices = np.flatnonzero(eligible_mask)

        if len(eligible_indices) == 0:
            st.warning(
                "No products were found in the estimated category. "
                "Try another image."
            )
            st.stop()

        # Compute cosine distances only for eligible products.
        candidate_embeddings = recommender.embeddings[eligible_indices]

        candidate_distances = cosine_distances(
            query_embedding,
            candidate_embeddings,
        )[0]

        # Use the nearest catalog match's colour as a soft preference.
        nearest_product = metadata.iloc[int(indices[0][0])]
        query_colour = str(
            nearest_product["baseColour"]
        ).strip().lower()

        candidates = []

        for dataset_index, distance in zip(
            eligible_indices, candidate_distances
        ):
            product = metadata.iloc[int(dataset_index)]
            colour = str(product["baseColour"]).strip().lower()

            color_match = (
                bool(query_colour)
                and query_colour != "nan"
                and bool(colour)
                and colour != "nan"
                and colour == query_colour
            )

            penalty = 0.04 if not color_match else 0.0

            candidates.append({
                "index": int(dataset_index),
                "distance": float(distance),
                "colour": colour,
                "color_match": color_match,
                "ranking_score": float(distance) + penalty,
            })

        candidates.sort(key=lambda item: item["ranking_score"])

        # Select five recommendations while avoiding near-duplicates.
        selected = []
        selected_embeddings = []
        diversity_threshold = 0.98

        for candidate in candidates:
            embedding = recommender.embeddings[candidate["index"]]

            is_duplicate = False

            if selected_embeddings:
                similarities = cosine_similarity(
                    embedding.reshape(1, -1),
                    np.asarray(selected_embeddings),
                )[0]

                if np.any(similarities >= diversity_threshold):
                    is_duplicate = True

            if not is_duplicate:
                selected.append(candidate)
                selected_embeddings.append(embedding)

            if len(selected) == 5:
                break

        st.subheader("Recommended products")

        if not selected:
            st.info("No suitable recommendations were found.")
        else:
            columns = st.columns(len(selected))

            for rank, candidate in enumerate(selected):
                product_index = candidate["index"]
                product = metadata.iloc[product_index]
                image_path = Path(
                    recommender.image_paths[product_index]
                )

                with columns[rank]:
                    if image_path.exists():
                        st.image(
                            str(image_path),
                            use_container_width=True,
                        )
                    else:
                        st.warning("Product image unavailable")

                    st.caption(f"Recommendation #{rank + 1}")
                    st.write(
                        f"**{str(product['articleType']).title()}**"
                    )
                    st.write(
                        f"Colour: {str(product['baseColour']).title()}"
                    )
                    st.metric(
                        "Cosine similarity",
                        f"{max(0.0, 1.0 - candidate['distance']):.3f}",
                    )

                    if candidate["color_match"]:
                        st.caption("✓ Colour matches nearest match")

        st.caption(
            "Recommendations are based on visual embeddings, "
            "estimated category and a soft colour preference. "
            "Similarity scores are not probabilities."
        )

    except Exception as error:
        st.error(f"Could not generate recommendations: {error}")
