

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------
# 1. Project and dataset paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

DATASET_DIR = Path(
    r"C:\Users\LENOVO\OneDrive\Documents\myntradataset"
)

EMBEDDINGS_FILE = ARTIFACTS_DIR / "embeddings.npy"
IMAGE_PATHS_FILE = ARTIFACTS_DIR / "image_paths.csv"
METADATA_FILE = DATASET_DIR / "styles.csv"


# --------------------------------------------------
# 2. Fashion recommender class
# --------------------------------------------------

class FashionRecommender:

    def __init__(self):
        print("Loading saved embeddings...")

        self.embeddings = np.load(EMBEDDINGS_FILE)

        self.image_paths = pd.read_csv(
            IMAGE_PATHS_FILE
        )["image_path"].tolist()

        if len(self.embeddings) != len(self.image_paths):
            raise ValueError(
                "Embedding count and image-path count do not match."
            )

        print("Loading product metadata...")

        metadata = pd.read_csv(
            METADATA_FILE,
            on_bad_lines="skip"
        )

        # Standardize product IDs before matching images.
        metadata["id"] = metadata["id"].astype(str)
        metadata = metadata.drop_duplicates(subset="id")
        metadata = metadata.set_index("id")

        # Extract IDs from image filenames.
        image_ids = [
            Path(image_path).stem
            for image_path in self.image_paths
        ]

        # Align metadata with the exact order of the embeddings.
        self.metadata = (
            metadata.reindex(image_ids)
            .rename_axis("id")
            .reset_index()
        )

        # Normalize fields used for filtering.
        for column in [
            "gender",
            "articleType",
            "subCategory",
            "baseColour"
        ]:
            self.metadata[column] = (
                self.metadata[column]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
            )

        matched_count = self.metadata["articleType"].ne("").sum()

        print("Embedding shape:", self.embeddings.shape)
        print("Image paths:", len(self.image_paths))
        print("Metadata matches:", matched_count)

        # Build the visual similarity index.
        print("Building KNN index...")

        self.knn = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_jobs=-1
        )

        self.knn.fit(self.embeddings)

        print("KNN index is ready!")

    # --------------------------------------------------
    # 3. Recommend similar products
    # --------------------------------------------------

    def recommend_by_index(
        self,
        image_index,
        k=5,
        color_penalty=0.04,
        diversity_threshold=0.98
    ):
        """
        Recommend visually similar products.

        Hard filters:
            Same gender and article type.

        Soft preference:
            Matching base colours receive a small ranking advantage.
        """
        if not 0 <= image_index < len(self.embeddings):
            raise IndexError("Image index is outside the dataset.")

        if k < 1:
            raise ValueError("k must be at least 1.")

        if color_penalty < 0:
            raise ValueError("color_penalty cannot be negative.")

        query = self.metadata.iloc[image_index]

        query_gender = str(query["gender"]).strip().lower()
        query_article = str(query["articleType"]).strip().lower()
        query_colour = str(query["baseColour"]).strip().lower()

        if (
            not query_gender or query_gender == "nan"
            or not query_article or query_article == "nan"
        ):
            raise ValueError(
                "Gender or article type is missing from metadata."
            )

        # Keep products with the same gender and article type.
        eligible_mask = (
            self.metadata["gender"].astype(
                str).str.strip().str.lower().eq(query_gender)
            & self.metadata["articleType"].astype(str).str.strip().str.lower().eq(query_article)
        ).to_numpy().copy()

        # Do not recommend the query image itself.
        eligible_mask[image_index] = False
        eligible_indices = np.flatnonzero(eligible_mask)

        if len(eligible_indices) == 0:
            return pd.DataFrame(columns=[
                "image_path",
                "gender",
                "articleType",
                "baseColour",
                "distance",
                "color_match",
                "ranking_score",
                "similarity"
            ])

        candidate_embeddings = self.embeddings[eligible_indices]

        candidate_pool_size = min(
            max(k * 20, 100),
            len(eligible_indices)
        )

        candidate_knn = NearestNeighbors(
            n_neighbors=candidate_pool_size,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1
        )
        candidate_knn.fit(candidate_embeddings)

        distances, local_indices = candidate_knn.kneighbors(
            self.embeddings[image_index:image_index + 1]
        )

        results = []

        for distance, local_index in zip(distances[0], local_indices[0]):
            dataset_index = int(eligible_indices[local_index])
            product = self.metadata.iloc[dataset_index]

            product_colour = str(product["baseColour"]).strip().lower()
            color_match = (
                bool(query_colour)
                and query_colour != "nan"
                and bool(product_colour)
                and product_colour != "nan"
                and query_colour == product_colour
            )

            ranking_score = float(distance) + (
                0.0 if color_match else color_penalty
            )

            results.append({
                "image_path": self.image_paths[dataset_index],
                "gender": product["gender"],
                "articleType": product["articleType"],
                "baseColour": product["baseColour"],
                "distance": float(distance),
                "color_match": color_match,
                "ranking_score": ranking_score,
                "similarity": float(1.0 - distance)
            })

        
        # Lower ranking scores are better.
        results.sort(key=lambda result: result["ranking_score"])

        # Select recommendations while avoiding near-duplicate images.
        selected = []
        selected_indices = []

        for result in results:
            candidate_index = self.image_paths.index(result["image_path"])
            candidate_embedding = self.embeddings[candidate_index]

            is_duplicate = False

            for selected_index in selected_indices:
                selected_embedding = self.embeddings[selected_index]

                # Cosine similarity between the two product embeddings.
                denominator = (
                    np.linalg.norm(candidate_embedding)
                    * np.linalg.norm(selected_embedding)
                )

                if denominator == 0:
                    continue

                cosine_similarity = float(
                    np.dot(candidate_embedding, selected_embedding)
                    / denominator
                )

                if cosine_similarity >= diversity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                selected.append(result)
                selected_indices.append(candidate_index)

            if len(selected) == k:
                break

        return pd.DataFrame(selected)



# --------------------------------------------------
# 4. Test the recommender
# --------------------------------------------------

if __name__ == "__main__":

    recommender = FashionRecommender()

    # Test with the white women's skirt: 10000.jpg.
    query_id = "10000"

    matching_indices = [
        index
        for index, image_path in enumerate(
            recommender.image_paths
        )
        if Path(image_path).stem == query_id
    ]

    if not matching_indices:
        raise ValueError(
            f"Image {query_id}.jpg was not found."
        )

    query_index = matching_indices[0]

    query_product = recommender.metadata.iloc[query_index]

    print("\nQuery product:")

    print(
        query_product[
            [
                "id",
                "gender",
                "subCategory",
                "articleType",
                "baseColour"
            ]
        ].to_string()
    )

    recommendations = recommender.recommend_by_index(
        image_index=query_index,
        k=5,
        color_penalty=0.04
    )

    print("\nTop 5 recommendations:")

    if recommendations.empty:
        print("No matching products found.")
    else:
        print(
            recommendations.to_string(index=False)
        )
