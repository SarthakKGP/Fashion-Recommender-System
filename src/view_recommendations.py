
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from src.recommender import FashionRecommender


def main():
    recommender = FashionRecommender()

    # Find the product with image filename 10000.jpg.
    query_matches = [
        i for i, path in enumerate(recommender.image_paths)
        if Path(path).name == "10000.jpg"
    ]

    if not query_matches:
        print("Image 10000.jpg was not found in the catalog.")
        return

    query_index = query_matches[0]

    recommendations = recommender.recommend_by_index(
        query_index,
        k=5
    )

    query_path = Path(recommender.image_paths[query_index])

    print("\nQuery image:", query_path.name)
    print("\nRecommended products:")
    print(recommendations[
        ["image_path", "articleType", "baseColour", "distance"]
    ].to_string(index=False))

    # Display the query image and recommended products.
    image_paths = [query_path] + [
        Path(path) for path in recommendations["image_path"]
    ]

    fig, axes = plt.subplots(
        1, len(image_paths),
        figsize=(4 * len(image_paths), 5)
    )

    if len(image_paths) == 1:
        axes = [axes]

    for i, (ax, path) in enumerate(zip(axes, image_paths)):
        if path.exists():
            with Image.open(path) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title("Query image" if i == 0 else f"Recommendation {i}")
        else:
            ax.set_title("Image not found")

        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
