
from pathlib import Path
import pandas as pd

# Dataset location (we are not copying the images)
DATASET_DIR = Path(
    r"C:\Users\LENOVO\OneDrive\Documents\myntradataset"
)

IMAGE_DIR = DATASET_DIR / "images"
METADATA_FILE = DATASET_DIR / "styles.csv"


def main():
    # Load product metadata
    df = pd.read_csv(METADATA_FILE, on_bad_lines="warn")

    print("Metadata shape:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())

    # Check for duplicate product IDs
    print("\nDuplicate IDs:", df["id"].duplicated().sum())

    # Check how many metadata IDs have corresponding images
    image_ids = {
        path.stem
        for path in IMAGE_DIR.glob("*.jpg")
    }

    metadata_ids = set(df["id"].astype(str))
    matching_ids = image_ids & metadata_ids

    print("\nImage files:", len(image_ids))
    print("Metadata records:", len(metadata_ids))
    print("Matching image IDs:", len(matching_ids))
    print("Metadata records without images:",
          len(metadata_ids - image_ids))

    print("\nMost common product categories:")
    print(df["articleType"].value_counts().head(10))


if __name__ == "__main__":
    main()
