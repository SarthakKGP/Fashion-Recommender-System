
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from torchvision.models import resnet50, ResNet50_Weights


# Dataset location
IMAGE_DIR = Path(
    r"C:\Users\LENOVO\OneDrive\Documents\myntradataset\images"
)

# Project output locations
PROJECT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"

# Processing settings
MAX_IMAGES = None       # None means process all discovered JPG images
BATCH_SIZE = 16
CHECKPOINT_EVERY = 500

DEVICE = torch.device("cpu")


def load_model():
    """Load pretrained ResNet50 as a feature extractor."""
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    # Replace the 1000-class classifier with an identity layer.
    model.fc = nn.Identity()

    model = model.to(DEVICE)
    model.eval()

    return model, weights.transforms()


def load_image(image_path, transform):
    """Read an image and prepare it for ResNet50."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        return transform(image)


def extract_batch(batch_tensors, model):
    """Return one 2048-dimensional vector per image."""
    batch = torch.stack(batch_tensors).to(DEVICE)

    with torch.inference_mode():
        embeddings = model(batch)

    return embeddings.cpu().numpy().astype(np.float32)


def save_checkpoint(embeddings, successful_paths, failed_images):
    """Save current progress so extraction can resume later."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(
        CHECKPOINT_DIR / "embeddings_checkpoint.npy",
        np.asarray(embeddings, dtype=np.float32),
    )

    pd.DataFrame({
        "image_path": [str(path) for path in successful_paths]
    }).to_csv(
        CHECKPOINT_DIR / "image_paths_checkpoint.csv",
        index=False,
    )

    pd.DataFrame(
        failed_images,
        columns=["image", "error"],
    ).to_csv(
        CHECKPOINT_DIR / "failed_images_checkpoint.csv",
        index=False,
    )


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Use the same deterministic image ordering on every run.
    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError(
            f"No JPG images found in {IMAGE_DIR}"
        )

    if MAX_IMAGES is not None:
        image_paths = image_paths[:MAX_IMAGES]

    checkpoint_embeddings = (
        CHECKPOINT_DIR / "embeddings_checkpoint.npy"
    )
    checkpoint_paths = (
        CHECKPOINT_DIR / "image_paths_checkpoint.csv"
    )
    checkpoint_failures = (
        CHECKPOINT_DIR / "failed_images_checkpoint.csv"
    )

    embeddings = []
    successful_paths = []
    failed_images = []

    # Resume from the previous checkpoint, if all required files exist.
    checkpoint_exists = (
        checkpoint_embeddings.exists()
        and checkpoint_paths.exists()
        and checkpoint_failures.exists()
    )

    if checkpoint_exists:
        print("Checkpoint found. Checking whether it can be resumed...")

        saved_embeddings = np.load(checkpoint_embeddings)
        saved_paths_df = pd.read_csv(checkpoint_paths)
        saved_failures_df = pd.read_csv(checkpoint_failures)

        saved_paths = saved_paths_df["image_path"].astype(str).tolist()

        current_paths = [str(path) for path in image_paths]

        # Resume only when the checkpoint matches the beginning of
        # the current, sorted image list.
        if (
            len(saved_paths) <= len(current_paths)
            and current_paths[:len(saved_paths)] == saved_paths
            and saved_embeddings.ndim == 2
            and saved_embeddings.shape == (len(saved_paths), 2048)
        ):
            embeddings = list(saved_embeddings)
            successful_paths = saved_paths
            failed_images = saved_failures_df.to_dict("records")

            # Skip images already handled successfully or unsuccessfully.
            handled = set(successful_paths)
            handled.update(
                str(IMAGE_DIR / item["image"])
                for item in failed_images
            )
            image_paths = [
                path for path in image_paths
                if str(path) not in handled
            ]

            print(
                f"Resuming with {len(successful_paths)} successful "
                f"images already saved."
            )
        else:
            print(
                "Checkpoint does not match this dataset ordering. "
                "Starting a fresh extraction."
            )

    model, transform = load_model()

    batch_tensors = []
    batch_paths = []
    newly_processed = 0
    start_time = time.time()

    progress = tqdm(image_paths, desc="Extracting embeddings")

    for image_path in progress:
        try:
            tensor = load_image(image_path, transform)
            batch_tensors.append(tensor)
            batch_paths.append(image_path)

        except (OSError, ValueError, UnidentifiedImageError) as error:
            failed_images.append({
                "image": image_path.name,
                "error": str(error),
            })
            newly_processed += 1

        if len(batch_tensors) >= BATCH_SIZE:
            batch_embeddings = extract_batch(batch_tensors, model)

            embeddings.extend(batch_embeddings)
            successful_paths.extend(str(path) for path in batch_paths)

            newly_processed += len(batch_paths)
            batch_tensors = []
            batch_paths = []

        # Save only when no unprocessed tensors remain in memory.
        if newly_processed >= CHECKPOINT_EVERY and not batch_tensors:
            save_checkpoint(
                embeddings, successful_paths, failed_images
            )
            newly_processed = 0
            progress.set_postfix(saved=len(successful_paths))

    # Process the final partial batch.
    if batch_tensors:
        batch_embeddings = extract_batch(batch_tensors, model)
        embeddings.extend(batch_embeddings)
        successful_paths.extend(str(path) for path in batch_paths)

    # Save final checkpoint and production files.
    save_checkpoint(embeddings, successful_paths, failed_images)

    embedding_matrix = np.asarray(embeddings, dtype=np.float32)

    np.save(ARTIFACTS_DIR / "embeddings.npy", embedding_matrix)

    pd.DataFrame({
        "image_path": successful_paths
    }).to_csv(
        ARTIFACTS_DIR / "image_paths.csv",
        index=False,
    )

    pd.DataFrame(
        failed_images,
        columns=["image", "error"],
    ).to_csv(
        ARTIFACTS_DIR / "failed_images.csv",
        index=False,
    )

    elapsed = time.time() - start_time

    print("\n--- Full Extraction Summary ---")
    print("Successful images:", len(successful_paths))
    print("Failed images:", len(failed_images))
    print("Embedding matrix shape:", embedding_matrix.shape)
    print(f"Time in this run: {elapsed / 60:.2f} minutes")
    print(f"Final embeddings: {ARTIFACTS_DIR / 'embeddings.npy'}")
    print(f"Image paths: {ARTIFACTS_DIR / 'image_paths.csv'}")


if __name__ == "__main__":
    main()
 