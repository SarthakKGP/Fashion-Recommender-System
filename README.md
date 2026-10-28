# Fashion Recommender System

A content-based fashion recommendation system that retrieves visually similar clothing products using deep learning embeddings, cosine similarity, and category-aware ranking. The project includes an interactive Streamlit application for image-based recommendations.

## Features

- **Deep Feature Extraction:** Uses pretrained ResNet50 to extract 2,048-dimensional feature embeddings from fashion images.
- **Visual Similarity Search:** Uses cosine distance and Scikit-learn's nearest-neighbor search to retrieve similar products.
- **Category-Aware Recommendations:** Filters catalog products by gender and article type for dataset-based queries; estimates categories for uploaded images using nearest catalog matches.
- **Colour-Aware Ranking:** Applies a soft ranking preference for matching colours.
- **Recommendation Diversity:** Reduces near-duplicate results using embedding similarity.
- **Interactive Web App:** Upload a clothing image and explore recommended catalog products through Streamlit.

## Tech Stack

Python · PyTorch · Torchvision · ResNet50 · NumPy · Pandas · Scikit-learn · Streamlit · Pillow

## Dataset and Embeddings

The project was developed using the Myntra Fashion Product Images dataset, containing approximately 44,441 processed images.

ResNet50 embeddings are stored locally as NumPy arrays. The dataset and generated artifacts are not included in this repository because of their size and dataset distribution considerations.

## Project Structure

```text
Fashion-Recommender-System/
├── app.py
├── requirements.txt
├── .gitignore
├── artifacts/                 # Generated locally; not committed
│   ├── embeddings.npy
│   └── image_paths.csv
└── src/
    ├── feature_extractor.py
    ├── inspect_dataset.py
    ├── recommender.py
    └── view_recommendations.py
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/SarthakKGP/Fashion-Recommender-System.git
cd Fashion-Recommender-System
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the dataset

Obtain the dataset separately and place it locally. Update the dataset path in the source files to match your machine.

The current implementation expects the metadata file `styles.csv` and the corresponding clothing images.

### 5. Generate the embeddings

Run the feature extraction module:

```bash
python -m src.feature_extractor
```

Feature extraction can take considerable time on a CPU. Confirm that the expected files, including `artifacts/embeddings.npy` and `artifacts/image_paths.csv`, have been generated before launching the app.

### 6. Launch the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, upload a clothing image, and explore the recommendations.

## How It Works

1. **Preprocessing:** Images are transformed using the pretrained ResNet50 model's expected input preprocessing.
2. **Embedding Extraction:** The final classification layer is removed, producing a 2,048-dimensional feature vector per image.
3. **Catalog Indexing:** Feature vectors are indexed using nearest-neighbor search with cosine distance.
4. **Category Estimation:** For uploaded images, nearby catalog matches provide a heuristic estimate of the likely gender and article type.
5. **Ranking:** Candidate products are ranked by visual distance, with a small colour mismatch penalty.
6. **Diversity Filtering:** Highly similar candidate embeddings are suppressed to encourage variety in the final results.

## Limitations

- Category estimation for uploaded images is a nearest-neighbor heuristic, not a separately trained clothing classifier.
- Visual similarity does not guarantee semantic equivalence or user preference.
- Similarity scores are not probabilities.
- Recommendation quality depends on the catalog, image preprocessing, and the selected ranking parameters.
- Initial setup requires obtaining the dataset and generating the local embeddings.

## Future Improvements

- Evaluate recommendation quality using Precision@K and Recall