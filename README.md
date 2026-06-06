# 🗞️ News Article Topic Clustering — Streamlit App

A premium, presentation-ready ML dashboard for the **News Article Topic Clustering** capstone project — TechCrush AI/ML Bootcamp, Cohort 6.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Setup & Run](#setup--run)
- [App Pages](#app-pages)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [ML Pipeline Summary](#ml-pipeline-summary)
- [Model Results](#model-results)
- [Project Structure](#project-structure)
- [Using the Real Trained Model](#using-the-real-trained-model)
- [Known Limitations](#known-limitations)

---

## Project Overview

This project implements an **unsupervised machine learning pipeline** to cluster 268,122 English news articles by topic. Two clustering algorithms — **MiniBatchKMeans** and **DBSCAN** — are trained, evaluated, and compared. The Streamlit app presents the full pipeline interactively, from raw data through preprocessing, modelling, and evaluation.

---

## Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the app

```bash
streamlit run app.py
```

### Requirements

| Package | Purpose |
|---------|---------|
| `streamlit` | App framework |
| `plotly` | Interactive charts |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models & TF-IDF |
| `nltk` | Text preprocessing |

---

## App Pages

| Page | Description |
|------|-------------|
| 🏠 Landing Page | Hero banner, key metrics, topic overview grid |
| 📖 Project Overview | Problem statement, methodology, and real-world value |
| 🔍 Dataset Explorer | Filterable article browser with search |
| 📊 EDA Dashboard | Interactive Plotly charts (timeline, cluster distribution, top sources) |
| 🤖 Modeling | Pipeline workflow, model cards, and evaluation metrics |
| 🔮 Cluster Insights | Per-cluster keyword analysis with TF-IDF bar charts |
| ⚖️ Model Comparison | KMeans vs DBSCAN side-by-side table + radar chart |
| ✨ Try It Yourself | Live text → cluster prediction with confidence score |
| 📈 Visualizations | 2D SVD scatter plot, treemap, bubble chart |
| 📋 Presentation Summary | Full summary slide for demos and presentations |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| **Frontend** | Streamlit + custom CSS (dark theme) |
| **Charts** | Plotly Express & Plotly Graph Objects |
| **Data** | Synthetic representative data (full dataset from Kaggle) |
| **Vectorisation** | TF-IDF (`max_features=5000`, `min_df=5`, `max_df=0.95`) |
| **Primary Model** | MiniBatchKMeans (`k=16`, `batch_size=1000`, `n_init=3`) |
| **Comparison Model** | DBSCAN (`eps=0.15`, `min_samples=10`, `metric=cosine`) |
| **Dimensionality Reduction** | TruncatedSVD (50 components for DBSCAN; 2D for visualisation) |
| **Text Preprocessing** | Lowercase, URL removal, lemmatisation, stopword removal (NLTK) |

---

## Dataset

**[286K Topic-Clustered News Articles](https://www.kaggle.com/datasets/jonasbecker98/286k-topic-clustered-news-articles)** — Jonas Becker (Kaggle)

| Property | Value |
|----------|-------|
| Total articles | 268,122 |
| Language | English only |
| Period | March 2016 – July 2021 |
| Format | JSON files organised by year/month |
| Original clusters | LDA (16 topics) + KMeans (17 clusters) |

### Key columns used

| Column | Description |
|--------|-------------|
| `maintext` | Full article body — primary input for clustering |
| `title` | Article headline |
| `date_publish` | Publication datetime (converted to `datetime64`) |
| `source_domain` | Publisher domain |
| `LDA_ID` | Original LDA topic label (0–15) |
| `kMeans_ID` | Original KMeans cluster label (0–16) |
| `LDA_topic_percentage` | LDA topic confidence score (mean: 0.74) |

**Dropped columns:** `date_download`, `date_modify`, `url`, `description` (scraping metadata, no clustering value)

---

## ML Pipeline Summary

### 1. Data Cleaning
- No missing values across all 15 columns
- No duplicate rows in the original load
- `date_publish` converted from `object` to `datetime64[ns, UTC]`
- 79 empty titles filled with `'Unknown Title'`
- Articles exceeding **5,000 words** (3.17% = 8,492 articles) removed as scraping outliers

### 2. Text Preprocessing
- Lowercasing, URL and punctuation removal
- Tokenisation, stopword removal (NLTK English), lemmatisation (WordNetLemmatizer)
- Processed in batches of 5,000 for memory efficiency

### 3. Vectorisation
- **TF-IDF** with 5,000 features → sparse matrix of shape `(41,080 × 5,000)` (~33 MB)

### 4. Modelling

**MiniBatchKMeans** — trained on full 41,080-article set
- `n_clusters=16` (matches original LDA topic count)
- `batch_size=1000`, `n_init=3`, `max_iter=100`, `random_state=42`

**DBSCAN** — trained on 30,000-article sample (memory constraint)
- Dimensionality reduced via TruncatedSVD (50 components) + L2 normalisation
- `eps=0.15` selected via k-distance elbow graph
- `min_samples=10`, `metric=cosine`

---

## Model Results

| Metric | KMeans | DBSCAN |
|--------|--------|--------|
| Number of clusters | 16 | 4 |
| Silhouette Score (↑ better) | **0.0611** | -0.1597 |
| Davies-Bouldin Score (↓ better) | 5.8577 | **1.4660** |
| Inertia | 36,045.84 | N/A |
| Noise points | 0 | 1,895 (6.32%) |
| Dataset coverage | **100%** | 93.68% |
| **Recommended for this task** | ✅ **YES** | ❌ NO |

### KMeans — Top Keywords Per Cluster (sample)

| Cluster | Top Keywords |
|---------|-------------|
| 0 | museum, art, city, building, artist, york |
| 2 | school, student, teacher, education, college |
| 4 | music, song, album, band |
| 5 | trump, president, republican, clinton |
| 8 | obama, president, bush, biden, white house |
| 9 | sauce, recipe, dish, pepper, cook |
| 12 | league, game, club, player, season, goal |

---

## Project Structure

```
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── tfidf_vectorizer.pkl    # (Optional) Saved TF-IDF vectorizer
├── kmeans_model.pkl        # (Optional) Saved KMeans model
└── News_Article_Topic_Cluttering_Capstone_Project.ipynb  # Training notebook
```

---

## Using the Real Trained Model

The live prediction page currently uses keyword matching as a proxy. To wire in the actual trained model, save it from the notebook and load it in the app:

**Save in the notebook:**
```python
import joblib
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(kmeans_model, "kmeans_model.pkl")
```

**Load in `app.py`:**
```python
import joblib
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("kmeans_model.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split()
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

cleaned = clean_text(user_input)
vec = vectorizer.transform([cleaned])
cluster = model.predict(vec)[0]
```

---

## Known Limitations

- **DBSCAN collapses to ~4 clusters** on this dataset due to high text density in TF-IDF space. It works best on more geometrically separable data.
- **Source bias**: a small number of publishers (e.g. CBS Local, Fox News) dominate the dataset, meaning some clusters may reflect editorial patterns rather than pure topic signals.
- **Live prediction** uses keyword heuristics by default, not the real TF-IDF + KMeans model (see above to upgrade).
- **Word count outliers**: the maximum article length in the raw data is 3,074,908 words — almost certainly a scraping error. These are filtered at >5,000 words during preprocessing.

---

*TechCrush AI/ML Bootcamp — Cohort 6 | Dataset: Jonas Becker (Kaggle)*
