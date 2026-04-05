# NLP Assignment – Twitter Sentiment Analysis

**Author:** Shreya Newaskar
**PRN:** 202301040240

---

## Overview

This project performs **binary sentiment analysis** on the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140), which contains 1.6 million tweets labeled as **positive** or **negative**. The goal is to preprocess raw tweet text using NLP techniques and train machine learning classifiers to predict tweet sentiment.

---

## Dataset

| File | Description |
|------|-------------|
| `training.1600000.processed.noemoticon.csv` | Sentiment140 dataset with 1.6M tweets |

**Columns used:**
- `target` – Sentiment label (`0` = Negative, `4` → remapped to `1` = Positive)
- `text` – Raw tweet content

---

## Project Structure

```
NLP_Assignment/
├── dataset/
│   └── archive (6)/
│       └── training.1600000.processed.noemoticon.csv
├── Code_nlp.ipynb     # Main Jupyter Notebook
├── requirements.txt   # Python dependencies
└── README.md
```

---

## Pipeline

### 1. Data Loading
- Loaded the CSV with `latin-1` encoding
- Retained only `target` and `text` columns
- Dropped rows with missing values (`dropna`)
- Remapped label `4` → `1` for binary classification

### 2. Dataset Reduction *(moved before preprocessing for efficiency)*
- Sampled **50,000 tweets** (random state = 42) **before** running the NLP pipeline
- This avoids preprocessing all 1.6M rows, significantly reducing runtime

### 3. NLP Preprocessing

| Step | Description |
|------|-------------|
| **Tokenization** | Split text into tokens using NLTK `word_tokenize` (lowercased) |
| **Stopword Removal** | Removed English stopwords using NLTK |
| **Stemming** | Applied Porter Stemmer to reduce words to their root form |
| **Lemmatization** | Applied WordNet Lemmatizer on non-stemmed tokens (independent step) |
| **Clean Text** | Combined pipeline: lowercasing → URL removal → mention removal → non-alpha removal → stopword removal → short token filter (`len > 2`) → stemming + lemmatization |

### 4. Train-Test Split
- **80% train / 20% test** (random state = 42)

### 5. Feature Extraction

| Vectorizer | Config |
|------------|--------|
| **CountVectorizer** | `max_features=5000`, `ngram_range=(1,2)` |
| **TF-IDF Vectorizer** | `max_features=5000`, `ngram_range=(1,2)` |

### 6. Model Training

| Model | Feature Input |
|-------|--------------|
| **Logistic Regression** (`max_iter=200`) | TF-IDF |
| **Multinomial Naive Bayes** | CountVectorizer |

---

## Results

| Model | Accuracy | Precision (avg) | Recall (avg) | F1 (avg) |
|-------|----------|-----------------|--------------|----------|
| Logistic Regression | **75.17%** | 0.75 | 0.75 | 0.75 |
| Naive Bayes | 74.70% | 0.75 | 0.75 | 0.75 |

### Key Observations
1. **Logistic Regression** slightly outperforms Naive Bayes (~0.5% higher accuracy).
2. Logistic Regression captures feature relationships, while Naive Bayes assumes word independence.
3. Both models show **balanced precision/recall** across classes — no class bias.
4. ~75% accuracy is reasonable given the noisy nature of Twitter data (slang, abbreviations, informal language).
5. NLP preprocessing (stopword removal, stemming, lemmatization) significantly improved model performance.

---

## Visualizations

- **Class Distribution** – Bar chart of positive vs. negative tweet counts (with colors)
- **Confusion Matrix** – Heatmap with true/predicted axis labels for Logistic Regression
- **Model Accuracy Comparison** – Bar chart with value labels and y-axis limits
- **Model Performance Comparison** – Line chart of Precision, Recall, F1 for both models
- **Top 20 Words** – Bar chart of the most frequent words in cleaned tweets (with `tight_layout`)

---

## Improvements Made

| Area | Change |
|------|--------|
| Data loading | Added `dropna()` to remove rows with missing text or labels |
| Sampling order | Moved sampling to **before** preprocessing (saves ~30x processing time) |
| NLTK downloads | Consolidated all downloads into a single cell at the start |
| Lemmatization | Fixed standalone `lemmatized` column to use `no_stopwords` (not stemmed tokens) |
| Clean text | Added `len(w) > 2` filter to remove very short noise tokens |
| Confusion matrix | Added `xticklabels`, `yticklabels`, axis titles, and `cmap='Blues'` |
| Accuracy bar chart | Added y-axis limits (`0.5–1.0`) and numeric value labels on bars |
| All charts | Added `tight_layout()` to prevent label cutoff |

---

## Requirements

Install all dependencies at once:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
```

Download NLTK resources (run once inside the notebook):

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## How to Run

1. Open `Code_nlp.ipynb` in **Google Colab** or **Jupyter Notebook**.
2. Upload the dataset CSV when prompted (Colab) or place it in the working directory.
3. Run all cells sequentially.
