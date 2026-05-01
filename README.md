# Data Science MSc — Coursework Portfolio

A collection of end-to-end machine learning projects completed during my Master's in Data Science. Each project covers a full pipeline: preprocessing, modeling, evaluation, and interpretation — with an emphasis on comparing classical and modern approaches on the same task.

> **Note:** I'm currently working on my thesis on an LLM/AI workflow. This repo collects the coursework that came before it.

---

## 📂 Projects

| # | Project | Domain | Core Techniques |
|---|---------|--------|-----------------|
| 1 | [Amazon Reviews — Sentiment Analysis](#1-amazon-reviews--sentiment-analysis) | NLP | Word2Vec, GloVe, DistilBERT (frozen + fine-tuned), LR, XGBoost, K-Means topic modeling |
| 2 | [Sleep Health & Lifestyle — Classification, Clustering & Association Rules](#2-sleep-health--lifestyle) | Health analytics | Decision Tree, kNN, Naive Bayes, Random Forest, AdaBoost, K-Means, Agglomerative, DBSCAN, Apriori (from scratch) |

---

## 1. Amazon Reviews — Sentiment Analysis

**File:** `amazon_reviews_final_assignement.py`

### Goal
Binary sentiment classification (positive / negative) on Amazon Unlocked Mobile reviews, comparing **classical NLP** with **transformer-based** approaches under the same evaluation protocol.

### Dataset
- **Amazon Unlocked Mobile** reviews
- Neutral reviews (rating = 3) removed
- Ratings > 3 → positive (1); ratings < 3 → negative (0)
- 30% stratified subset used for computational efficiency while preserving class balance

### Pipeline
- **Two preprocessing tracks:**
  - *Classical models:* lowercasing, URL/special-char removal, tokenization, stopword removal, Porter stemming
  - *BERT:* minimal cleaning (lowercase + URL removal) — preserves syntactic structure
- Stratified train / validation / test split shared across all models
- Empty-review filtering for token-based models to avoid invalid embeddings

### Models Compared
| Embedding | Classifier | Notes |
|-----------|------------|-------|
| Word2Vec (trained from scratch) | Logistic Regression / XGBoost | 100-dim, mean-pooled |
| GloVe (`glove-wiki-gigaword-200`) | Logistic Regression / XGBoost | Pretrained, mean-pooled |
| DistilBERT (frozen) — CLS token | Logistic Regression / XGBoost | Feature extraction only |
| DistilBERT (frozen) — mean pooling | Logistic Regression / XGBoost | Attention-mask aware |
| DistilBERT — fine-tuned end-to-end | Built-in classification head | 3 epochs, early stopping |

### Key Results (F1-score, sorted)
1. **Fine-tuned DistilBERT** — best overall (~0.979)
2. BERT mean-pooled + XGBoost (~0.966)
3. BERT CLS + XGBoost / LR
4. GloVe + XGBoost
5. Word2Vec + LR (baseline)

### Findings
- **Transformers > static embeddings** — context and polysemy handling matter
- **Fine-tuning > feature extraction**, but frozen BERT embeddings are a strong, cheaper alternative
- **XGBoost consistently beats Logistic Regression** across every embedding type
- Hand-crafted hard cases (negation, mixed sentiment, sarcasm) are handled correctly only by fine-tuned BERT

### Beyond Classification
- **EDA:** class distribution, LR coefficient distribution, word clouds for positive vs negative reviews
- **Brand-level sentiment** analysis across the top mobile manufacturers
- **K-Means topic modeling** on a hardware/software keyword whitelist (battery, screen, camera, charger, signal, etc.) with cluster centroid heatmaps and per-cluster brand/rating profiling

---

## 2. Sleep Health & Lifestyle

**File:** `sleep_mbads_final.py`

### Goal
Predict and characterize sleep disorders (`None`, `Insomnia`, `Sleep Apnea`) from lifestyle and health indicators using **supervised**, **unsupervised**, and **association-rule** learning.

### Dataset
- **Sleep Health and Lifestyle Dataset**
- Demographics, occupation, sleep metrics, physical activity, stress, heart rate, blood pressure, BMI

### Feature Engineering
- Split blood pressure → `Systolic` / `Diastolic`
- Derived `Pulse Pressure = Systolic − Diastolic`
- Derived `Steps per Activity = Daily Steps / Physical Activity Level` (with median imputation for zeros)
- Consolidated BMI categories (`Normal Weight` → `Normal`)
- One-hot encoding for categoricals; standardization for distance-based models

### A. Classification
Five classifiers benchmarked on the same stratified split:

- Decision Tree (entropy / C4.5-style)
- k-Nearest Neighbors (k=5, scaled features)
- Gaussian Naive Bayes
- Random Forest (200 trees)
- AdaBoost (200 estimators)

Evaluated with accuracy, classification report, and confusion matrices. Random Forest feature importances visualized for interpretation.

### B. Clustering
Three approaches on standardized numeric features:

- **K-Means** with elbow method → k=5; cluster profiles interpreted into descriptive personas (e.g., *"Younger, stressed, poor sleep"*, *"Middle-aged, very rested, sedentary"*); PCA 2D visualization
- **Agglomerative (Ward linkage)** with truncated dendrogram → 4 clusters
- **DBSCAN** with k-distance graph for ε selection; noise points reported separately

Cross-tabulation of cluster membership against actual sleep disorder labels.

### C. Association Rule Mining (Apriori — from scratch)
- Implemented **Apriori** without `mlxtend` — custom candidate generation, support counting, confidence and lift calculation
- Numeric features discretized into Low / Medium / High via `qcut`
- `min_support = 0.10`, `min_confidence = 0.60`
- Filtered rules whose consequent contains `Insomnia` or `Sleep Apnea` and `lift > 1.2` to surface clinically meaningful patterns

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Core:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- **NLP:** `nltk`, `gensim`, `transformers`, `tensorflow`, `datasets`, `wordcloud`
- **Boosting:** `xgboost`
- **Notebook origin:** Google Colab (originally `.ipynb`, exported to `.py`)

---

## 🚀 How to Run

Both scripts were originally Colab notebooks and reference Google Drive paths. To run locally:

1. Clone the repo
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. Create an environment and install dependencies
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn nltk gensim wordcloud xgboost
   pip install "transformers>=4.40,<5" datasets tensorflow
   ```
3. Update the dataset paths near the top of each script (replace the `/content/drive/MyDrive/...` paths with your local paths).
4. Run:
   ```bash
   python amazon_reviews_final_assignement.py
   python sleep_mbads_final.py
   ```

> Datasets are not included in this repo. Sources: Amazon Unlocked Mobile (Kaggle) and Sleep Health and Lifestyle Dataset (Kaggle).

---

## 📁 Repository Structure

```
.
├── amazon_reviews_final_assignement.py    # NLP project — sentiment analysis
├── sleep_mbads_final.py                   # Health analytics project — classification, clustering, ARM
└── README.md
```

---

## 👤 Author

**Eleanna Chatziathanasiadou**
MSc Data Science and Business Intelligence, *University of Macedonia, Greece*
Currently writing my thesis on an LLM/AI workflow.

- 🔗 LinkedIn: [(https://www.linkedin.com/in/eleanna-c-951662213/)]

*This repository is for academic and portfolio purposes.*
