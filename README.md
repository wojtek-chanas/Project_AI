# Sentiment Analysis System
## Customer Review Sentiment Analysis Tool
**Course:** DLBDSEAIS02 — Project: Artificial Intelligence  
**Task:** Task 2 — Sentiment Analysis of Customer Reviews

---

## Overview

An end-to-end sentiment analysis system that classifies customer product reviews into three sentiment classes — **negative**, **neutral**, and **positive** — using a pretrained HuggingFace BERT model.

### System Components

| Component | File | Description |
|-----------|------|-------------|
| NLP Model | HuggingFace | `nlptown/bert-base-multilingual-uncased-sentiment` — pretrained BERT, no local training required |
| REST API | `api.py` | FastAPI server exposing prediction and monitoring endpoints |
| Database | `reviews.db` | SQLite database storing predictions and human-verified labels |
| Dashboard | `dashboard.py` | Streamlit web interface for analysts |

---

## Installation

### Prerequisites
- Python 3.10 or higher
- Anaconda (recommended) or any Python environment

### Install dependencies
```bash
pip install transformers fastapi uvicorn streamlit sqlalchemy pandas requests tqdm torch
```

> **Note:** `torch` is approximately 2 GB. The HuggingFace model (~700 MB) is downloaded automatically on first run and cached locally.

---

## Running the System

The system requires **two terminal windows** running simultaneously.

### Step 1 — Navigate to the project directory
Open both terminals and run:
```bash
cd "C:\Users\Gebruiker\Project AI"
```

### Step 2 — Start the API server (Terminal 1)
```bash
uvicorn api:app --port 8000
```

Wait until you see:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

> The first startup takes 30–60 seconds while the model loads into memory. Subsequent startups are faster as the model is cached.

### Step 3 — Start the dashboard (Terminal 2)
```bash
streamlit run dashboard.py
```

The dashboard opens automatically at **http://localhost:8501**

---

## API Reference

Interactive API documentation (Swagger UI) is available at **http://localhost:8000/docs** when the server is running.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Classify sentiment of a review text |
| `/correct` | POST | Submit a human-verified label for a stored prediction |
| `/reviews` | GET | Retrieve all stored reviews |
| `/quality` | GET | Check model accuracy against verified labels |
| `/export-training-data` | GET | Export verified reviews to CSV for fine-tuning |
| `/health` | GET | API health check |

### Example: Classify a review
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely fantastic!"}'
```

Response:
```json
{
  "text": "This product is absolutely fantastic!",
  "sentiment": "positive",
  "stars": "5 stars",
  "confidence": 0.9471
}
```

---

## Sentiment Classes

| Stars | Label | Class |
|-------|-------|-------|
| ⭐ 1–2 | negative | 1-2 star ratings |
| ⭐⭐⭐ 3 | neutral | 3 star ratings |
| ⭐⭐⭐⭐⭐ 4–5 | positive | 4-5 star ratings |

---

## Model Quality Monitoring

The system continuously monitors prediction quality against human-verified labels.

- Accuracy is computed by comparing model predictions to analyst corrections stored in the database
- If accuracy drops below **70%**, the dashboard displays a retraining alert
- Verified data can be exported to CSV via the dashboard for fine-tuning in a cloud environment
- A minimum of **10 verified reviews** is required to compute a meaningful quality estimate

---

## Dataset

- **Source:** Amazon Customer Reviews (Kaggle)
- **Size:** 34,660 reviews, 21 features
- **Ground truth:** Star ratings (1–5) converted to sentiment labels — no manual annotation required
- **Evaluation:** Stratified sample of 300 reviews (100 per class)
- **Accuracy on evaluation sample:** 71.3%

---

## Project Structure

```
Project AI/
├── api.py                    # FastAPI REST API server
├── dashboard.py              # Streamlit analyst dashboard
├── reviews.db                # SQLite database (created on first run)
├── training_data_export.csv  # Export of verified labels (generated on demand)
├── README.md                 # This file
└── Project AI.ipynb          # Development notebook
```

---

## GitHub

Repository: [Insert GitHub link here]
