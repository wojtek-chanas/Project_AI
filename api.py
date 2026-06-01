"""
Sentiment Analysis API
======================
REST API for customer review sentiment analysis.
Built with FastAPI, HuggingFace Transformers, and SQLite.

Endpoints:
    POST /predict              - Classify sentiment of a review
    POST /correct              - Submit human correction for a prediction
    GET  /reviews              - Retrieve all stored reviews
    GET  /quality              - Check model quality against verified labels
    GET  /export-training-data - Export verified reviews for fine-tuning
    GET  /health               - API health check
"""

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import sqlite3
import pandas as pd
from datetime import datetime
import os

app = FastAPI(
    title="Sentiment Analysis API",
    description="Customer review sentiment classification using HuggingFace BERT model.",
    version="1.0.0"
)

# Absolute path to the database file, co-located with this script
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reviews.db")

# Accuracy threshold below which retraining is recommended
QUALITY_THRESHOLD = 0.70

# Minimum number of verified reviews required for a meaningful quality estimate
MIN_VERIFIED = 10

# Load pretrained multilingual BERT model at startup.
# Output: star rating label (1-5 stars) + confidence score.
# Mapped to three sentiment classes: negative / neutral / positive.
classifier = pipeline(
    "text-classification",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class ReviewInput(BaseModel):
    """Input model for a single review prediction request."""
    text: str


class CorrectionInput(BaseModel):
    """Input model for submitting a human correction to a stored prediction."""
    review_id: int
    verified_sentiment: str


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def stars_to_sentiment(label: str) -> str:
    """
    Map a star rating label to a sentiment class.

    Args:
        label: Star rating string returned by the model, e.g. '4 stars'.

    Returns:
        One of 'negative', 'neutral', or 'positive'.
    """
    stars = int(label[0])
    if stars <= 2:
        return "negative"
    elif stars == 3:
        return "neutral"
    else:
        return "positive"


def init_db() -> None:
    """Initialize SQLite database and create the reviews table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            text               TEXT    NOT NULL,
            sentiment          TEXT    NOT NULL,
            stars              TEXT    NOT NULL,
            confidence         REAL    NOT NULL,
            verified_sentiment TEXT,
            timestamp          TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def get_db_connection() -> sqlite3.Connection:
    """Return an open connection to the reviews database."""
    return sqlite3.connect(DB_PATH)


# Run once at startup to ensure the database is ready
init_db()


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict")
def predict(review: ReviewInput) -> dict:
    """
    Classify the sentiment of a customer review.

    The raw text is passed to the BERT model, which returns a star rating
    label and confidence score. The label is mapped to a three-class sentiment
    and the result is persisted to the database before being returned.
    """
    result = classifier(review.text, truncation=True, max_length=512)[0]
    prediction = {
        "text":       review.text,
        "sentiment":  stars_to_sentiment(result["label"]),
        "stars":      result["label"],
        "confidence": round(result["score"], 4),
    }
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO reviews (text, sentiment, stars, confidence, timestamp) VALUES (?, ?, ?, ?, ?)",
        (prediction["text"], prediction["sentiment"], prediction["stars"],
         prediction["confidence"], datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return prediction


@app.post("/correct")
def correct(correction: CorrectionInput) -> dict:
    """
    Submit a human correction for a stored prediction.

    Updates the verified_sentiment field for the given review ID.
    Corrections feed the human-in-the-loop quality control workflow:
    once enough verified labels are collected, /quality uses them to
    assess whether the model needs retraining.
    """
    conn = get_db_connection()
    conn.execute(
        "UPDATE reviews SET verified_sentiment = ? WHERE id = ?",
        (correction.verified_sentiment, correction.review_id)
    )
    conn.commit()
    conn.close()
    return {"status": "updated", "id": correction.review_id}


@app.get("/reviews")
def get_reviews() -> list:
    """Retrieve all stored reviews ordered by timestamp descending."""
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM reviews ORDER BY timestamp DESC", conn)
    conn.close()
    return df.to_dict(orient="records")


@app.get("/quality")
def quality_check() -> dict:
    """
    Check model quality against human-verified labels.

    Compares the model's original predictions with analyst corrections.
    Returns a retraining alert when accuracy falls below QUALITY_THRESHOLD.
    Requires at least MIN_VERIFIED verified reviews for a meaningful estimate.
    """
    conn = get_db_connection()
    df = pd.read_sql(
        "SELECT sentiment, verified_sentiment FROM reviews WHERE verified_sentiment IS NOT NULL",
        conn
    )
    conn.close()

    if len(df) < MIN_VERIFIED:
        return {
            "status": "insufficient_data",
            "verified_count": len(df),
            "message": f"Need at least {MIN_VERIFIED} verified reviews, have {len(df)}",
        }

    accuracy = float((df["sentiment"] == df["verified_sentiment"]).mean())
    needs_retraining = accuracy < QUALITY_THRESHOLD

    return {
        "status":         "retrain_needed" if needs_retraining else "ok",
        "accuracy":       round(accuracy, 3),
        "verified_count": len(df),
        "threshold":      QUALITY_THRESHOLD,
        "message": (
            f"Accuracy {accuracy:.1%} — retraining recommended"
            if needs_retraining
            else f"Accuracy {accuracy:.1%} — model performing well"
        ),
    }


@app.get("/export-training-data")
def export_training_data() -> dict:
    """
    Export human-verified reviews to CSV for model fine-tuning.

    Produces a two-column CSV (text, sentiment) containing all reviews
    that have been verified by an analyst. This file can be used to
    fine-tune the model in a cloud environment when quality monitoring
    indicates performance degradation.
    """
    conn = get_db_connection()
    df = pd.read_sql(
        "SELECT text, verified_sentiment AS sentiment FROM reviews WHERE verified_sentiment IS NOT NULL",
        conn
    )
    conn.close()

    export_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data_export.csv")
    df.to_csv(export_path, index=False)

    return {
        "status":  "exported",
        "records": len(df),
        "path":    export_path,
        "message": f"Exported {len(df)} verified reviews for fine-tuning",
    }

@app.post("/auto-verify")
def auto_verify(threshold: float = 0.80) -> dict:
    """
    Automatically verify predictions where confidence exceeds the threshold.
    
    Reviews with high model confidence are unlikely to be wrong and can be
    verified without human review. This reduces the manual workload while
    maintaining data quality for the retraining pipeline.
    
    Args:
        threshold: Minimum confidence score for automatic verification (default 0.80).
    
    Returns:
        Count of automatically verified reviews.
    """
    conn = get_db_connection()
    cursor = conn.execute("""
        UPDATE reviews
        SET verified_sentiment = sentiment
        WHERE verified_sentiment IS NULL
          AND confidence >= ?
    """, (threshold,))
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return {
        "status": "ok",
        "auto_verified": count,
        "threshold": threshold,
        "message": f"Auto-verified {count} reviews with confidence >= {threshold:.0%}"
    }

@app.get("/health")
def health() -> dict:
    """Health check endpoint. Returns ok if the API is running."""
    return {"status": "ok"}