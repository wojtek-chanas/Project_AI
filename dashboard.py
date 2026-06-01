"""
Sentiment Analysis Dashboard
=============================
Streamlit dashboard for the customer review sentiment analysis system.

Provides:
  - Model quality monitoring with retraining alert
  - Auto-verification of high-confidence predictions
  - Needs Verification queue — one review at a time with three-button workflow
  - Single-review sentiment analysis
  - Review history table with sentiment distribution chart
  - Human-in-the-loop correction workflow
  - Training data export for model fine-tuning

Usage:
    streamlit run dashboard.py

Requires the FastAPI server to be running on http://localhost:8000.
"""

import streamlit as st
import requests
import pandas as pd

# Base URL of the FastAPI backend
API_URL = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_quality() -> dict:
    """Fetch model quality metrics from the API."""
    return requests.get(f"{API_URL}/quality").json()


def get_reviews() -> list:
    """Fetch all stored reviews from the API."""
    response = requests.get(f"{API_URL}/reviews")
    return response.json() if response.status_code == 200 else []


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Sentiment Analysis Dashboard")
st.markdown("Customer review sentiment analysis system — *DLBDSEAIS02 Task 2*")
st.divider()


# ---------------------------------------------------------------------------
# Section 1: Model Quality
# ---------------------------------------------------------------------------

st.header("Model Quality")

quality = get_quality()

if quality["status"] == "ok":
    st.success(quality["message"])
elif quality["status"] == "retrain_needed":
    st.warning(quality["message"])
    if st.button("📥 Export Training Data for Retraining"):
        export = requests.get(f"{API_URL}/export-training-data").json()
        st.info(f"{export['message']}  →  `{export['path']}`")
else:
    st.info(quality["message"])

col1, col2, col3 = st.columns(3)
col1.metric("Verified Reviews", quality.get("verified_count", 0))
col2.metric(
    "Model Accuracy",
    f"{quality['accuracy'] * 100:.1f}%" if "accuracy" in quality else "N/A"
)
col3.metric("Threshold", f"{quality.get('threshold', 0.70) * 100:.0f}%")


# Initialize state
if "auto_verify_enabled" not in st.session_state:
    st.session_state.auto_verify_enabled = False

#st.toggle(
#    "Enable Auto-verification",
#    key="auto_verify_enabled",
#    help="Automatically verify predictions with confidence >= 80%"
#)

if "auto_verify_enabled" not in st.session_state:
    st.session_state.auto_verify_enabled = False

if "auto_verify_threshold" not in st.session_state:
    st.session_state.auto_verify_threshold = 90

st.toggle(
    "Enable Auto-verification",
    key="auto_verify_enabled",
    help="Automatically verify predictions with confidence >= threshold"
)

if st.session_state.auto_verify_enabled:
    st.slider(
        "Confidence threshold",
        min_value=50,
        max_value=99,
        key="auto_verify_threshold",
        format="%d%%",
        help="Predictions with confidence above this value will be auto-verified"
    )

    if st.button("⚡ Auto-verify High-Confidence Predictions"):
        threshold = st.session_state.auto_verify_threshold / 100
        response = requests.post(
            f"{API_URL}/auto-verify",
            params={"threshold": threshold}
        )
        result = response.json()
        st.success(result["message"])
        st.rerun()
else:
    st.caption("Auto-verification is disabled.")
# ---------------------------------------------------------------------------
# Section 3: Review History
# ---------------------------------------------------------------------------

st.header("Review History")

reviews = get_reviews()

if reviews:
    df = pd.DataFrame(reviews)

    # --- Summary metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", len(df))
    col2.metric("Pending Verification", int(df["verified_sentiment"].isna().sum()))
    col3.metric("Verified", int(df["verified_sentiment"].notna().sum()))

    st.divider()

    # --- Needs Verification queue ---
    st.subheader("Needs Verification")
    pending = df[df["verified_sentiment"].isna()]

    if pending.empty:
        st.success("All reviews have been verified.")
    else:
        st.caption(f"{len(pending)} review(s) awaiting verification.")
        review = pending.iloc[0]

        st.info(f"**Review ID: {int(review['id'])}**\n\n{review['text']}")
        st.caption(
            f"Model prediction: **{review['sentiment'].upper()}** "
            f"({float(review['confidence']):.0%} confidence)"
        )

        col1, col2, col3 = st.columns(3)
        if col1.button("👎 Negative", use_container_width=True):
            requests.post(f"{API_URL}/correct",
                          json={"review_id": int(review['id']),
                                "verified_sentiment": "negative"})
            st.rerun()
        if col2.button("😐 Neutral", use_container_width=True):
            requests.post(f"{API_URL}/correct",
                          json={"review_id": int(review['id']),
                                "verified_sentiment": "neutral"})
            st.rerun()
        if col3.button("👍 Positive", use_container_width=True):
            requests.post(f"{API_URL}/correct",
                          json={"review_id": int(review['id']),
                                "verified_sentiment": "positive"})
            st.rerun()

    st.divider()

    # --- Sentiment distribution chart ---
    st.subheader("Sentiment Distribution")
    st.bar_chart(df["sentiment"].value_counts())

    # --- Full review table ---
    st.subheader("All Reviews")
    st.dataframe(
        df[["id", "text", "sentiment", "stars", "confidence",
            "verified_sentiment", "timestamp"]],
        use_container_width=True
    )

    st.divider()

    # --- Manual correction form ---
    st.subheader("Correct a Prediction")
    st.caption("Use this form to submit a verified label for any stored review by ID.")

    review_id = st.number_input("Review ID", min_value=1, step=1)
    new_sentiment = st.selectbox(
        "Verified Sentiment",
        ["positive", "neutral", "negative"],
        help="Select the correct sentiment label for this review."
    )

    if st.button("Submit Correction"):
        response = requests.post(
            f"{API_URL}/correct",
            json={"review_id": int(review_id), "verified_sentiment": new_sentiment}
        )
        if response.status_code == 200:
            st.success(f"Review {review_id} updated to **{new_sentiment}**.")
            st.rerun()
        else:
            st.error("Failed to submit correction. Check that the review ID exists.")

else:
    st.info("No reviews stored yet. Use the Analyse section above to classify your first review.")