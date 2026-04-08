---
title: AI Sentiment Dashboard
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: streamlit
app_file: app.py
pinned: false
---

# 🤖 AI Sentiment Intelligence Studio

Analyze sentiment from **text, tweets, and datasets** using Machine Learning.

Built with Streamlit + HuggingFace Transformers (DistilBERT).

## Features
- 🔍 Instant text sentiment prediction with confidence scores
- 🐦 Batch tweet analyzer (up to 200 tweets)
- 📂 CSV dataset analyzer with full dashboard
- ☁️ Word cloud + word frequency charts
- 📈 Sentiment timeline & distribution analytics
- 🌐 Real-time topic sentiment monitor

## Tech Stack
- **Model**: DistilBERT (HuggingFace Transformers) → fallback to Logistic Regression
- **Framework**: Streamlit
- **Charts**: Plotly, Matplotlib, WordCloud
- **ML**: scikit-learn, PyTorch

## Usage
1. Open the **Single Text** tab and type any review or comment
2. Use the **Tweet Analyzer** tab to paste multiple tweets
3. Upload a CSV in the **CSV Analyzer** tab
4. View full charts in the **Analytics** tab
5. Track any brand/topic in the **Topic Monitor** tab
