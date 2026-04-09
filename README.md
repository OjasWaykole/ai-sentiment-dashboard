# 🧠 AI Sentiment Intelligence Studio

> Production-grade NLP sentiment analysis platform — text, social media, and datasets using HuggingFace Transformers.

🔗 **Live Demo** → [huggingface.co/spaces/OjasWaykole/ai-sentiment-dashboard](https://huggingface.co/spaces/OjasWaykole/ai-sentiment-dashboard)

---

## ✨ Features

| Feature | Description |
|---|---|
| ⚡ **Quick Demo** | One-click demo in the hero strip — no typing needed |
| 🔍 **Text Analysis** | Classify any sentence with confidence score + aspect breakdown |
| 🐦 **Social Media Analyzer** | Paste up to 200 tweets — distribution charts, word cloud, report |
| 📂 **Dataset Analyzer** | Upload CSV → pick text column → classify every row automatically |
| 📊 **Analytics Dashboard** | Pie chart, bar chart, sentiment timeline, word frequency, word cloud |
| 🎯 **Aspect-Based Sentiment** | Identifies sentiment per dimension: Battery · Camera · Price · Support etc. |
| 📄 **Download Report** | Executive summary (count, %, avg confidence) + full results in one CSV |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| NLP Model | HuggingFace Transformers — DistilBERT-SST2 |
| Fallback ML | Scikit-learn — Logistic Regression + TF-IDF |
| Web App | Streamlit |
| Charts | Plotly Express + Plotly Graph Objects |
| Word Cloud | WordCloud + Matplotlib |
| Data | Pandas + NumPy |
| Deployment | HuggingFace Spaces |

---

## 🤖 Model Details

**Primary — DistilBERT** (`distilbert-base-uncased-finetuned-sst-2-english`)
- Fine-tuned on Stanford Sentiment Treebank v2 (SST-2)
- ~91% accuracy · 60% faster than BERT · 40% fewer parameters

**Fallback — Logistic Regression + TF-IDF bigrams**
- 3-class: Positive / Negative / Neutral
- Activates automatically when transformers are unavailable

---

## 🚀 Run Locally

```bash
git clone https://github.com/OjasWaykole/ai-sentiment-dashboard
cd ai-sentiment-dashboard
pip install -r requirements.txt
streamlit run app.py
```

---

## 📄 CV Entry

```
AI Sentiment Intelligence Dashboard                              [Live Demo ↗]
• Built production-grade NLP sentiment platform using Python and ML
• Implemented DistilBERT transformer (HuggingFace) with sklearn fallback
• Developed Streamlit web app with 4 analysis modes + aspect-based sentiment
• Integrated charts, word cloud, sentiment timeline, and downloadable reports
• Deployed publicly on HuggingFace Spaces — instantly testable by recruiters
Tech: Python · HuggingFace Transformers · Streamlit · Plotly · Scikit-learn · Pandas
```

---

## 👨‍💻 Author

**Ojas Waykole** — 2nd Year B.Tech CSE, GCE Jalgaon (NMU University)  
💼 Open to AI/ML & NLP Internship Opportunities · May–December
