metadata
title: AI Sentiment Intelligence Studio
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
python_version: '3.10'
app_file: app.py
pinned: false
🧠 AI Sentiment Intelligence Studio
Production-grade NLP sentiment analysis platform — analyze text, social media posts, and datasets using HuggingFace Transformers.

🔗 Live Demo https://huggingface.co/spaces/OjasWaykole/ai-sentiment-dashboard

✨ Features
Feature	Description
⚡ Quick Demo	One-click demo to instantly test sentiment analysis
🔍 Text Analysis	Classify any sentence or paragraph with confidence score
🐦 Social Media Analyzer	Paste tweets and analyze sentiment distribution
📂 Dataset Analyzer	Upload CSV datasets and classify each row automatically
📊 Analytics Dashboard	Pie chart, bar chart, sentiment timeline
☁️ Word Cloud	Visualize most common keywords
📄 Download Report	Export sentiment results as CSV
🤖 Model
Primary Model

DistilBERT distilbert-base-uncased-finetuned-sst-2-english

~91% accuracy
trained on Stanford SST-2 dataset
optimized transformer inference
Fallback Model

Logistic Regression
TF-IDF Vectorizer
Scikit-learn implementation
🛠 Tech Stack
Layer	Technology
NLP Model	HuggingFace Transformers
ML Fallback	Scikit-learn
Web App	Streamlit
Charts	Plotly
Word Cloud	WordCloud
Data	Pandas + NumPy
Deployment	HuggingFace Spaces
🚀 Run Locally
Clone repository

git clone https://github.com/OjasWaykole/ai-sentiment-dashboard
cd ai-sentiment-dashboard
Install dependencies

pip install -r requirements.txt
Run the app

streamlit run app.py
Open browser

http://localhost:8501
👨‍💻 Author
Ojas Waykole B.Tech Computer Science Government College of Engineering, Jalgaon (NMU University)

📧 Email: owaykole@gmail.com

💼 Open to AI / ML / NLP Internship Opportunities
