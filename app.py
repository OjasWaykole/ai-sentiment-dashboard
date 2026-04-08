import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Sentiment Intelligence Studio",
    page_icon="🚀",
    layout="wide"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<h1 style='text-align:center;
background: linear-gradient(90deg,#6366f1,#06b6d4,#22c55e);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;'>
AI Sentiment Intelligence Studio
</h1>
""", unsafe_allow_html=True)

st.write("Analyze reviews, tweets and datasets with Machine Learning")

st.divider()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

data = pd.read_csv("train.csv", encoding="latin-1")
data = data[['text','sentiment']]
data = data.dropna()

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['sentiment'], test_size=0.2
)

vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec, y_train)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚡ AI Tools")

feature = st.sidebar.radio(
    "Choose Tool",
    [
        "Sentiment Predictor",
        "Paragraph Analyzer",
        "Tweet Analyzer",
        "CSV Dataset Analyzer",
        "Analytics Dashboard",
        "Live Sentiment Monitor"
    ]
)

# --------------------------------------------------
# SENTIMENT PREDICTOR
# --------------------------------------------------

if feature == "Sentiment Predictor":

    st.subheader("🔍 Instant Sentiment Prediction")

    user_input = st.text_area("Enter sentence")

    if st.button("Analyze Sentiment"):

        vec = vectorizer.transform([user_input])

        prediction = model.predict(vec)[0]
        confidence = model.predict_proba(vec).max()

        col1, col2 = st.columns(2)

        with col1:

            if prediction == "positive":
                st.success("😊 Positive Sentiment")

            elif prediction == "negative":
                st.error("😡 Negative Sentiment")

            else:
                st.info("😐 Neutral Sentiment")

        with col2:
            st.metric("Confidence", f"{round(confidence*100,2)}%")

        # Explainable AI
        feature_names = vectorizer.get_feature_names_out()

        vec = vectorizer.transform([user_input])

        important_words = []

        for idx in vec.nonzero()[1]:
            important_words.append(feature_names[idx])

        if important_words:
            st.write("Detected keywords:")
            st.write(", ".join(important_words))

        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append((user_input, prediction))

# --------------------------------------------------
# PARAGRAPH ANALYZER
# --------------------------------------------------

elif feature == "Paragraph Analyzer":

    st.subheader("📄 Paragraph Sentiment Analysis")

    paragraph = st.text_area("Paste paragraph")

    if st.button("Analyze Paragraph"):

        sentences = paragraph.split(".")

        results = []

        for sentence in sentences:

            if sentence.strip() != "":
                vec = vectorizer.transform([sentence])
                pred = model.predict(vec)[0]
                results.append((sentence.strip(), pred))

        df = pd.DataFrame(results, columns=["Sentence","Sentiment"])

        st.dataframe(df)

# --------------------------------------------------
# TWEET ANALYZER
# --------------------------------------------------

elif feature == "Tweet Analyzer":

    st.subheader("🐦 Tweet Sentiment Analyzer")

    tweets = st.text_area("Paste tweets (one per line)")

    if st.button("Analyze Tweets"):

        tweet_list = tweets.split("\n")

        results = []

        for tweet in tweet_list:

            if tweet.strip() != "":
                vec = vectorizer.transform([tweet])
                pred = model.predict(vec)[0]

                results.append((tweet, pred))

        df = pd.DataFrame(results, columns=["Tweet","Sentiment"])

        st.dataframe(df)

        csv = df.to_csv(index=False)

        st.download_button(
            "Download Results",
            csv,
            "tweet_sentiment_results.csv"
        )

# --------------------------------------------------
# CSV DATASET ANALYZER
# --------------------------------------------------

elif feature == "CSV Dataset Analyzer":

    st.subheader("📂 Upload Dataset")

    file = st.file_uploader("Upload CSV")

    if file:

        df = pd.read_csv(file)

        predictions = []

        for review in df["text"]:

            vec = vectorizer.transform([str(review)])
            pred = model.predict(vec)[0]

            predictions.append(pred)

        df["Predicted Sentiment"] = predictions

        st.dataframe(df)

        csv = df.to_csv(index=False)

        st.download_button(
            "Download Results",
            csv,
            "dataset_sentiment_results.csv"
        )

# --------------------------------------------------
# ANALYTICS DASHBOARD
# --------------------------------------------------

elif feature == "Analytics Dashboard":

    st.subheader("📊 Sentiment Insights")

    total_reviews = len(data)

    positive = (data["sentiment"] == "positive").sum()
    neutral = (data["sentiment"] == "neutral").sum()
    negative = (data["sentiment"] == "negative").sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Reviews", total_reviews)
    col2.metric("Positive", positive)
    col3.metric("Neutral", neutral)
    col4.metric("Negative", negative)

    sample = data.sample(200)

    preds = []

    for text in sample["text"]:

        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]

        preds.append(pred)

    chart_data = pd.Series(preds).value_counts()

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            x=chart_data.index,
            y=chart_data.values,
            color=chart_data.index,
            title="Sentiment Count"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig2 = px.pie(
            values=chart_data.values,
            names=chart_data.index,
            title="Sentiment Percentage"
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🌎 Word Cloud")

    text_data = " ".join(data["text"].astype(str))

    wordcloud = WordCloud(
        width=900,
        height=400,
        background_color="white",
        colormap="viridis"
    ).generate(text_data)

    fig3, ax3 = plt.subplots(figsize=(10,5))

    ax3.imshow(wordcloud)
    ax3.axis("off")

    st.pyplot(fig3)

# --------------------------------------------------
# LIVE SENTIMENT MONITOR
# --------------------------------------------------

elif feature == "Live Sentiment Monitor":

    st.subheader("🔴 Live Sentiment Monitoring")

    live_input = st.text_area("Enter live comments or tweets (one per line)")

    if st.button("Start Monitoring"):

        lines = live_input.split("\n")

        sentiments = []

        progress = st.progress(0)

        for i, line in enumerate(lines):

            if line.strip() != "":

                vec = vectorizer.transform([line])

                pred = model.predict(vec)[0]

                sentiments.append(pred)

                progress.progress((i+1)/len(lines))

                time.sleep(0.1)

        df = pd.DataFrame({
            "Text": lines,
            "Sentiment": sentiments
        })

        st.dataframe(df)

        sentiment_counts = df["Sentiment"].value_counts()

        fig = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            color=sentiment_counts.index,
            title="Live Sentiment Trend"
        )

        st.plotly_chart(fig)

# --------------------------------------------------
# HISTORY PANEL
# --------------------------------------------------

st.sidebar.subheader("📜 Prediction History")

if "history" in st.session_state:

    for item in st.session_state.history[-5:]:

        st.sidebar.write(f"{item[0]} → {item[1]}")