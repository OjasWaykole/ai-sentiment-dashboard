import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re, io, base64, time, random

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Sentiment Intelligence Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Theme / CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .hero-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e94560;
    }
    .hero-box h1 { color: #e94560; font-size: 2.2rem; margin-bottom: 0.3rem; }
    .hero-box p  { color: #a0aec0; font-size: 1rem; }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .tag-pos { background:#1a4731; color:#68d391; padding:4px 12px; border-radius:20px; font-weight:600; }
    .tag-neg { background:#4a1c1c; color:#fc8181; padding:4px 12px; border-radius:20px; font-weight:600; }
    .tag-neu { background:#2d3748; color:#a0aec0; padding:4px 12px; border-radius:20px; font-weight:600; }
    .example-btn { cursor:pointer; }
    div[data-testid="stDownloadButton"] button {
        background: #e94560; color: white; border: none; border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Model loader (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML model...")
def load_model():
    """
    Tries to load a HuggingFace transformer pipeline first.
    Falls back to a lightweight sklearn Logistic Regression trained
    on a small seed dataset so the app always works even on CPU-only spaces.
    """
    try:
        from transformers import pipeline
        clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        return ("transformer", clf)
    except Exception:
        pass

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer
        import pickle, os

        seed_texts = [
            "I love this product", "amazing service", "excellent quality", "great experience",
            "very happy", "wonderful", "fantastic", "best ever", "highly recommend", "perfect",
            "I hate this", "terrible service", "awful quality", "worst experience", "very bad",
            "horrible", "dreadful", "never again", "waste of money", "poor quality",
            "it was okay", "not bad", "average", "nothing special", "could be better",
            "decent enough", "mediocre", "so-so", "neither good nor bad", "just fine",
        ]
        seed_labels = [1]*10 + [0]*10 + [2]*10

        vec = TfidfVectorizer(ngram_range=(1, 2))
        X = vec.fit_transform(seed_texts)
        clf = LogisticRegression(max_iter=500, multi_class='multinomial')
        clf.fit(X, seed_labels)
        return ("sklearn", vec, clf)
    except Exception as e:
        return ("dummy", None)


def predict(text, model_pack):
    """Returns (label_str, confidence_dict)"""
    text = text.strip()
    if not text:
        return "Neutral", {"Positive": 0.33, "Negative": 0.33, "Neutral": 0.34}

    kind = model_pack[0]

    if kind == "transformer":
        pipe = model_pack[1]
        res = pipe(text[:512])[0]
        label = res["label"]
        score = res["score"]
        if label == "POSITIVE":
            return "Positive", {"Positive": score, "Negative": round((1-score)*0.4, 3), "Neutral": round((1-score)*0.6, 3)}
        else:
            return "Negative", {"Negative": score, "Positive": round((1-score)*0.4, 3), "Neutral": round((1-score)*0.6, 3)}

    if kind == "sklearn":
        vec, clf = model_pack[1], model_pack[2]
        X = vec.transform([text])
        probs = clf.predict_proba(X)[0]
        classes = clf.classes_
        label_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
        conf = {label_map[c]: round(float(p), 3) for c, p in zip(classes, probs)}
        pred = label_map[clf.predict(X)[0]]
        return pred, conf

    # dummy fallback
    words = text.lower().split()
    pos_w = {"love","great","good","amazing","excellent","wonderful","fantastic","best","happy","perfect"}
    neg_w = {"hate","bad","terrible","awful","horrible","worst","poor","dreadful","useless","disappointing"}
    ps = sum(1 for w in words if w in pos_w)
    ns = sum(1 for w in words if w in neg_w)
    if ps > ns:
        return "Positive", {"Positive": 0.75, "Negative": 0.10, "Neutral": 0.15}
    elif ns > ps:
        return "Negative", {"Negative": 0.75, "Positive": 0.10, "Neutral": 0.15}
    return "Neutral", {"Neutral": 0.60, "Positive": 0.22, "Negative": 0.18}


def sentiment_color(s):
    if s == "Positive": return "tag-pos"
    if s == "Negative": return "tag-neg"
    return "tag-neu"


def clean_text(t):
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"@\w+|#\w+", "", t)
    t = re.sub(r"[^a-zA-Z\s]", "", t)
    return t.strip().lower()


def make_wordcloud(text_series):
    combined = " ".join(text_series.fillna("").apply(clean_text))
    if len(combined.strip()) < 5:
        return None
    wc = WordCloud(width=800, height=350, background_color="black",
                   colormap="RdYlGn", max_words=100).generate(combined)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")
    return fig


def prob_chart(conf_dict):
    colors = {"Positive": "#68d391", "Negative": "#fc8181", "Neutral": "#a0aec0"}
    fig = go.Figure(go.Bar(
        x=list(conf_dict.keys()),
        y=[v * 100 for v in conf_dict.values()],
        marker_color=[colors.get(k, "#gray") for k in conf_dict.keys()],
        text=[f"{v*100:.1f}%" for v in conf_dict.values()],
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#a0aec0",
        yaxis=dict(range=[0, 110], showgrid=False, zeroline=False),
        xaxis=dict(showgrid=False),
        height=280,
        margin=dict(t=20, b=10, l=10, r=10),
    )
    return fig


def timeline_chart(df):
    if "time" not in df.columns:
        df = df.copy()
        df["time"] = pd.date_range(start="2024-01-01", periods=len(df), freq="h")
    score_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
    df["score"] = df["Sentiment"].map(score_map).fillna(0)
    df["rolling"] = df["score"].rolling(window=max(1, len(df)//5), min_periods=1).mean()
    fig = px.line(df, x="time", y="rolling", title="Sentiment Trend Over Time",
                  color_discrete_sequence=["#e94560"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#a0aec0", height=300,
                      yaxis=dict(title="Score", range=[-1.2, 1.2]))
    return fig


def word_freq_chart(text_series, top_n=15):
    stopwords = {"the","a","an","is","in","it","to","and","or","of","was","for","that","this","on","at","i","my","me","so","be"}
    words = " ".join(text_series.fillna("").apply(clean_text)).split()
    words = [w for w in words if w not in stopwords and len(w) > 2]
    common = Counter(words).most_common(top_n)
    if not common:
        return None
    fig = px.bar(x=[w[1] for w in common], y=[w[0] for w in common],
                 orientation="h", title="Top Words",
                 color=[w[1] for w in common], color_continuous_scale="RdYlGn")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#a0aec0", height=400, showlegend=False,
                      yaxis=dict(autorange="reversed"), margin=dict(l=100))
    return fig


def dist_chart(df):
    counts = df["Sentiment"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    color_map = {"Positive": "#68d391", "Negative": "#fc8181", "Neutral": "#a0aec0"}
    fig = px.pie(counts, values="Count", names="Sentiment",
                 color="Sentiment", color_discrete_map=color_map,
                 title="Sentiment Distribution")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0", height=300)
    return fig


# ─── Load model ──────────────────────────────────────────────────────────────
model_pack = load_model()
model_kind = model_pack[0]

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Model Info")
    if model_kind == "transformer":
        st.success("🤗 Transformer (DistilBERT)")
        st.markdown("""
| Property | Value |
|---|---|
| Model | DistilBERT |
| Dataset | SST-2 |
| Labels | Positive / Negative |
| Accuracy | ~91% |
""")
    elif model_kind == "sklearn":
        st.info("📐 Logistic Regression")
        st.markdown("""
| Property | Value |
|---|---|
| Model | Logistic Regression |
| Vectorizer | TF-IDF |
| Labels | Pos / Neg / Neutral |
| Training | Seed dataset |
""")
    else:
        st.warning("🔧 Rule-based fallback")

    st.divider()
    st.markdown("## 🛠️ Settings")
    show_confidence = st.toggle("Show confidence chart", value=True)
    show_wordcloud  = st.toggle("Show word cloud", value=True)
    show_timeline   = st.toggle("Show timeline chart", value=True)
    show_wordfreq   = st.toggle("Show word frequency", value=True)

    st.divider()
    st.markdown("## 📖 About")
    st.caption(
        "AI Sentiment Intelligence Studio — built with Streamlit + "
        "HuggingFace Transformers. Analyzes text, tweets, and CSV datasets."
    )

# ─── Hero section ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
  <h1>🤖 AI Sentiment Intelligence Studio</h1>
  <p>
    Analyze sentiment from <b>reviews, tweets, and datasets</b> using Machine Learning.<br>
    Understand public opinion and customer feedback in seconds.
  </p>
  <p style="margin-top:0.8rem;">
    🔍 Text Prediction &nbsp;|&nbsp;
    🐦 Tweet Batch Analyzer &nbsp;|&nbsp;
    📊 CSV Dataset Dashboard &nbsp;|&nbsp;
    ☁️ Word Cloud &nbsp;|&nbsp;
    📈 Analytics
  </p>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Single Text",
    "🐦 Tweet Analyzer",
    "📂 CSV Analyzer",
    "📈 Analytics",
    "🌐 Topic Monitor",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Text Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Analyze Any Text")

    # Example buttons
    st.markdown("**Try an example:**")
    examples = [
        "I absolutely love this product — best purchase I've made all year!",
        "The customer service was terrible. I waited 2 hours for nothing.",
        "The experience was okay. Nothing really stood out, but no complaints either.",
    ]
    c1, c2, c3 = st.columns(3)
    clicked = None
    if c1.button("😊 Positive example", use_container_width=True): clicked = examples[0]
    if c2.button("😠 Negative example", use_container_width=True): clicked = examples[1]
    if c3.button("😐 Neutral example",  use_container_width=True): clicked = examples[2]

    default_val = clicked if clicked else st.session_state.get("single_text", "")
    user_text = st.text_area("Enter your text here:", value=default_val,
                              height=140, key="single_text",
                              placeholder="Type a review, tweet, comment…")

    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if user_text.strip():
            with st.spinner("Analyzing…"):
                label, conf = predict(user_text, model_pack)

            tag_cls = sentiment_color(label)
            st.markdown(f"""
            <div style="background:#1a1a2e;border-radius:12px;padding:1.2rem;margin-top:1rem;border:1px solid #2d3748;">
              <h3 style="color:#e2e8f0;margin-bottom:0.5rem;">Result</h3>
              <span class="{tag_cls}" style="font-size:1.1rem;">{label}</span>
              <p style="color:#718096;margin-top:0.8rem;font-size:0.9rem;">
                Top confidence: <b style="color:#e2e8f0;">{max(conf.values())*100:.1f}%</b>
              </p>
            </div>
            """, unsafe_allow_html=True)

            if show_confidence:
                st.plotly_chart(prob_chart(conf), use_container_width=True)
        else:
            st.warning("Please enter some text first.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Tweet Batch Analyzer
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Batch Tweet Sentiment Analyzer")
    st.caption("Paste one tweet per line — up to 200 tweets at once.")

    tweet_input = st.text_area(
        "Paste tweets (one per line):",
        height=200,
        placeholder=(
            "I love the new iPhone update!\n"
            "Tesla's Autopilot is getting worse, not better.\n"
            "Just tried the new coffee shop downtown, pretty decent.\n"
            "This new policy is absolutely ridiculous.\n"
            "Had an okay day. Nothing special."
        ),
    )

    if st.button("🐦 Analyze Tweets", type="primary", use_container_width=True):
        tweets = [t.strip() for t in tweet_input.strip().splitlines() if t.strip()]
        if tweets:
            results = []
            bar = st.progress(0, text="Analyzing tweets…")
            for i, tweet in enumerate(tweets):
                label, conf = predict(tweet, model_pack)
                results.append({
                    "Tweet": tweet,
                    "Sentiment": label,
                    "Confidence": f"{max(conf.values())*100:.1f}%",
                    "Positive %": f"{conf.get('Positive',0)*100:.1f}",
                    "Negative %": f"{conf.get('Negative',0)*100:.1f}",
                    "Neutral %":  f"{conf.get('Neutral',0)*100:.1f}",
                })
                bar.progress((i+1)/len(tweets), text=f"Analyzed {i+1}/{len(tweets)}")
            bar.empty()

            df = pd.DataFrame(results)
            st.success(f"✅ Analyzed **{len(df)}** tweets")

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            pos_pct = (df["Sentiment"]=="Positive").mean()*100
            neg_pct = (df["Sentiment"]=="Negative").mean()*100
            neu_pct = (df["Sentiment"]=="Neutral").mean()*100
            m1.metric("Total Tweets", len(df))
            m2.metric("😊 Positive",  f"{pos_pct:.0f}%")
            m3.metric("😠 Negative",  f"{neg_pct:.0f}%")
            m4.metric("😐 Neutral",   f"{neu_pct:.0f}%")

            # Distribution pie
            st.plotly_chart(dist_chart(df), use_container_width=True)

            # Data table
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Word cloud
            if show_wordcloud:
                with st.spinner("Generating word cloud…"):
                    fig = make_wordcloud(df["Tweet"])
                    if fig: st.pyplot(fig)

            # Download
            csv_bytes = df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Results CSV", csv_bytes,
                               "tweet_sentiment.csv", "text/csv",
                               use_container_width=True)
        else:
            st.warning("Please paste at least one tweet.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CSV Analyzer
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("CSV Dataset Analyzer")
    st.caption("Upload a CSV with a text column. The app will analyze sentiment for every row.")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded:
        raw_df = pd.read_csv(uploaded)
        st.info(f"Loaded **{len(raw_df)}** rows, **{len(raw_df.columns)}** columns")
        st.dataframe(raw_df.head(5), use_container_width=True)

        text_col = st.selectbox("Select the text column:", raw_df.columns.tolist())
        max_rows = st.slider("Max rows to analyze:", 10, min(500, len(raw_df)), min(100, len(raw_df)))

        if st.button("📊 Run Analysis", type="primary", use_container_width=True):
            subset = raw_df[[text_col]].dropna().head(max_rows).copy()
            subset.columns = ["Text"]

            labels, confs = [], []
            bar = st.progress(0, text="Analyzing rows…")
            for i, row in enumerate(subset["Text"]):
                lbl, cf = predict(str(row), model_pack)
                labels.append(lbl)
                confs.append(max(cf.values()))
                if i % 10 == 0:
                    bar.progress((i+1)/len(subset), text=f"{i+1}/{len(subset)}")
            bar.progress(1.0, text="Done!")
            bar.empty()

            subset["Sentiment"] = labels
            subset["Confidence"] = [f"{c*100:.1f}%" for c in confs]
            st.session_state["csv_results"] = subset
            st.success(f"✅ Analyzed {len(subset)} rows")

        if "csv_results" in st.session_state:
            df = st.session_state["csv_results"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows Analyzed", len(df))
            m2.metric("😊 Positive", f"{(df['Sentiment']=='Positive').mean()*100:.0f}%")
            m3.metric("😠 Negative", f"{(df['Sentiment']=='Negative').mean()*100:.0f}%")
            m4.metric("😐 Neutral",  f"{(df['Sentiment']=='Neutral').mean()*100:.0f}%")

            col_l, col_r = st.columns(2)
            with col_l:
                st.plotly_chart(dist_chart(df), use_container_width=True)
            with col_r:
                if show_wordfreq:
                    fig = word_freq_chart(df["Text"])
                    if fig: st.plotly_chart(fig, use_container_width=True)

            if show_timeline:
                st.plotly_chart(timeline_chart(df), use_container_width=True)

            if show_wordcloud:
                with st.spinner("Generating word cloud…"):
                    wc_fig = make_wordcloud(df["Text"])
                    if wc_fig: st.pyplot(wc_fig)

            st.dataframe(df, use_container_width=True, hide_index=True)

            csv_out = df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Full Results", csv_out,
                               "sentiment_analysis.csv", "text/csv",
                               use_container_width=True)
    else:
        # Demo with sample data
        st.markdown("---")
        if st.button("▶️ Run Demo with Sample Data", use_container_width=True):
            sample_texts = [
                "Absolutely loved the new update!", "Terrible customer support.",
                "Product is decent, nothing special.", "Best app I've ever used.",
                "Keeps crashing on my phone.", "Pretty good overall experience.",
                "The battery life is amazing.", "Disappointed with the quality.",
                "Works exactly as described.", "Not worth the price at all.",
                "Shipping was super fast!", "Product broke after 2 days.",
                "Great value for the money.", "Interface is confusing.",
                "Highly recommend to everyone.", "Very slow and laggy.",
            ]
            demo_df = pd.DataFrame({"Text": sample_texts})
            labels, confs = [], []
            for t in sample_texts:
                lbl, cf = predict(t, model_pack)
                labels.append(lbl)
                confs.append(max(cf.values()))
            demo_df["Sentiment"] = labels
            demo_df["Confidence"] = [f"{c*100:.1f}%" for c in confs]
            st.session_state["csv_results"] = demo_df
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Analytics Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📈 Analytics Dashboard")

    if "csv_results" not in st.session_state:
        st.info("💡 Run a CSV analysis first (Tab 3) or the demo, then come back here for the full dashboard.")
    else:
        df = st.session_state["csv_results"]

        st.markdown("### Overview Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Texts", len(df))
        m2.metric("😊 Positive", f"{(df['Sentiment']=='Positive').sum()}")
        m3.metric("😠 Negative", f"{(df['Sentiment']=='Negative').sum()}")
        m4.metric("😐 Neutral",  f"{(df['Sentiment']=='Neutral').sum()}")
        avg_conf = df["Confidence"].str.replace("%","").astype(float).mean()
        m5.metric("Avg Confidence", f"{avg_conf:.1f}%")

        st.markdown("### Sentiment Distribution")
        st.plotly_chart(dist_chart(df), use_container_width=True)

        st.markdown("### Sentiment Timeline")
        if show_timeline:
            st.plotly_chart(timeline_chart(df), use_container_width=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("### Word Frequency")
            if show_wordfreq:
                fig = word_freq_chart(df["Text"])
                if fig: st.plotly_chart(fig, use_container_width=True)
        with col_r:
            st.markdown("### Word Cloud")
            if show_wordcloud:
                wc_fig = make_wordcloud(df["Text"])
                if wc_fig: st.pyplot(wc_fig)

        st.markdown("### Full Data Table")
        sentiment_filter = st.multiselect("Filter by sentiment:",
                                          ["Positive", "Negative", "Neutral"],
                                          default=["Positive", "Negative", "Neutral"])
        filtered = df[df["Sentiment"].isin(sentiment_filter)]
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download Filtered Results",
                           filtered.to_csv(index=False).encode(),
                           "filtered_results.csv", "text/csv",
                           use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Topic / Live Monitor
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🌐 Real-Time Topic Sentiment Monitor")
    st.caption(
        "Enter a topic or brand name. The app generates a simulated live "
        "sentiment feed — useful to prototype a real social-media monitoring tool."
    )

    topic = st.text_input("Enter a topic / brand:", placeholder="Tesla, iPhone, Python…")

    templates = {
        "positive": [
            "{t} is amazing, I love it!", "Best thing I've used: {t}",
            "So impressed with {t} today!", "{t} never disappoints 🔥",
            "Can't imagine life without {t}", "Huge fan of {t} ❤️",
        ],
        "negative": [
            "{t} is really frustrating me today.", "Disappointed with {t} again.",
            "{t} keeps failing. Not happy.", "Why is {t} so bad lately?",
            "Worst experience with {t}.", "Totally let down by {t}.",
        ],
        "neutral": [
            "Just tried {t} for the first time.", "Heard a lot about {t} recently.",
            "Not sure what to think of {t}.", "{t} seems okay I guess.",
            "Using {t} for my project.", "Interesting to see how {t} evolves.",
        ],
    }

    n_tweets = st.slider("Number of simulated posts:", 20, 200, 50, step=10)

    if st.button("🔴 Start Monitor", type="primary", use_container_width=True) and topic:
        sentiments = random.choices(
            ["positive", "negative", "neutral"],
            weights=[0.50, 0.30, 0.20],
            k=n_tweets,
        )
        posts = []
        for s in sentiments:
            text = random.choice(templates[s]).format(t=topic)
            posts.append(text)

        sim_df = pd.DataFrame({"Text": posts})
        labels, confs = [], []
        bar = st.progress(0)
        for i, t in enumerate(posts):
            lbl, cf = predict(t, model_pack)
            labels.append(lbl)
            confs.append(max(cf.values()))
            bar.progress((i+1)/len(posts))
        bar.empty()

        sim_df["Sentiment"] = labels
        sim_df["Confidence"] = [f"{c*100:.1f}%" for c in confs]
        sim_df["time"] = pd.date_range(end=pd.Timestamp.now(), periods=len(sim_df), freq="5min")

        pos_pct = (sim_df["Sentiment"]=="Positive").mean()*100
        neg_pct = (sim_df["Sentiment"]=="Negative").mean()*100
        neu_pct = (sim_df["Sentiment"]=="Neutral").mean()*100

        overall = "🟢 Mostly Positive" if pos_pct > 50 else ("🔴 Mostly Negative" if neg_pct > 40 else "🟡 Mixed")

        st.markdown(f"""
        <div style="background:#1a1a2e;border-radius:12px;padding:1.5rem;border:1px solid #2d3748;margin-bottom:1rem;">
          <h3 style="color:#e2e8f0;margin:0 0 0.5rem 0;">📊 Topic: <span style="color:#e94560;">{topic}</span></h3>
          <p style="color:#a0aec0;font-size:1.1rem;margin:0">Overall Sentiment: <b style="color:#e2e8f0">{overall}</b></p>
          <p style="color:#718096;font-size:0.9rem;margin-top:0.4rem">
            Posts analyzed: {n_tweets} &nbsp;|&nbsp;
            😊 {pos_pct:.0f}% &nbsp;|&nbsp;
            😠 {neg_pct:.0f}% &nbsp;|&nbsp;
            😐 {neu_pct:.0f}%
          </p>
        </div>
        """, unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(dist_chart(sim_df), use_container_width=True)
        with col_r:
            st.plotly_chart(timeline_chart(sim_df), use_container_width=True)

        if show_wordcloud:
            wc_fig = make_wordcloud(sim_df["Text"])
            if wc_fig: st.pyplot(wc_fig)

        st.dataframe(sim_df[["Text","Sentiment","Confidence"]], use_container_width=True, hide_index=True)

        st.download_button("⬇️ Download Monitor Results",
                           sim_df.to_csv(index=False).encode(),
                           f"{topic}_monitor.csv", "text/csv",
                           use_container_width=True)

    elif not topic and st.button("🔴 Start Monitor", key="mon2"):
        st.warning("Please enter a topic name first.")
