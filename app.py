"""
AI Sentiment Intelligence Studio — Production Edition
Author : Ojas Waykole | 2nd Year B.Tech CSE | GCE Jalgaon
Stack  : Streamlit · HuggingFace Transformers · Scikit-learn · Plotly · WordCloud
"""

import re, io
from collections import Counter
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
from wordcloud import WordCloud

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Sentiment Intelligence Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Dark Neon Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .block-container { padding-top: 1rem; padding-bottom: 2rem; }

  .hero-wrap {
    background: linear-gradient(135deg,#0d0d1a 0%,#111128 40%,#0a1628 100%);
    border:1px solid #e94560; border-radius:18px;
    padding:2.2rem 2.8rem 2rem; margin-bottom:1rem;
    position:relative; overflow:hidden;
  }
  .hero-wrap::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:260px; height:260px;
    background:radial-gradient(circle,rgba(233,69,96,.18) 0%,transparent 70%);
  }
  .hero-title {
    font-family:'Space Mono',monospace;
    font-size:2rem; font-weight:700; color:#e94560;
    margin:0 0 .4rem; letter-spacing:-.5px;
  }
  .hero-sub  { color:#8892a4; font-size:.95rem; margin:0 0 1rem; }
  .hero-badges span {
    display:inline-block; margin:.3rem .5rem 0 0;
    padding:4px 13px; border-radius:20px; font-size:.78rem; font-weight:500;
  }
  .b-red   {background:#2b0d15;color:#e94560;border:1px solid #e94560;}
  .b-blue  {background:#0d1c2b;color:#60a5fa;border:1px solid #60a5fa;}
  .b-green {background:#0d2018;color:#4ade80;border:1px solid #4ade80;}
  .b-gold  {background:#1f1a0d;color:#fbbf24;border:1px solid #fbbf24;}
  .b-purple{background:#1a0d2b;color:#c084fc;border:1px solid #c084fc;}

  .tag-pos{background:#0d2018;color:#4ade80;padding:6px 16px;border-radius:20px;font-weight:700;border:1px solid #4ade80;font-size:1.05rem;}
  .tag-neg{background:#2b0d15;color:#f87171;padding:6px 16px;border-radius:20px;font-weight:700;border:1px solid #f87171;font-size:1.05rem;}
  .tag-neu{background:#1a1f2e;color:#94a3b8;padding:6px 16px;border-radius:20px;font-weight:700;border:1px solid #475569;font-size:1.05rem;}

  .result-card{background:#0d1117;border:1px solid #1e293b;border-radius:14px;padding:1.4rem 1.6rem;margin-top:1rem;}
  .result-card h3{color:#94a3b8;font-size:.75rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;margin-bottom:.8rem;}
  .conf-note{color:#475569;font-size:.85rem;margin-top:.7rem;}
  .conf-note b{color:#cbd5e1;}

  .model-card{background:linear-gradient(135deg,#0a0f1a,#0d1628);border:1px solid #1e3a5f;border-radius:14px;padding:1.2rem 1.5rem;margin-top:.8rem;}
  .model-card h4{color:#60a5fa;margin:0 0 .8rem;font-size:.95rem;font-weight:700;}
  .model-row{display:flex;justify-content:space-between;padding:.35rem 0;border-bottom:1px solid #1e293b;font-size:.83rem;}
  .model-row:last-child{border-bottom:none;}
  .model-row .mk{color:#64748b;} .model-row .mv{color:#e2e8f0;font-weight:500;}

  .aspect-wrap{background:#0d1117;border:1px solid #1e293b;border-radius:14px;padding:1.3rem 1.5rem;margin-top:1rem;}
  .aspect-wrap h3{color:#94a3b8;font-size:.75rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;margin-bottom:1rem;}
  .aspect-row{display:flex;align-items:center;justify-content:space-between;padding:.4rem 0;border-bottom:1px solid #0f172a;font-size:.87rem;}
  .aspect-row:last-child{border-bottom:none;}
  .aspect-kw{color:#e2e8f0;font-weight:500;}
  .asp-pos{background:#0d2018;color:#4ade80;padding:3px 10px;border-radius:12px;font-size:.78rem;font-weight:700;border:1px solid #4ade80;}
  .asp-neg{background:#2b0d15;color:#f87171;padding:3px 10px;border-radius:12px;font-size:.78rem;font-weight:700;border:1px solid #f87171;}
  .asp-neu{background:#1a1f2e;color:#94a3b8;padding:3px 10px;border-radius:12px;font-size:.78rem;font-weight:700;border:1px solid #475569;}
  .asp-conf{color:#475569;font-size:.78rem;}

  .report-card{background:#0a1628;border:1px solid #1e3a5f;border-radius:12px;padding:1.1rem 1.4rem;margin-top:.8rem;}
  .report-card h4{color:#fbbf24;margin:0 0 .7rem;font-size:.9rem;font-weight:600;}
  .report-card p{color:#8892a4;font-size:.82rem;margin:0;line-height:1.7;}

  .info-card{background:#0a1628;border:1px solid #1e3a5f;border-radius:12px;padding:1.2rem 1.4rem;margin-bottom:1rem;}
  .info-card h4{color:#60a5fa;margin:0 0 .5rem;font-size:.92rem;font-weight:600;}
  .info-card p{color:#8892a4;font-size:.83rem;line-height:1.7;margin:0;}

  .cv-block{background:#0d1117;border-left:3px solid #e94560;border-radius:0 10px 10px 0;padding:1.2rem 1.5rem;margin:1rem 0;font-family:'Space Mono',monospace;font-size:.82rem;color:#94a3b8;line-height:1.8;}
  .cv-block .cv-title{color:#e94560;font-weight:700;font-size:.9rem;}
  .cv-block .cv-tech{color:#60a5fa;margin-top:.5rem;}

  section[data-testid="stSidebar"]{background:#080d14 !important;}
  div[data-testid="stDownloadButton"] button{background:#e94560 !important;color:white !important;border:none !important;border-radius:8px !important;font-weight:600 !important;}

  [data-testid="metric-container"]{background:#0d1117;border:1px solid #1e293b;border-radius:10px;padding:.8rem 1rem;}
  [data-testid="metric-container"] label{color:#64748b !important;font-size:.78rem !important;}
  [data-testid="metric-container"] [data-testid="stMetricValue"]{color:#e2e8f0 !important;font-size:1.45rem !important;font-weight:700 !important;}

  button[data-baseweb="tab"][aria-selected="true"]{color:#e94560 !important;border-bottom:2px solid #e94560 !important;}
  button[data-baseweb="tab"]{color:#64748b !important;}
  .stSpinner > div > div{border-top-color:#e94560 !important;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE DATA
# ─────────────────────────────────────────────────────────────────────────────
QUICK_DEMO_TEXT = (
    "I absolutely love this new phone. The battery life is incredible, "
    "the camera is outstanding, and the price is very reasonable for what you get!"
)

SINGLE_EXAMPLES = {
    "😊 Positive":   "I absolutely love this product — best purchase I've made all year! Premium build and super fast delivery.",
    "😠 Negative":   "Absolutely terrible. Stopped working after 2 days, customer support was rude and completely unhelpful.",
    "😐 Neutral":    "The product is okay. Does what it claims, nothing more. No major complaints but nothing impressive either.",
    "📱 Tech Tweet": "Just updated to the latest Android version. New UI is clean and gestures feel very smooth.",
    "🍽️ Food Review":"Tried the new restaurant downtown. Food was average, service slow, but ambience was nice.",
}

SAMPLE_TWEETS = [
    "I absolutely love the new iPhone update! Super smooth and fast 🚀",
    "Tesla's build quality has really gone downhill. Very disappointing.",
    "Just finished reading Atomic Habits — genuinely life-changing book.",
    "Why is every app asking for unnecessary permissions? Ridiculous.",
    "The new Zomato update is actually pretty decent, no major complaints.",
    "Amazon delivery was 3 days late. No apology, no refund. Never again.",
    "ChatGPT just helped me debug a nasty production issue in 10 minutes!",
    "The movie was okay. Not great, not terrible. Just average.",
    "Swiggy delivered cold food for the third time this week. Furious.",
    "Weekend hiking trip with friends — best decision I've made this year!",
    "Public WiFi at this airport is absolutely unusable. Completely useless.",
    "New season of the show dropped and it's actually fire 🔥",
    "The gym was packed again today. Need to start going at 6 AM.",
    "Product packaging was damaged on arrival. Support was very helpful though.",
    "Team won the match last night. What a game! Absolutely historic.",
    "This new policy is absolutely ridiculous. Who approved this?",
    "Loving the new VS Code extension — saves me so much time every day.",
    "Worst hotel stay of my life. Dirty rooms, rude staff, broken AC.",
    "Had a pretty okay day. Nothing special really happened.",
    "The new MacBook battery life is insane — 18 hours easily!",
]

SAMPLE_REVIEWS = [
    ("Absolutely loved this product — best purchase I've made all year!",       "Electronics"),
    ("Terrible customer service. Waited 2 hours and got zero help.",             "Support"),
    ("Decent product overall. Does what it claims, nothing more.",               "Electronics"),
    ("The battery life is incredible. Easily lasts 2 full days.",                "Electronics"),
    ("Stopped working after 3 days. Total waste of money.",                      "Electronics"),
    ("Packaging was great and delivery was super fast. Very satisfied.",          "Shipping"),
    ("Interface is confusing and laggy. The UX needs a lot of work.",            "Software"),
    ("Highly recommend to everyone. Great value for the price.",                 "Electronics"),
    ("Not worth it at all. Cheaper alternatives work far better.",               "Electronics"),
    ("Pretty good for the price. Not perfect but definitely usable.",            "Electronics"),
    ("Broke after first use. Absolute garbage quality control.",                 "Electronics"),
    ("Works exactly as described in the listing. Happy customer.",               "Electronics"),
    ("Customer support was rude and unhelpful. Really bad experience.",          "Support"),
    ("Solid build quality and great design. Very happy with this.",              "Electronics"),
    ("Average product. Nothing special but no major issues either.",             "Electronics"),
    ("Shipping was incredibly fast! Got it within 24 hours.",                    "Shipping"),
    ("Completely different from the photos. False advertising.",                 "Electronics"),
    ("Great sound quality, easy setup. Exactly what I was looking for.",         "Electronics"),
    ("App keeps crashing on my phone. Very frustrating experience.",             "Software"),
    ("Exceeded all expectations! Will definitely buy from this brand again.",    "Electronics"),
]

STOPWORDS = {
    "the","a","an","is","in","it","to","and","or","of","was","for","that",
    "this","on","at","i","my","me","so","be","are","its","but","not","with",
    "have","had","has","just","really","very","been","they","you","we","he",
    "she","as","by","from","im","ive","get","got","do","did","our","their",
    "would","could","should","wont","dont","cant","after","before","also","ll",
    "ve","re","s","t","m","d",
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL — 3-tier (Transformer → sklearn → lexicon)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model…")
def load_model():
    try:
        from transformers import pipeline
        clf = pipeline("sentiment-analysis",
                       model="distilbert-base-uncased-finetuned-sst-2-english")
        return ("transformer", clf)
    except Exception:
        pass
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer
        pos = ["love this","amazing product","excellent quality","great experience",
               "very happy","wonderful","fantastic result","best ever",
               "highly recommend","perfect quality","super fast","exceeded expectations",
               "really impressed","absolutely brilliant","works flawlessly"]
        neg = ["hate this","terrible product","awful quality","worst experience",
               "very bad","horrible","dreadful","never again","waste of money",
               "poor quality","damaged","complete disappointment","really bad",
               "absolutely useless","stopped working","broke after"]
        neu = ["it was okay","not bad","average quality","nothing special",
               "could be better","decent enough","mediocre","just fine",
               "neither good nor bad","sort of okay","passable","tolerable",
               "does the job","so so","not great not terrible"]
        texts  = pos + neg + neu
        labels = [1]*len(pos) + [0]*len(neg) + [2]*len(neu)
        vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        X   = vec.fit_transform(texts)
        clf = LogisticRegression(max_iter=1000, C=2.0)
        clf.fit(X, labels)
        return ("sklearn", vec, clf)
    except Exception:
        pass
    return ("lexicon", None)


def predict(text: str, mp: tuple):
    text = text.strip()
    if not text:
        return "Neutral", {"Positive":0.33,"Negative":0.33,"Neutral":0.34}
    kind = mp[0]
    if kind == "transformer":
        res = mp[1](text[:512])[0]
        s   = res["score"]
        if res["label"] == "POSITIVE":
            return "Positive", {"Positive":round(s,3),"Negative":round((1-s)*.35,3),"Neutral":round((1-s)*.65,3)}
        return "Negative", {"Negative":round(s,3),"Positive":round((1-s)*.35,3),"Neutral":round((1-s)*.65,3)}
    if kind == "sklearn":
        vec,clf = mp[1],mp[2]
        X = vec.transform([text])
        probs = clf.predict_proba(X)[0]
        lmap  = {0:"Negative",1:"Positive",2:"Neutral"}
        conf  = {lmap[c]:round(float(p),3) for c,p in zip(clf.classes_,probs)}
        return lmap[clf.predict(X)[0]], conf
    POS={"love","great","good","amazing","excellent","wonderful","fantastic","best","happy",
         "perfect","brilliant","superb","outstanding","awesome","recommend","impressive",
         "satisfied","delighted","pleased","incredible","fast","smooth","clean","easy",
         "helpful","friendly","beautiful"}
    NEG={"hate","bad","terrible","awful","horrible","worst","poor","dreadful","useless",
         "disappointing","broken","damaged","waste","ridiculous","frustrating","annoying",
         "disgusting","pathetic","garbage","trash","slow","laggy","crashing","scam",
         "fake","rude","dirty","late"}
    words = re.findall(r"[a-z]+", text.lower())
    ps = sum(1 for w in words if w in POS)
    ns = sum(1 for w in words if w in NEG)
    if ps>ns: return "Positive",{"Positive":0.72,"Negative":0.10,"Neutral":0.18}
    if ns>ps: return "Negative",{"Negative":0.72,"Positive":0.10,"Neutral":0.18}
    return "Neutral",{"Neutral":0.60,"Positive":0.22,"Negative":0.18}


def run_batch(texts: list, mp: tuple):
    labels,confs = [],[]
    for t in texts:
        lbl,cf = predict(str(t),mp)
        labels.append(lbl); confs.append(round(max(cf.values())*100,1))
    return labels,confs


# ─────────────────────────────────────────────────────────────────────────────
# ASPECT-BASED SENTIMENT  ← new
# ─────────────────────────────────────────────────────────────────────────────
ASPECT_KW = {
    "Battery":     ["battery","charge","charging","power","drain","backup"],
    "Camera":      ["camera","photo","picture","image","lens","video","selfie"],
    "Price":       ["price","cost","value","expensive","cheap","worth","money","affordable"],
    "Display":     ["screen","display","brightness","resolution","panel","colour","color"],
    "Build":       ["build","quality","design","material","plastic","metal","premium"],
    "Delivery":    ["delivery","shipping","arrived","packaging","dispatched","courier"],
    "Support":     ["support","service","help","response","staff","team","refund","return"],
    "Software":    ["software","app","update","feature","bug","crash","interface","ui","ux"],
    "Sound":       ["sound","audio","speaker","headphone","earphone","noise","volume","bass"],
    "Performance": ["speed","performance","fast","slow","lag","processor","ram","powerful"],
}

def aspect_sentiment(text: str, mp: tuple) -> list:
    tl = text.lower()
    out = []
    for aspect, kws in ASPECT_KW.items():
        for kw in kws:
            if kw in tl:
                sents    = re.split(r"[.!?,;]", text)
                relevant = " ".join(s for s in sents if kw in s.lower()) or text
                lbl,cf   = predict(relevant, mp)
                out.append({"Aspect":aspect,"Keyword":kw,
                             "Sentiment":lbl,"Confidence":round(max(cf.values())*100,1)})
                break
    return out


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATOR  ← new
# ─────────────────────────────────────────────────────────────────────────────
def generate_report(df: pd.DataFrame, source: str="Dataset") -> bytes:
    buf  = io.StringIO()
    now  = datetime.now().strftime("%Y-%m-%d %H:%M")
    total = len(df)
    buf.write(f"AI Sentiment Intelligence Studio — Sentiment Report\n")
    buf.write(f"Generated:,{now}\nSource:,{source}\nTotal Records:,{total}\n\n")
    buf.write("=== SUMMARY METRICS ===\nSentiment,Count,Percentage\n")
    for lbl in ["Positive","Negative","Neutral"]:
        cnt = (df["Sentiment"]==lbl).sum()
        buf.write(f"{lbl},{cnt},{cnt/total*100:.1f}%\n")
    if "Confidence (%)" in df.columns:
        buf.write(f"\nAverage Confidence,{df['Confidence (%)'].mean():.1f}%\n")
        buf.write(f"Min Confidence,{df['Confidence (%)'].min():.1f}%\n")
        buf.write(f"Max Confidence,{df['Confidence (%)'].max():.1f}%\n")
    buf.write("\n=== FULL RESULTS ===\n")
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {"Positive":"#4ade80","Negative":"#f87171","Neutral":"#94a3b8"}
_BG = "rgba(0,0,0,0)"; _F = "#94a3b8"
_T  = lambda t: dict(text=t,font=dict(color="#e2e8f0",size=14))

def chart_confidence(conf):
    fig = go.Figure(go.Bar(
        x=list(conf.keys()), y=[v*100 for v in conf.values()],
        marker_color=[COLORS.get(k,"#888") for k in conf.keys()],
        text=[f"{v*100:.1f}%" for v in conf.values()], textposition="outside",
    ))
    fig.update_layout(paper_bgcolor=_BG,plot_bgcolor=_BG,font_color=_F,height=260,
                      yaxis=dict(range=[0,115],showgrid=False,zeroline=False),
                      xaxis=dict(showgrid=False,tickfont=dict(size=13,color="#cbd5e1")),
                      margin=dict(t=20,b=10,l=10,r=10),title=_T("Confidence Scores"))
    return fig

def chart_pie(df):
    c = df["Sentiment"].value_counts().reset_index(); c.columns=["Sentiment","Count"]
    fig = px.pie(c,values="Count",names="Sentiment",color="Sentiment",
                 color_discrete_map=COLORS,hole=0.42)
    fig.update_traces(textfont_size=13,textinfo="percent+label",
                      marker=dict(line=dict(color="#0d1117",width=2)))
    fig.update_layout(paper_bgcolor=_BG,font_color=_F,height=300,
                      legend=dict(font=dict(color=_F)),title=_T("Sentiment Distribution"))
    return fig

def chart_bar(df):
    c = df["Sentiment"].value_counts().reset_index(); c.columns=["Sentiment","Count"]
    fig = px.bar(c,x="Sentiment",y="Count",color="Sentiment",
                 color_discrete_map=COLORS,text="Count")
    fig.update_traces(textposition="outside")
    fig.update_layout(paper_bgcolor=_BG,plot_bgcolor=_BG,font_color=_F,height=280,
                      showlegend=False,xaxis=dict(showgrid=False),
                      yaxis=dict(showgrid=False,zeroline=False),
                      margin=dict(t=30,b=10),title=_T("Count by Sentiment"))
    return fig

def chart_timeline(df):
    df = df.copy()
    if "time" not in df.columns:
        df["time"] = pd.date_range(start="2024-01-01",periods=len(df),freq="h")
    df["score"]   = df["Sentiment"].map({"Positive":1,"Neutral":0,"Negative":-1}).fillna(0)
    df["rolling"] = df["score"].rolling(window=max(1,len(df)//6),min_periods=1).mean()
    fig = px.area(df,x="time",y="rolling",color_discrete_sequence=["#e94560"])
    fig.update_traces(line_width=2,fillcolor="rgba(233,69,96,.12)")
    fig.update_layout(paper_bgcolor=_BG,plot_bgcolor=_BG,font_color=_F,height=280,
                      xaxis=dict(showgrid=False,title=""),
                      yaxis=dict(range=[-1.3,1.3],title="Score",gridcolor="#1e293b",
                                 zeroline=True,zerolinecolor="#334155"),
                      margin=dict(t=30,b=10),title=_T("Sentiment Trend Over Time"))
    return fig

def chart_wordfreq(text_series, top_n=15):
    words  = " ".join(text_series.fillna("").apply(_clean)).split()
    words  = [w for w in words if w not in STOPWORDS and len(w)>2]
    common = Counter(words).most_common(top_n)
    if not common: return None
    lbs,vals = zip(*common)
    fig = px.bar(x=list(vals),y=list(lbs),orientation="h",
                 color=list(vals),color_continuous_scale="RdYlGn",text=list(vals))
    fig.update_traces(textposition="outside")
    fig.update_layout(paper_bgcolor=_BG,plot_bgcolor=_BG,font_color=_F,
                      height=420,showlegend=False,coloraxis_showscale=False,
                      xaxis=dict(showgrid=False),yaxis=dict(autorange="reversed",showgrid=False),
                      margin=dict(l=110,t=30),title=_T(f"Top {top_n} Keywords"))
    return fig

def make_wordcloud(text_series):
    words = " ".join(text_series.fillna("").apply(_clean)).split()
    words = [w for w in words if w not in STOPWORDS and len(w)>2]
    if len(words)<5: return None
    wc  = WordCloud(width=900,height=360,background_color="#0d1117",
                    colormap="RdYlGn",max_words=120,collocations=False).generate(" ".join(words))
    fig,ax = plt.subplots(figsize=(11,4.2))
    ax.imshow(wc,interpolation="bilinear"); ax.axis("off")
    fig.patch.set_facecolor("#0d1117"); fig.tight_layout(pad=0)
    return fig

def _clean(t):
    t = re.sub(r"http\S+|www\S+","",t)
    t = re.sub(r"[@#]\w+","",t)
    t = re.sub(r"[^a-zA-Z\s]","",t)
    return t.strip().lower()

def tag_class(s): return {"Positive":"tag-pos","Negative":"tag-neg"}.get(s,"tag-neu")

def show_metrics(df, ncols=4):
    total = len(df)
    pos=(df["Sentiment"]=="Positive").sum()
    neg=(df["Sentiment"]=="Negative").sum()
    neu=(df["Sentiment"]=="Neutral").sum()
    if ncols==5 and "Confidence (%)" in df.columns:
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("📝 Total",           total)
        m2.metric("😊 Positive",        f"{pos} ({pos/total*100:.0f}%)")
        m3.metric("😠 Negative",        f"{neg} ({neg/total*100:.0f}%)")
        m4.metric("😐 Neutral",         f"{neu} ({neu/total*100:.0f}%)")
        m5.metric("🎯 Avg Confidence",  f"{df['Confidence (%)'].mean():.1f}%")
    else:
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("📝 Total",    total)
        m2.metric("😊 Positive", f"{pos} ({pos/total*100:.0f}%)")
        m3.metric("😠 Negative", f"{neg} ({neg/total*100:.0f}%)")
        m4.metric("😐 Neutral",  f"{neu} ({neu/total*100:.0f}%)")

def _report_download(df, label, fname):
    """Render report card + two download buttons."""
    st.markdown(f"""
<div class="report-card">
  <h4>📄 Download Sentiment Report</h4>
  <p>Full report = executive summary metrics (total, %, avg confidence)
     + all classified rows in one CSV file.</p>
</div>""", unsafe_allow_html=True)
    r1,r2 = st.columns(2)
    r1.download_button("📄 Download Full Report",
                       generate_report(df,label),
                       fname,"text/csv",use_container_width=True)
    r2.download_button("⬇️ Raw Results CSV",
                       df.to_csv(index=False).encode(),
                       fname.replace("report_","raw_"),"text/csv",
                       use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k,v in {"single_text":"","csv_results":None,"tweet_df":None,"demo_result":None}.items():
    if k not in st.session_state: st.session_state[k]=v

model_pack = load_model()
model_kind = model_pack[0]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Model Status")
    if model_kind=="transformer":
        st.success("🤗 DistilBERT — Active")
        st.markdown("""
| | |
|---|---|
|**Model**|DistilBERT-SST2|
|**Dataset**|Stanford SST-2|
|**Accuracy**|~91%|
|**Type**|Transformer|
|**Labels**|Positive / Negative|
""")
    elif model_kind=="sklearn":
        st.info("📐 Logistic Regression — Active")
    else:
        st.warning("🔧 Lexicon fallback — Active")

    st.divider()
    st.markdown("### ⚙️ Display Settings")
    show_conf      = st.toggle("Confidence chart",  value=True)
    show_aspect    = st.toggle("Aspect analysis",   value=True)
    show_pie       = st.toggle("Pie chart",         value=True)
    show_bar       = st.toggle("Bar chart",         value=True)
    show_timeline  = st.toggle("Trend timeline",    value=True)
    show_wordfreq  = st.toggle("Word frequency",    value=True)
    show_wordcloud = st.toggle("Word cloud",        value=True)

    st.divider()
    st.markdown("### 📖 About")
    st.markdown("""
<div class="info-card">
  <h4>AI Sentiment Intelligence Studio</h4>
  <p>NLP dashboard for real-time text sentiment.<br>
  DistilBERT transformer + scikit-learn fallback.<br><br>
  <b style="color:#60a5fa;">Use cases:</b><br>
  • Brand & social media monitoring<br>
  • Customer feedback analytics<br>
  • Product review classification</p>
</div>""", unsafe_allow_html=True)
    st.caption("Built by **Ojas Waykole** · GCE Jalgaon")


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-title">🧠 AI Sentiment Intelligence Studio</div>
  <p class="hero-sub">
    Real-time NLP sentiment analysis for text, social media, and large datasets.<br>
    Powered by HuggingFace Transformers · Deployed on HuggingFace Spaces.
  </p>
  <div class="hero-badges">
    <span class="b-red">🔍 Text Analysis</span>
    <span class="b-blue">🐦 Social Media</span>
    <span class="b-green">📂 Dataset Upload</span>
    <span class="b-gold">📊 Analytics</span>
    <span class="b-purple">💼 Portfolio Project</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ✅ IMPROVEMENT 2 — QUICK DEMO STRIP
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("#### ⚡ Quick Demo — try the AI in one click")
qd1,qd2,qd3,qd4 = st.columns([2.5,1,1,1])
qd1.caption("No typing needed — load any example instantly and see live AI prediction.")
run_demo = qd2.button("⚡ Analyze Example",  use_container_width=True)
load_csv = qd3.button("📂 Load Sample CSV",  use_container_width=True)
load_twt = qd4.button("🐦 Load Tweets",      use_container_width=True)

if run_demo:
    with st.spinner("Running AI…"):
        lbl,cf = predict(QUICK_DEMO_TEXT,model_pack)
    st.session_state["demo_result"] = (QUICK_DEMO_TEXT,lbl,cf)

if load_csv:
    texts,cats = zip(*SAMPLE_REVIEWS)
    with st.spinner("Loading sample CSV…"):
        labels,confs = run_batch(list(texts),model_pack)
    st.session_state["csv_results"] = pd.DataFrame(
        {"Text":list(texts),"Category":list(cats),"Sentiment":labels,"Confidence (%)":confs})
    st.success("✅ Sample dataset loaded — open **Dataset Analyzer** tab.")

if load_twt:
    with st.spinner("Analyzing sample tweets…"):
        labels,confs = run_batch(SAMPLE_TWEETS,model_pack)
    st.session_state["tweet_df"] = pd.DataFrame(
        {"Tweet":SAMPLE_TWEETS,"Sentiment":labels,"Confidence (%)":confs})
    st.success("✅ Sample tweets loaded — open **Social Media Analyzer** tab.")

if st.session_state["demo_result"]:
    demo_text,lbl,cf = st.session_state["demo_result"]
    st.markdown(f"""
<div class="result-card">
  <h3>⚡ Quick Demo Result</h3>
  <p style="color:#94a3b8;font-size:.88rem;margin-bottom:.7rem;font-style:italic;">"{demo_text}"</p>
  <span class="{tag_class(lbl)}">{lbl}</span>
  <p class="conf-note">Confidence: <b>{max(cf.values())*100:.1f}%</b>
     &nbsp;·&nbsp; Model: <b>{model_kind}</b></p>
</div>""", unsafe_allow_html=True)
    if show_conf:
        st.plotly_chart(chart_confidence(cf), use_container_width=True)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# ✅ IMPROVEMENT 1 — RENAMED TABS (SaaS-style)
# ─────────────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "🔍 Text Analysis",
    "🐦 Social Media Analyzer",
    "📂 Dataset Analyzer",
    "📊 Analytics Dashboard",
    "👨‍💻 About the Project",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TEXT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🔍 Text Analysis")
    st.caption("Analyze any sentence, review, or paragraph — with aspect-level breakdown.")

    # ✅ IMPROVEMENT 3 — MODEL EXPLANATION CARD
    with st.expander("🧠 How the AI Model Works", expanded=False):
        if model_kind=="transformer":
            st.markdown("""
<div class="model-card">
  <h4>🤗 DistilBERT Transformer — Active</h4>
  <div class="model-row"><span class="mk">Model</span><span class="mv">distilbert-base-uncased-finetuned-sst-2-english</span></div>
  <div class="model-row"><span class="mk">Architecture</span><span class="mv">Transformer (BERT-family, 6 attention layers)</span></div>
  <div class="model-row"><span class="mk">Training Dataset</span><span class="mv">Stanford Sentiment Treebank v2 (SST-2)</span></div>
  <div class="model-row"><span class="mk">Accuracy</span><span class="mv">~91% on SST-2 benchmark</span></div>
  <div class="model-row"><span class="mk">Speed vs BERT</span><span class="mv">60% faster · 40% smaller · 97% of BERT's understanding</span></div>
  <div class="model-row"><span class="mk">Task</span><span class="mv">Binary sentiment classification → Positive / Negative</span></div>
  <div class="model-row"><span class="mk">Max Input</span><span class="mv">512 tokens (~400 words)</span></div>
  <div class="model-row"><span class="mk">Source</span><span class="mv">HuggingFace Model Hub (open-source)</span></div>
</div>""", unsafe_allow_html=True)
        elif model_kind=="sklearn":
            st.markdown("""
<div class="model-card">
  <h4>📐 Logistic Regression + TF-IDF — Active</h4>
  <div class="model-row"><span class="mk">Vectorizer</span><span class="mv">TF-IDF with bigrams (1,2)</span></div>
  <div class="model-row"><span class="mk">Classifier</span><span class="mv">Logistic Regression (C=2.0, max_iter=1000)</span></div>
  <div class="model-row"><span class="mk">Labels</span><span class="mv">Positive / Negative / Neutral</span></div>
  <div class="model-row"><span class="mk">Why active</span><span class="mv">Transformers unavailable in this environment</span></div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="model-card">
  <h4>🔧 Lexicon Fallback — Active</h4>
  <div class="model-row"><span class="mk">Method</span><span class="mv">Keyword matching on curated POS/NEG word lists</span></div>
  <div class="model-row"><span class="mk">Labels</span><span class="mv">Positive / Negative / Neutral</span></div>
</div>""", unsafe_allow_html=True)

    st.markdown("**Try an example:**")
    ex_cols = st.columns(len(SINGLE_EXAMPLES))
    for col,(lbl,txt) in zip(ex_cols, SINGLE_EXAMPLES.items()):
        if col.button(lbl, use_container_width=True):
            st.session_state["single_text"] = txt

    user_text = st.text_area(
        "Enter your text:", height=130, key="single_text",
        placeholder="Type a review, tweet, feedback, or any sentence…",
    )

    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if user_text.strip():
            with st.spinner("Running AI model…"):
                label,conf = predict(user_text, model_pack)
            st.markdown(f"""
<div class="result-card">
  <h3>Sentiment Result</h3>
  <span class="{tag_class(label)}">{label}</span>
  <p class="conf-note">Confidence: <b>{max(conf.values())*100:.1f}%</b>
     &nbsp;·&nbsp; Model: <b>{model_kind}</b>
     &nbsp;·&nbsp; Characters: <b>{len(user_text)}</b></p>
</div>""", unsafe_allow_html=True)
            if show_conf:
                st.plotly_chart(chart_confidence(conf), use_container_width=True)
            if show_aspect:
                aspects = aspect_sentiment(user_text, model_pack)
                if aspects:
                    rows = "".join(
                        f"""<div class="aspect-row">
                              <span class="aspect-kw">🔹 {a['Aspect']}</span>
                              <span class="asp-{a['Sentiment'].lower()}">{a['Sentiment']}</span>
                              <span class="asp-conf">{a['Confidence']}% confidence</span>
                            </div>"""
                        for a in aspects
                    )
                    st.markdown(f"""
<div class="aspect-wrap">
  <h3>🎯 Aspect-Based Sentiment Analysis</h3>
  {rows}
</div>""", unsafe_allow_html=True)
                else:
                    st.caption("ℹ️ No product/service aspects detected in this text.")
        else:
            st.warning("Please enter some text to analyze.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SOCIAL MEDIA ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🐦 Social Media Analyzer")
    st.caption("Paste up to 200 tweets or social posts — one per line. See distribution, word cloud, and download a report.")

    c1,c2 = st.columns([1,3])
    if c1.button("📥 Load Sample Tweets", use_container_width=True):
        st.session_state["tweet_input_text"] = "\n".join(SAMPLE_TWEETS)
    if c2.button("🗑️ Clear", use_container_width=True):
        st.session_state["tweet_input_text"] = ""

    tweet_input = st.text_area(
        "Paste posts here — one per line:", height=200,
        key="tweet_input_text",
        placeholder="I love the new iPhone update!\nTesla's quality has gone downhill.\nHad an okay day — nothing special.",
    )

    if st.button("🐦 Analyze All Posts", type="primary", use_container_width=True):
        tweets = [t.strip() for t in (tweet_input or "").strip().splitlines() if t.strip()]
        if tweets:
            with st.spinner(f"Analyzing {len(tweets)} posts…"):
                labels,confs = run_batch(tweets,model_pack)
            st.session_state["tweet_df"] = pd.DataFrame(
                {"Tweet":tweets,"Sentiment":labels,"Confidence (%)":confs})
        else:
            st.warning("Please paste at least one post.")

    if st.session_state["tweet_df"] is not None:
        df = st.session_state["tweet_df"]
        st.success(f"✅ Analyzed **{len(df)}** posts")
        show_metrics(df, ncols=5)
        st.markdown("")

        if show_pie and show_bar:
            cl,cr = st.columns(2)
            with cl: st.plotly_chart(chart_pie(df), use_container_width=True)
            with cr: st.plotly_chart(chart_bar(df), use_container_width=True)
        elif show_pie: st.plotly_chart(chart_pie(df), use_container_width=True)
        elif show_bar: st.plotly_chart(chart_bar(df), use_container_width=True)

        if show_wordcloud:
            with st.spinner("Generating word cloud…"):
                wc_fig = make_wordcloud(df["Tweet"])
                if wc_fig:
                    st.markdown("**☁️ Word Cloud**")
                    st.pyplot(wc_fig, use_container_width=True)

        st.markdown("##### 📋 Results Table")
        filt = st.multiselect("Filter:", ["Positive","Negative","Neutral"],
                              default=["Positive","Negative","Neutral"], key="twt_filt")
        st.dataframe(df[df["Sentiment"].isin(filt)], use_container_width=True, hide_index=True)
        # ✅ IMPROVEMENT 4 — DOWNLOAD REPORT
        _report_download(df, "Social Media Posts", "sentiment_report_social.csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATASET ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📂 Dataset Analyzer")
    st.caption("Upload any CSV with a text column. The AI classifies every row and returns an enriched dataset.")

    with st.expander("📌 Expected CSV format", expanded=False):
        st.markdown("""
**Minimum requirement:** one column with text.

| review_text | category | rating |
|---|---|---|
| Great product, fast delivery! | Electronics | 5 |
| Stopped working after 2 days. | Electronics | 1 |
| Decent, nothing special. | Apparel | 3 |

The app adds `Sentiment` and `Confidence (%)` columns automatically.
""")

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded:
        raw_df = pd.read_csv(uploaded)
        st.info(f"✅ Loaded **{len(raw_df)} rows** × **{len(raw_df.columns)} columns**")
        st.dataframe(raw_df.head(5), use_container_width=True)
        text_col = st.selectbox("Select the text column:", raw_df.columns.tolist())
        max_rows = st.slider("Max rows to analyze:",
                             10, min(500,len(raw_df)), min(100,len(raw_df)))
        if st.button("📊 Run Analysis", type="primary", use_container_width=True):
            subset = raw_df[[text_col]].dropna().head(max_rows).copy()
            subset.columns = ["Text"]
            with st.spinner(f"Classifying {len(subset)} rows…"):
                labels,confs = run_batch(subset["Text"].tolist(),model_pack)
            subset["Sentiment"]     = labels
            subset["Confidence (%)"]= confs
            st.session_state["csv_results"] = subset
            st.success(f"✅ Done — {len(subset)} rows classified.")
    else:
        st.markdown("---")
        st.markdown("##### No file? Run the built-in demo (20 product reviews)")
        if st.button("▶️ Load Sample Reviews + Run Demo", use_container_width=True):
            texts,cats = zip(*SAMPLE_REVIEWS)
            with st.spinner("Running demo…"):
                labels,confs = run_batch(list(texts),model_pack)
            st.session_state["csv_results"] = pd.DataFrame(
                {"Text":list(texts),"Category":list(cats),"Sentiment":labels,"Confidence (%)":confs})
            st.success("✅ Demo loaded!")

    if st.session_state["csv_results"] is not None:
        df = st.session_state["csv_results"]
        st.markdown("---")
        show_metrics(df, ncols=5)
        st.markdown("")

        cl,cr = st.columns(2)
        with cl:
            if show_pie: st.plotly_chart(chart_pie(df), use_container_width=True)
        with cr:
            if show_bar: st.plotly_chart(chart_bar(df), use_container_width=True)

        if show_timeline:
            st.plotly_chart(chart_timeline(df), use_container_width=True)
        if show_wordfreq:
            fig_wf = chart_wordfreq(df["Text"])
            if fig_wf: st.plotly_chart(fig_wf, use_container_width=True)
        if show_wordcloud:
            with st.spinner("Generating word cloud…"):
                wc_fig = make_wordcloud(df["Text"])
                if wc_fig:
                    st.markdown("**☁️ Word Cloud**")
                    st.pyplot(wc_fig, use_container_width=True)

        st.markdown("##### 📋 Full Results")
        st.dataframe(df, use_container_width=True, hide_index=True)
        # ✅ IMPROVEMENT 4 — DOWNLOAD REPORT
        _report_download(df, "CSV Dataset", "sentiment_report_dataset.csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📊 Analytics Dashboard")
    st.caption("Deep-dive visual analytics — charts, trends, keywords, word cloud, and filtered export.")

    source = st.session_state["csv_results"] or st.session_state["tweet_df"]

    if source is None:
        st.markdown("""
<div class="info-card">
  <h4>💡 No data loaded</h4>
  <p>Run an analysis in <b>Dataset Analyzer</b> or <b>Social Media Analyzer</b> first,
     or use the <b>Quick Demo</b> buttons above the tabs.</p>
</div>""", unsafe_allow_html=True)
    else:
        df = source.copy()

        # ✅ BONUS — Metrics always above charts
        show_metrics(df, ncols=5)
        st.markdown("")

        cl,cr = st.columns(2)
        with cl:
            if show_pie: st.plotly_chart(chart_pie(df), use_container_width=True)
        with cr:
            if show_bar: st.plotly_chart(chart_bar(df), use_container_width=True)

        if show_timeline:
            st.plotly_chart(chart_timeline(df), use_container_width=True)

        text_col = "Text" if "Text" in df.columns else "Tweet"
        cl2,cr2  = st.columns(2)
        with cl2:
            if show_wordfreq:
                fig_wf = chart_wordfreq(df[text_col])
                if fig_wf: st.plotly_chart(fig_wf, use_container_width=True)
        with cr2:
            if show_wordcloud:
                wc_fig = make_wordcloud(df[text_col])
                if wc_fig: st.pyplot(wc_fig, use_container_width=True)

        st.markdown("##### 🔎 Filter & Export")
        filt = st.multiselect("Filter by sentiment:",
                              ["Positive","Negative","Neutral"],
                              default=["Positive","Negative","Neutral"],
                              key="analytics_filter")
        filtered = df[df["Sentiment"].isin(filt)]
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        # ✅ IMPROVEMENT 4 — DOWNLOAD REPORT
        _report_download(filtered, "Analytics Dashboard", "sentiment_report_analytics.csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT THE PROJECT
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("👨‍💻 About the Project")

    st.markdown("### 🧠 AI Model Information")
    if model_kind=="transformer":
        st.markdown("""
<div class="model-card">
  <h4>🤗 DistilBERT Transformer — Currently Active</h4>
  <div class="model-row"><span class="mk">Model</span><span class="mv">distilbert-base-uncased-finetuned-sst-2-english</span></div>
  <div class="model-row"><span class="mk">Architecture</span><span class="mv">Transformer (BERT-family, 6 attention layers)</span></div>
  <div class="model-row"><span class="mk">Training Dataset</span><span class="mv">Stanford Sentiment Treebank v2 (SST-2)</span></div>
  <div class="model-row"><span class="mk">Benchmark Accuracy</span><span class="mv">~91% on SST-2</span></div>
  <div class="model-row"><span class="mk">Speed vs BERT</span><span class="mv">60% faster · 40% fewer parameters · 97% performance retained</span></div>
  <div class="model-row"><span class="mk">Task</span><span class="mv">Binary Sentiment Classification (Positive / Negative)</span></div>
  <div class="model-row"><span class="mk">Max Token Limit</span><span class="mv">512 tokens (~400 words)</span></div>
  <div class="model-row"><span class="mk">Source</span><span class="mv">HuggingFace Model Hub (open-source)</span></div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📄 CV / Resume Entry")
    st.markdown("""
<div class="cv-block">
  <div class="cv-title">AI Sentiment Intelligence Dashboard &nbsp;|&nbsp; Live Demo ↗</div>
  <br>
  • Built a production-grade NLP sentiment analysis platform using Python and Machine Learning<br>
  • Implemented DistilBERT transformer (HuggingFace) with scikit-learn Logistic Regression fallback<br>
  • Developed a full-stack interactive web app using Streamlit with 4 analysis modes<br>
  • Added aspect-based sentiment analysis — identifies sentiment per product/service dimension<br>
  • Integrated visual analytics: pie/bar charts, sentiment timeline, word cloud, keyword frequency<br>
  • Built downloadable sentiment reports with executive summary metrics (count, %, confidence)<br>
  • Deployed publicly on HuggingFace Spaces — instantly accessible and testable by anyone<br>
  <div class="cv-tech">Tech: Python · HuggingFace Transformers · Streamlit · Plotly · Scikit-learn · Pandas</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ✅ Skills Demonstrated")
    ca,cb = st.columns(2)
    with ca:
        st.markdown("""
**ML & NLP**
- ✅ Pre-trained transformer (DistilBERT)
- ✅ TF-IDF + Logistic Regression NLP
- ✅ Aspect-based sentiment analysis
- ✅ 3-tier model fallback logic
- ✅ Batch inference pipeline

**Data Engineering**
- ✅ CSV ingestion with Pandas
- ✅ Dynamic column detection
- ✅ Multi-section report generation
- ✅ Filtered CSV export
""")
    with cb:
        st.markdown("""
**Web App / Frontend**
- ✅ Streamlit multi-tab dashboard
- ✅ Custom CSS dark neon theme
- ✅ Session state management
- ✅ Interactive Plotly charts
- ✅ Quick demo strip in hero

**DevOps & Deployment**
- ✅ Public deploy — HuggingFace Spaces
- ✅ Zero-setup for recruiters (live URL)
- ✅ requirements.txt reproducibility
- ✅ Graceful model degradation
""")

    st.markdown("---")
    st.markdown("### 👨‍💻 Author")
    st.markdown("""
**Ojas Waykole** — 2nd Year B.Tech CSE  
Government College of Engineering, Jalgaon (NMU University) · Maharashtra, India

🔗 [HuggingFace Space](https://huggingface.co/spaces/OjasWaykole/ai-sentiment-dashboard)  
💼 Open to **AI/ML & NLP Internship Opportunities** · May–December
""")
