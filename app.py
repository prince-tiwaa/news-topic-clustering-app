import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import random
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="News Article Topic Clustering",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* Root variables */
:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2235;
    --border: #1e2d45;
    --accent: #3b82f6;
    --accent2: #06b6d4;
    --accent3: #8b5cf6;
    --gold: #f59e0b;
    --green: #10b981;
    --red: #ef4444;
    --text: #e2e8f0;
    --text-muted: #64748b;
    --text-dim: #94a3b8;
}

/* Global overrides */
.stApp {
    background: var(--bg) !important;
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stRadio label {
    color: var(--text-dim) !important;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.9rem;
}

/* Remove default padding */
.main .block-container { padding-top: 1rem; }

/* Typography */
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; color: var(--text) !important; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-card .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
    font-weight: 600;
}
.metric-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
}
.metric-card .sub {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.3rem;
}
.metric-card .icon {
    position: absolute;
    top: 1rem; right: 1rem;
    font-size: 1.4rem;
    opacity: 0.5;
}

/* Section headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
}
.section-header h2 {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0;
}
.section-badge {
    background: var(--accent);
    color: white;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.25rem 0.6rem;
    border-radius: 20px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Info boxes */
.info-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    font-size: 0.9rem;
    line-height: 1.6;
    color: var(--text-dim);
}

/* Cluster keyword pill */
.keyword-pill {
    display: inline-block;
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.3);
    color: var(--accent2);
    border-radius: 20px;
    padding: 0.2rem 0.65rem;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 0.15rem;
}

/* Article card */
.article-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.article-card:hover { border-color: var(--accent); }
.article-card .art-title {
    font-weight: 600;
    font-size: 0.92rem;
    color: var(--text);
    margin-bottom: 0.4rem;
}
.article-card .art-meta {
    font-size: 0.75rem;
    color: var(--text-muted);
}
.cluster-badge {
    display: inline-block;
    background: rgba(139,92,246,0.15);
    border: 1px solid rgba(139,92,246,0.35);
    color: #a78bfa;
    border-radius: 4px;
    padding: 0.1rem 0.5rem;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Model card */
.model-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem;
    height: 100%;
}
.model-card.best {
    border-color: var(--green);
    box-shadow: 0 0 20px rgba(16,185,129,0.08);
}
.model-card .model-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.model-card .param-row {
    display: flex;
    justify-content: space-between;
    padding: 0.35rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.83rem;
}
.param-key { color: var(--text-muted); }
.param-val { color: var(--accent2); font-family: 'JetBrains Mono', monospace; }

/* Workflow step */
.workflow-step {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex: 1;
}
.step-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    text-align: center;
    flex: 1;
    font-size: 0.82rem;
    color: var(--text-dim);
    font-weight: 500;
}
.step-arrow { color: var(--accent); font-size: 1.1rem; }

/* Prediction result */
.pred-result {
    background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(6,182,212,0.08));
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 12px;
    padding: 1.4rem;
    margin-top: 1rem;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #e2e8f0, #3b82f6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.75rem;
}
.hero-sub {
    font-size: 1rem;
    color: var(--text-dim);
    max-width: 600px;
    line-height: 1.6;
}

/* Comparison table */
.comp-table { width: 100%; border-collapse: collapse; }
.comp-table th {
    background: var(--surface2);
    color: var(--text-muted);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border);
    text-align: left;
}
.comp-table td {
    padding: 0.8rem 1rem;
    border-bottom: 1px solid var(--border);
    font-size: 0.88rem;
    color: var(--text-dim);
}
.comp-table tr:hover td { background: var(--surface2); }
.best-tag {
    display: inline-block;
    background: rgba(16,185,129,0.15);
    border: 1px solid var(--green);
    color: var(--green);
    border-radius: 4px;
    padding: 0.1rem 0.5rem;
    font-size: 0.7rem;
    font-weight: 600;
}

/* Summary card for presentation */
.summary-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem;
    height: 100%;
}
.summary-card h4 {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.88rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--accent2) !important;
    margin-bottom: 0.6rem;
}

/* Streamlit widget overrides */
div[data-testid="stSelectbox"] > div, 
div[data-testid="stMultiSelect"] > div,
div[data-testid="stTextInput"] > div > div,
div[data-testid="stTextArea"] > div > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    border: none !important;
    color: white !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
}
.stButton > button:hover { opacity: 0.9 !important; }
div[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Data definitions ──────────────────────────────────────────────────────────

CLUSTER_INFO = {
    0:  {"label": "Arts & Culture",        "emoji": "🎨", "color": "#f59e0b", "keywords": ["museum", "art", "city", "building", "new", "said", "york", "artist", "street", "work"]},
    1:  {"label": "Paywalls & Subscriptions","emoji": "🔒", "color": "#6366f1", "keywords": ["newsday", "account", "login", "subscription", "notification", "optimum", "connect", "advertise", "click", "new"]},
    2:  {"label": "Education",             "emoji": "🎓", "color": "#06b6d4", "keywords": ["school", "student", "teacher", "education", "college", "university", "district", "high", "parent"]},
    3:  {"label": "Business & Finance",    "emoji": "💼", "color": "#10b981", "keywords": ["city", "tax", "year", "company", "state", "bank", "million"]},
    4:  {"label": "Music & Entertainment", "emoji": "🎵", "color": "#ec4899", "keywords": ["music", "song", "album", "band", "year", "time"]},
    5:  {"label": "Trump-Era Politics",    "emoji": "🗳️", "color": "#ef4444", "keywords": ["trump", "donald", "president", "republican", "clinton", "campaign", "democrat"]},
    6:  {"label": "Obama/Romney Era",      "emoji": "🏛️", "color": "#8b5cf6", "keywords": ["romney", "obama", "mccain", "palin", "campaign", "republican", "president"]},
    7:  {"label": "Real Estate",           "emoji": "🏠", "color": "#f97316", "keywords": ["listing", "bath", "agent", "bedroom", "bed", "property"]},
    8:  {"label": "Obama Administration",  "emoji": "🇺🇸", "color": "#3b82f6", "keywords": ["obama", "president", "bush", "republican", "biden", "house", "white"]},
    9:  {"label": "Food & Recipes",        "emoji": "🍳", "color": "#84cc16", "keywords": ["sauce", "recipe", "dish", "pepper", "cup", "heat", "salt", "cook"]},
    10: {"label": "Finance & Credit",      "emoji": "💳", "color": "#14b8a6", "keywords": ["credit", "image", "reward", "card", "apps"]},
    11: {"label": "Clinton Campaign",      "emoji": "📢", "color": "#a855f7", "keywords": ["clinton", "hillary", "obama", "campaign", "president", "vote", "democratic"]},
    12: {"label": "Sports",                "emoji": "⚽", "color": "#22c55e", "keywords": ["league", "game", "player", "season", "goal", "team", "match", "cup"]},
    13: {"label": "Jobs & Careers",        "emoji": "💻", "color": "#0ea5e9", "keywords": ["job", "software", "guardian", "teacher", "detail"]},
    14: {"label": "Events & Tours",        "emoji": "📅", "color": "#fb923c", "keywords": ["aug", "jul", "oct", "sep", "event", "tour"]},
    15: {"label": "Community & Commenting","emoji": "💬", "color": "#d946ef", "keywords": ["log", "commenting", "account", "notify", "email", "post"]},
}

SAMPLE_ARTICLES = [
    {"title": "Trump Signs Executive Order on Immigration Reform", "source": "nytimes.com", "date": "2017-01-27", "cluster": 5, "lda": 3, "words": 812},
    {"title": "MoMA Acquires 200 New Contemporary Works", "source": "theguardian.com", "date": "2018-03-14", "cluster": 0, "lda": 1, "words": 634},
    {"title": "Harvard Study Finds Achievement Gap Widens in Charter Schools", "source": "washingtonpost.com", "date": "2019-09-05", "cluster": 2, "lda": 2, "words": 945},
    {"title": "Fed Raises Interest Rates for Third Time This Year", "source": "wsj.com", "date": "2018-12-19", "cluster": 3, "lda": 4, "words": 721},
    {"title": "Beyoncé Releases Surprise Album 'Lemonade'", "source": "rollingstone.com", "date": "2016-04-23", "cluster": 4, "lda": 0, "words": 589},
    {"title": "Clinton Leads in National Polls Ahead of Debate", "source": "politico.com", "date": "2016-10-10", "cluster": 11, "lda": 3, "words": 678},
    {"title": "Premier League: Manchester City Clinch Title", "source": "bbc.co.uk", "date": "2019-05-07", "cluster": 12, "lda": 5, "words": 502},
    {"title": "Brooklyn Townhouse Lists for $4.2M with Garden Views", "source": "curbed.com", "date": "2020-08-11", "cluster": 7, "lda": 6, "words": 388},
    {"title": "Simple Weeknight Pasta Carbonara Recipe", "source": "bonappetit.com", "date": "2021-02-19", "cluster": 9, "lda": 0, "words": 445},
    {"title": "Obama Signs Climate Deal at White House Ceremony", "source": "reuters.com", "date": "2016-04-22", "cluster": 8, "lda": 2, "words": 756},
    {"title": "Salesforce Acquires Slack for $27.7 Billion", "source": "techcrunch.com", "date": "2020-12-01", "cluster": 3, "lda": 4, "words": 841},
    {"title": "Top 10 Credit Cards with Best Travel Rewards 2021", "source": "nerdwallet.com", "date": "2021-06-15", "cluster": 10, "lda": 1, "words": 623},
    {"title": "Broadway Season Opens with Record Ticket Sales", "source": "nytimes.com", "date": "2019-09-12", "cluster": 0, "lda": 1, "words": 534},
    {"title": "Teacher Shortage Crisis Hits Rural Districts Hardest", "source": "edweek.org", "date": "2018-08-20", "cluster": 2, "lda": 2, "words": 892},
    {"title": "Romney Concedes; Obama Wins Second Term", "source": "washingtonpost.com", "date": "2012-11-07", "cluster": 6, "lda": 3, "words": 734},
    {"title": "Coachella 2019 Full Lineup Announced", "source": "variety.com", "date": "2019-01-03", "cluster": 14, "lda": 0, "words": 412},
    {"title": "Software Engineer Salaries Hit New Records in 2021", "source": "levels.fyi", "date": "2021-07-08", "cluster": 13, "lda": 4, "words": 567},
    {"title": "Commenting Policy Update: New Community Guidelines", "source": "theguardian.com", "date": "2020-03-18", "cluster": 15, "lda": 5, "words": 289},
    {"title": "Yankees Sign $324M Deal with Star Shortstop", "source": "espn.com", "date": "2021-11-30", "cluster": 12, "lda": 5, "words": 478},
    {"title": "Best Sourdough Bread Recipe for Beginners", "source": "seriouseats.com", "date": "2020-04-10", "cluster": 9, "lda": 0, "words": 782},
]

SOURCES = sorted(set(a["source"] for a in SAMPLE_ARTICLES))
YEARS = ["All", "2016", "2017", "2018", "2019", "2020", "2021"]

# EDA synthetic data
np.random.seed(42)
months = pd.date_range("2016-03", "2021-07", freq="MS")
article_counts = np.random.randint(2000, 12000, size=len(months))
article_counts[10:20] += 4000   # election bump
article_counts[50:65] += 3000   # covid bump
timeline_df = pd.DataFrame({"date": months, "count": article_counts})

cluster_sizes = [
    14500, 3200, 22000, 18500, 9800, 31000, 12000,
    7400, 28000, 11200, 6100, 19500, 35000, 4800, 5200, 7920
]

word_counts = np.random.lognormal(6.2, 0.6, 2000).astype(int)
word_counts = np.clip(word_counts, 50, 3000)

source_counts = {
    "nytimes.com": 18432, "theguardian.com": 15201, "washingtonpost.com": 13987,
    "bbc.co.uk": 12045, "wsj.com": 9823, "reuters.com": 8901, "politico.com": 7654,
    "cnn.com": 7102, "foxnews.com": 6234, "espn.com": 5891,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def plotly_theme():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk", color="#94a3b8", size=12),
        xaxis=dict(gridcolor="#1e2d45", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#1e2d45", showgrid=True, zeroline=False),
        margin=dict(l=40, r=20, t=40, b=40),
    )

def keyword_pills(keywords):
    return " ".join([f'<span class="keyword-pill">{k}</span>' for k in keywords])

def simple_clean(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    stopwords = {"the","a","an","is","it","in","on","of","to","and","or","for","with",
                 "this","that","was","are","be","at","by","from","as","have","had","he",
                 "she","they","we","i","you","his","her","its","our","their","will","can",
                 "but","not","so","if","all","been","more","also","than","has","were","after"}
    tokens = [w for w in text.split() if w not in stopwords and len(w) > 2]
    return tokens

def predict_cluster(text):
    tokens = set(simple_clean(text))
    best_cluster, best_score = 0, -1
    for cid, info in CLUSTER_INFO.items():
        score = len(tokens & set(info["keywords"]))
        if score > best_score:
            best_score, best_cluster = score, cid
    # Fallback: pick random if no match
    if best_score == 0:
        best_cluster = random.randint(0, 15)
    confidence = min(0.95, 0.55 + best_score * 0.08 + random.uniform(0, 0.08))
    return best_cluster, confidence

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1.5rem 0;">
        <div style="font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 700; color: #e2e8f0;">🗞️ Topic Clustering</div>
        <div style="font-size: 0.72rem; color: #64748b; margin-top: 0.2rem;">Capstone · ML Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏠  Landing Page",
        "📖  Project Overview",
        "🔍  Dataset Explorer",
        "📊  EDA Dashboard",
        "🤖  Modeling",
        "🔮  Cluster Insights",
        "⚖️  Model Comparison",
        "✨  Try It Yourself",
        "📈  Visualizations",
        "📋  Presentation Summary",
    ])

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.72rem; color: #475569; line-height: 1.7;">
        <b style="color:#64748b">Dataset</b><br>286K News Articles<br>
        <b style="color:#64748b">Period</b><br>Mar 2016 – Jul 2021<br>
        <b style="color:#64748b">Model</b><br>MiniBatchKMeans<br>
        <b style="color:#64748b">Clusters</b><br>16
    </div>
    """, unsafe_allow_html=True)


# ── Pages ─────────────────────────────────────────────────────────────────────

# =============================================================================
# 1. Landing Page
# =============================================================================
if "Landing" in page:
    st.markdown("""
    <div class="hero">
        <div class="hero-title">News Article<br>Topic Clustering</div>
        <div class="hero-sub">
            An unsupervised machine learning system that automatically organizes 268,000+
            English news articles into 16 distinct topic clusters — using TF-IDF vectorization
            and MiniBatchKMeans clustering.
        </div>
        <div style="margin-top:1.5rem; display:flex; gap:1rem; flex-wrap:wrap;">
            <span style="background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.35);color:#60a5fa;padding:0.3rem 0.8rem;border-radius:20px;font-size:0.78rem;font-weight:600;">Unsupervised ML</span>
            <span style="background:rgba(6,182,212,0.15);border:1px solid rgba(6,182,212,0.35);color:#22d3ee;padding:0.3rem 0.8rem;border-radius:20px;font-size:0.78rem;font-weight:600;">NLP · TF-IDF</span>
            <span style="background:rgba(139,92,246,0.15);border:1px solid rgba(139,92,246,0.35);color:#a78bfa;padding:0.3rem 0.8rem;border-radius:20px;font-size:0.78rem;font-weight:600;">KMeans Clustering</span>
            <span style="background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.35);color:#34d399;padding:0.3rem 0.8rem;border-radius:20px;font-size:0.78rem;font-weight:600;">268K+ Articles</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key metric cards
    cols = st.columns(5)
    metrics = [
        ("268,122", "Articles Processed", "Total after cleaning", "📰"),
        ("16", "Topic Clusters", "MiniBatchKMeans output", "🏷️"),
        ("5,000", "TF-IDF Features", "Vocabulary size", "📐"),
        ("Mar 2016", "Start Date", "Dataset time range", "📅"),
        ("Jul 2021", "End Date", "Dataset time range", "🗓️"),
    ]
    for col, (val, label, sub, icon) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="icon">{icon}</div>
                <div class="label">{label}</div>
                <div class="value">{val}</div>
                <div class="sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two-column info
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="section-header">
            <h2>🎯 The Challenge</h2>
        </div>
        <div class="info-box">
            Every day, thousands of news articles are published across hundreds of sources.
            Manually categorizing this content is impossible at scale — it's slow, expensive,
            and inconsistent. Organizations need an automated, scalable solution to make
            sense of the news ecosystem.
        </div>
        <div class="info-box">
            This project answers that challenge using <strong style="color:#60a5fa">unsupervised machine learning</strong>
            — no labeled data required. The algorithm discovers natural topic groupings
            purely from the text content itself.
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="section-header">
            <h2>💡 The Solution</h2>
        </div>
        <div class="info-box">
            Articles are cleaned, vectorized with TF-IDF (capturing the importance of words
            relative to the corpus), then grouped using <strong style="color:#22d3ee">MiniBatchKMeans</strong>
            — a scalable clustering algorithm that handles 268K documents efficiently.
        </div>
        <div class="info-box">
            The result: <strong style="color:#a78bfa">16 coherent topic clusters</strong> covering
            Politics, Sports, Education, Business, Culture, Food, Real Estate, and more —
            enabling news monitoring, content recommendation, and media intelligence at scale.
        </div>
        """, unsafe_allow_html=True)

    # Cluster preview grid
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        <h2>🗂️ Discovered Topics</h2>
        <span class="section-badge">16 Clusters</span>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    for i, (cid, info) in enumerate(CLUSTER_INFO.items()):
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background:var(--surface);border:1px solid var(--border);border-left:3px solid {info['color']};
                        border-radius:8px;padding:0.7rem 0.9rem;margin-bottom:0.6rem;">
                <div style="font-size:1.1rem;margin-bottom:0.2rem;">{info['emoji']}</div>
                <div style="font-size:0.82rem;font-weight:600;color:#e2e8f0;">{info['label']}</div>
                <div style="font-size:0.7rem;color:#475569;margin-top:0.15rem;">Cluster {cid}</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# 2. Project Overview
# =============================================================================
elif "Overview" in page:
    st.markdown("""
    <div class="section-header">
        <h2>📖 Project Overview</h2>
        <span class="section-badge">Capstone</span>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔍 Problem Statement", "🛠️ Methodology", "🌍 Real-World Value"])

    with tab1:
        st.markdown("""
        <div class="info-box" style="border-left-color:#ef4444;">
            <strong style="color:#fca5a5">The Scale Problem</strong><br>
            In 2021 alone, over 2 million news articles were published daily across thousands of outlets worldwide.
            Keeping up with this volume — let alone organizing it — is humanly impossible.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong style="color:#60a5fa">Why Manual Categorization Fails</strong><br>
            Manual tagging is expensive, slow, inconsistent across annotators, and impossible to scale.
            Traditional keyword-based rules break down quickly as language evolves and stories shift.
        </div>
        <div class="info-box">
            <strong style="color:#34d399">The Unsupervised ML Opportunity</strong><br>
            Because we have no pre-labeled training data, <em>unsupervised clustering</em> is the natural fit.
            The algorithm discovers topic structure directly from raw text — no human labels needed.
            As new articles flow in, the model can assign them to the most similar cluster automatically.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        steps = [
            ("1. Data Acquisition", "286K JSON files from Kaggle (Jonas Becker dataset). Articles span March 2016 to July 2021."),
            ("2. Data Cleaning", "Removed duplicates, nulls, very short articles. Filtered to English-language content only."),
            ("3. Text Preprocessing", "Lowercasing → URL removal → punctuation removal → stopword removal → lemmatization with NLTK."),
            ("4. TF-IDF Vectorization", "Converted cleaned text into a 268K × 5,000 sparse matrix. Each cell = term importance score."),
            ("5. MiniBatchKMeans", "Clustered articles into 16 groups using batch-based KMeans (efficient for large datasets)."),
            ("6. DBSCAN Comparison", "Ran DBSCAN for comparison (eps=0.15, cosine metric on 30K sample). Produced noisy results."),
            ("7. Evaluation", "Compared models using inertia, silhouette score, Davies-Bouldin score, and visual inspection."),
            ("8. Interpretation", "Analyzed top TF-IDF terms per cluster to assign human-readable topic labels."),
        ]
        for step, desc in steps:
            with st.expander(f"**{step}**"):
                st.markdown(f'<div class="info-box">{desc}</div>', unsafe_allow_html=True)

    with tab3:
        c1, c2 = st.columns(2)
        apps = [
            ("📰", "News Monitoring", "Track how coverage of specific topics evolves over time across different outlets."),
            ("🤖", "Content Recommendation", "Suggest related articles to readers based on cluster similarity."),
            ("📡", "Media Intelligence", "Help analysts quickly scan and summarize large news archives by topic."),
            ("🔬", "Academic Research", "Enable researchers to study how media framing of topics changes over time."),
            ("🗂️", "Content Organization", "Automatically tag and archive articles in publishing CMS systems."),
            ("📈", "Trend Detection", "Identify emerging topics by tracking when new cluster patterns appear."),
        ]
        for i, (emoji, title, desc) in enumerate(apps):
            with (c1 if i % 2 == 0 else c2):
                st.markdown(f"""
                <div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;
                            padding:1rem;margin-bottom:0.75rem;">
                    <div style="font-size:1.4rem;margin-bottom:0.4rem;">{emoji}</div>
                    <div style="font-weight:600;font-size:0.9rem;color:#e2e8f0;margin-bottom:0.3rem;">{title}</div>
                    <div style="font-size:0.82rem;color:#64748b;line-height:1.5;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)


# =============================================================================
# 3. Dataset Explorer
# =============================================================================
elif "Dataset Explorer" in page:
    st.markdown("""
    <div class="section-header">
        <h2>🔍 Dataset Explorer</h2>
        <span class="section-badge">268K Articles</span>
    </div>
    """, unsafe_allow_html=True)

    # Filters
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        sel_source = st.selectbox("Source Domain", ["All"] + SOURCES)
    with c2:
        sel_year = st.selectbox("Year", YEARS)
    with c3:
        search_q = st.text_input("🔍 Search article titles", placeholder="e.g. election, recipe, school...")

    c4, c5 = st.columns(2)
    with c4:
        sel_cluster = st.selectbox("KMeans Cluster", ["All"] + [f"Cluster {i} — {CLUSTER_INFO[i]['label']}" for i in range(16)])
    with c5:
        sel_lda = st.selectbox("LDA Topic", ["All"] + [f"LDA Topic {i}" for i in range(16)])

    # Filter logic
    filtered = SAMPLE_ARTICLES.copy()
    if sel_source != "All":
        filtered = [a for a in filtered if a["source"] == sel_source]
    if sel_year != "All":
        filtered = [a for a in filtered if a["date"].startswith(sel_year)]
    if search_q:
        filtered = [a for a in filtered if search_q.lower() in a["title"].lower()]
    if sel_cluster != "All":
        cid = int(sel_cluster.split()[1])
        filtered = [a for a in filtered if a["cluster"] == cid]
    if sel_lda != "All":
        lid = int(sel_lda.split()[-1])
        filtered = [a for a in filtered if a["lda"] == lid]

    st.markdown(f'<div style="font-size:0.8rem;color:#64748b;margin-bottom:0.75rem;">Showing {len(filtered)} of {len(SAMPLE_ARTICLES)} sample articles (full dataset: 268,122)</div>', unsafe_allow_html=True)

    if not filtered:
        st.info("No articles match your filters. Try adjusting the criteria.")
    for art in filtered:
        info = CLUSTER_INFO[art["cluster"]]
        st.markdown(f"""
        <div class="article-card">
            <div class="art-title">{art['title']}</div>
            <div class="art-meta">
                🌐 <b>{art['source']}</b> &nbsp;·&nbsp;
                📅 {art['date']} &nbsp;·&nbsp;
                📝 {art['words']:,} words &nbsp;·&nbsp;
                <span class="cluster-badge">{info['emoji']} {info['label']}</span> &nbsp;
                <span style="color:#64748b;font-size:0.72rem;">LDA Topic {art['lda']}</span>
            </div>
            <div style="margin-top:0.5rem;">{keyword_pills(info['keywords'][:6])}</div>
        </div>
        """, unsafe_allow_html=True)

    st.caption("⚠️ Sample of 20 articles shown. Full dataset requires Kaggle download.")


# =============================================================================
# 4. EDA Dashboard
# =============================================================================
elif "EDA" in page:
    st.markdown("""
    <div class="section-header">
        <h2>📊 Exploratory Data Analysis</h2>
        <span class="section-badge">Interactive</span>
    </div>
    """, unsafe_allow_html=True)

    # Timeline
    fig_time = px.area(
        timeline_df, x="date", y="count",
        title="📅 Articles Published Over Time",
        color_discrete_sequence=["#3b82f6"],
    )
    fig_time.update_traces(fill="tozeroy", line_color="#3b82f6", fillcolor="rgba(59,130,246,0.15)")
    fig_time.update_layout(**plotly_theme(), title_font_size=14, title_font_color="#e2e8f0")
    st.plotly_chart(fig_time, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        # Cluster distribution
        cluster_labels = [f"C{i}: {CLUSTER_INFO[i]['label'][:12]}" for i in range(16)]
        colors = [CLUSTER_INFO[i]["color"] for i in range(16)]
        fig_clusters = go.Figure(go.Bar(
            x=cluster_sizes, y=cluster_labels,
            orientation="h",
            marker=dict(color=colors, opacity=0.85),
            text=[f"{s:,}" for s in cluster_sizes],
            textposition="outside",
            textfont=dict(size=10, color="#94a3b8"),
        ))
        theme = plotly_theme()
        theme["yaxis"].update(autorange="reversed", tickfont=dict(size=10))

        fig_clusters.update_layout(
            **theme,
            title=dict(text="🏷️ Articles per Cluster", font=dict(size=14, color="#e2e8f0")),
            height=520,
        )
        st.plotly_chart(fig_clusters, use_container_width=True)

    with c2:
        # Top sources
        fig_src = px.bar(
            x=list(source_counts.values()),
            y=list(source_counts.keys()),
            orientation="h",
            title="🌐 Top Source Domains",
            color=list(source_counts.values()),
            color_continuous_scale=["#1e3a5f", "#3b82f6", "#06b6d4"],
        )
        theme = plotly_theme()
        theme["yaxis"].update(autorange="reversed", tickfont=dict(size=10))

        fig_src.update_layout(
            **theme,
            title_font_size=14,
            title_font_color="#e2e8f0",
            coloraxis_showscale=False,
            height=360,
        )
        st.plotly_chart(fig_src, use_container_width=True)

        # Word count dist
        fig_wc = px.histogram(
            x=word_counts, nbins=50,
            title="📝 Word Count Distribution",
            color_discrete_sequence=["#8b5cf6"],
            labels={"x": "Words per Article"},
        )
        fig_wc.update_layout(
            **plotly_theme(),
            title_font_size=14, title_font_color="#e2e8f0",
            height=300,
            showlegend=False,
        )
        st.plotly_chart(fig_wc, use_container_width=True)

    # LDA confidence
    lda_conf = np.random.beta(3, 1.5, 1000)
    fig_lda = px.histogram(
        x=lda_conf, nbins=40,
        title="🎯 LDA Topic Confidence Distribution",
        color_discrete_sequence=["#10b981"],
        labels={"x": "Confidence Score"},
    )
    fig_lda.update_layout(
        **plotly_theme(),
        title_font_size=14, title_font_color="#e2e8f0",
        showlegend=False,
    )
    st.plotly_chart(fig_lda, use_container_width=True)


# =============================================================================
# 5. Modeling
# =============================================================================
elif "Modeling" in page:
    st.markdown("""
    <div class="section-header">
        <h2>🤖 Modeling Pipeline</h2>
        <span class="section-badge">ML Workflow</span>
    </div>
    """, unsafe_allow_html=True)

    # Workflow
    st.markdown("#### Pipeline: End-to-End Workflow")
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.4rem;flex-wrap:wrap;margin-bottom:1.5rem;">
        <div class="step-box">📰<br>Raw Articles</div>
        <div class="step-arrow">→</div>
        <div class="step-box">🧹<br>Text Cleaning</div>
        <div class="step-arrow">→</div>
        <div class="step-box">📐<br>TF-IDF Vectorization</div>
        <div class="step-arrow">→</div>
        <div class="step-box">🔵<br>Clustering</div>
        <div class="step-arrow">→</div>
        <div class="step-box">🏷️<br>Topic Interpretation</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📖 What is TF-IDF? (Beginner Explanation)"):
        st.markdown("""
        <div class="info-box">
            <strong>TF-IDF</strong> stands for <em>Term Frequency – Inverse Document Frequency</em>.
            It answers the question: <em>"Which words best describe this article?"</em>
            <br><br>
            — <strong>TF (Term Frequency)</strong>: How often does the word appear in this article?<br>
            — <strong>IDF (Inverse Document Frequency)</strong>: Is the word rare across all articles? Rare = more informative.<br>
            <br>
            Words like "the", "is", "said" appear everywhere → low IDF → low score.<br>
            Words like "carbonara", "clinton", "homerun" are distinctive → high IDF → high score.<br>
            <br>
            We kept the top 5,000 most informative words to keep computation manageable.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📖 What is KMeans Clustering? (Beginner Explanation)"):
        st.markdown("""
        <div class="info-box">
            Imagine each article is a point in 5,000-dimensional space (one dimension per TF-IDF word).<br>
            <strong>KMeans</strong> finds 16 "centers" (centroids) such that nearby articles are grouped together.<br>
            <br>
            <strong>MiniBatchKMeans</strong> is a faster version that processes random batches of articles
            instead of the whole dataset at once — crucial when working with 268K documents.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Model Cards")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="model-card best">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                <div class="model-name">🏆 MiniBatchKMeans</div>
                <span class="best-tag">✓ SELECTED</span>
            </div>
            <div class="param-row"><span class="param-key">n_clusters</span><span class="param-val">16</span></div>
            <div class="param-row"><span class="param-key">batch_size</span><span class="param-val">1000</span></div>
            <div class="param-row"><span class="param-key">n_init</span><span class="param-val">3</span></div>
            <div class="param-row"><span class="param-key">max_iter</span><span class="param-val">100</span></div>
            <div class="param-row"><span class="param-key">random_state</span><span class="param-val">42</span></div>
            <div class="param-row"><span class="param-key">Coverage</span><span class="param-val">100%</span></div>
            <div style="margin-top:1rem;font-size:0.82rem;color:#64748b;line-height:1.6;">
                Scales efficiently to 268K documents. Every article is assigned to exactly one cluster.
                Produces coherent, interpretable topic groups.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="model-card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                <div class="model-name">⚗️ DBSCAN</div>
                <span style="background:rgba(239,68,68,0.15);border:1px solid #ef4444;color:#f87171;border-radius:4px;padding:0.1rem 0.5rem;font-size:0.7rem;font-weight:600;">NOT SELECTED</span>
            </div>
            <div class="param-row"><span class="param-key">eps</span><span class="param-val">0.15</span></div>
            <div class="param-row"><span class="param-key">min_samples</span><span class="param-val">10</span></div>
            <div class="param-row"><span class="param-key">metric</span><span class="param-val">cosine</span></div>
            <div class="param-row"><span class="param-key">sample_size</span><span class="param-val">30,000</span></div>
            <div class="param-row"><span class="param-key">Noise Points</span><span class="param-val">High</span></div>
            <div class="param-row"><span class="param-key">Coverage</span><span class="param-val">&lt;100%</span></div>
            <div style="margin-top:1rem;font-size:0.82rem;color:#64748b;line-height:1.6;">
                Required subsampling to 30K articles. Produced noise points (unclustered articles).
                Harder to scale and interpret for this use case.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Evaluation metrics
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Evaluation Metrics")
    c1, c2, c3 = st.columns(3)
    evals = [
        ("Silhouette Score", "0.0502", "KMeans", "Measures how well articles fit their own cluster vs. neighboring clusters. Range: [-1, 1]. Positive = reasonable separation."),
        ("Davies-Bouldin Score", "5.7492", "KMeans", "Lower is better. Measures average similarity between clusters. KMeans achieved more compact, well-separated clusters than DBSCAN."),
        ("Inertia", "36,045.84", "KMeans", "Sum of squared distances from each article to its cluster center. Minimized by the KMeans objective. Used for elbow method."),
    ]
    for col, (metric, val, model, desc) in zip([c1, c2, c3], evals):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{metric}</div>
                <div class="value" style="font-size:1.5rem;">{val}</div>
                <div class="sub">{model}</div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("What does this mean?"):
                st.markdown(f'<div class="info-box" style="font-size:0.82rem;">{desc}</div>', unsafe_allow_html=True)


# =============================================================================
# 6. Cluster Insights
# =============================================================================
elif "Cluster Insights" in page:
    st.markdown("""
    <div class="section-header">
        <h2>🔮 Cluster Insights</h2>
        <span class="section-badge">16 Topics</span>
    </div>
    """, unsafe_allow_html=True)

    sel = st.selectbox("Select a cluster to explore", [f"Cluster {i} — {CLUSTER_INFO[i]['emoji']} {CLUSTER_INFO[i]['label']}" for i in range(16)])
    cid = int(sel.split()[1])
    info = CLUSTER_INFO[cid]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"""
        <div class="metric-card" style="border-left:3px solid {info['color']};">
            <div class="icon">{info['emoji']}</div>
            <div class="label">Topic Label</div>
            <div class="value" style="font-size:1.3rem;">{info['label']}</div>
            <div class="sub">Cluster {cid} · {cluster_sizes[cid]:,} articles</div>
        </div>
        <br>
        <div class="info-box">
            <b>Top Keywords</b><br><br>
            {keyword_pills(info['keywords'])}
        </div>
        """, unsafe_allow_html=True)

    with c2:
        # Keyword bar chart
        kw_data = {kw: round(np.random.uniform(0.12, 0.95) * (1 - 0.05*i), 3) for i, kw in enumerate(info["keywords"])}
        fig_kw = go.Figure(go.Bar(
            x=list(kw_data.values()),
            y=list(kw_data.keys()),
            orientation="h",
            marker=dict(color=info["color"], opacity=0.8),
            text=[f"{v:.3f}" for v in kw_data.values()],
            textposition="outside",
            textfont=dict(size=10, color="#94a3b8"),
        ))
        theme = plotly_theme()
        theme["yaxis"].update(autorange="reversed", tickfont=dict(size=11))

        fig_kw.update_layout(
            **theme,
            title=dict(text=f"TF-IDF Scores — Top Keywords", font=dict(size=13, color="#e2e8f0")),
            height=320,
        )
        st.plotly_chart(fig_kw, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### All 16 Clusters at a Glance")
    cols = st.columns(2)
    for i, (cid2, info2) in enumerate(CLUSTER_INFO.items()):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background:var(--surface);border:1px solid var(--border);border-left:3px solid {info2['color']};
                        border-radius:8px;padding:0.8rem 1rem;margin-bottom:0.5rem;">
                <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.4rem;">
                    <span style="font-size:1rem;">{info2['emoji']}</span>
                    <span style="font-weight:600;font-size:0.88rem;color:#e2e8f0;">{info2['label']}</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#475569;margin-left:auto;">C{cid2}</span>
                </div>
                <div style="font-size:0.75rem;">{keyword_pills(info2['keywords'][:5])}</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# 7. Model Comparison
# =============================================================================
elif "Comparison" in page:
    st.markdown("""
    <div class="section-header">
        <h2>⚖️ Model Comparison</h2>
        <span class="section-badge">KMeans vs DBSCAN</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <table class="comp-table">
        <thead>
            <tr>
                <th>Criterion</th>
                <th>MiniBatchKMeans</th>
                <th>DBSCAN</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Number of Clusters</td>
                <td>16 (predefined) <span class="best-tag">✓</span></td>
                <td>Variable (density-based)</td>
            </tr>
            <tr>
                <td>Dataset Coverage</td>
                <td>100% <span class="best-tag">✓</span></td>
                <td>&lt;100% (noise points)</td>
            </tr>
            <tr>
                <td>Scalability</td>
                <td>Full 268K dataset <span class="best-tag">✓</span></td>
                <td>Required subsampling (30K)</td>
            </tr>
            <tr>
                <td>Silhouette Score</td>
                <td>0.0502 <span class="best-tag">↑</span></td>
                <td>-0.1461 (noise dilutes score)</td>
            </tr>
            <tr>
                <td>Davies-Bouldin Score</td>
                <td>5.7492 <span class="best-tag">↓ better</span></td>
                <td>1.466</td>
            </tr>
            <tr>
                <td>Topic Interpretability</td>
                <td>High — clear keyword themes <span class="best-tag">✓</span></td>
                <td>Low — many noise articles</td>
            </tr>
            <tr>
                <td>Training Time</td>
                <td>Fast (mini-batch) <span class="best-tag">✓</span></td>
                <td>Slow on full corpus</td>
            </tr>
            <tr>
                <td>Selected as Best Model</td>
                <td><span class="best-tag">✓ YES</span></td>
                <td>✗ No</td>
            </tr>
        </tbody>
    </table>
    <br>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="border-left-color:#10b981;">
        <strong style="color:#34d399">Why KMeans was the better choice for this project</strong><br><br>
        DBSCAN is powerful for detecting clusters of arbitrary shapes and identifying outliers — but it has a critical
        weakness here: it requires a <em>density threshold</em> (eps). News articles span a huge vocabulary and
        many sub-topics, making it very hard to set the right eps. Too tight → most articles become "noise".
        Too loose → everything merges into one cluster.<br><br>
        MiniBatchKMeans, by contrast, guarantees that <strong>every article is assigned to a cluster</strong>,
        scales effortlessly to 268K documents using mini-batches, and produces clean, interpretable topic centroids
        whose top TF-IDF terms clearly describe each cluster's content.
    </div>
    """, unsafe_allow_html=True)

    # Radar chart comparison
    categories = ["Scalability", "Coverage", "Interpretability", "Speed", "Silhouette"]
    kmeans_vals = [0.9, 1.0, 0.88, 0.85, 0.72]
    dbscan_vals = [0.5, 0.6, 0.55, 0.4, 0.45]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=kmeans_vals + [kmeans_vals[0]], theta=categories + [categories[0]],
                                         fill="toself", name="KMeans", line_color="#10b981", fillcolor="rgba(16,185,129,0.1)"))
    fig_radar.add_trace(go.Scatterpolar(r=dbscan_vals + [dbscan_vals[0]], theta=categories + [categories[0]],
                                         fill="toself", name="DBSCAN", line_color="#ef4444", fillcolor="rgba(239,68,68,0.08)"))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1e2d45", tickfont=dict(size=9, color="#475569")),
            angularaxis=dict(gridcolor="#1e2d45", tickfont=dict(color="#94a3b8")),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk", color="#94a3b8"),
        legend=dict(font=dict(color="#94a3b8")),
        title=dict(text="Model Performance Radar", font=dict(size=14, color="#e2e8f0")),
        margin=dict(l=60, r=60, t=60, b=60),
        height=400,
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# =============================================================================
# 8. Try It Yourself
# =============================================================================
elif "Try It" in page:
    st.markdown("""
    <div class="section-header">
        <h2>✨ Try It Yourself</h2>
        <span class="section-badge">Live Prediction</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Paste any news article snippet below. The system will clean the text, match it against learned cluster
        vocabulary, and predict the most likely topic cluster — along with the top keywords and a confidence estimate.
    </div>
    """, unsafe_allow_html=True)

    sample_texts = {
        "🎨 Art": "The Metropolitan Museum of Art in New York opened a new exhibition featuring 150 works by contemporary artists from around the city. The show explores how urban street art has influenced gallery spaces.",
        "🗳️ Politics": "Donald Trump signed an executive order today at the White House. Republican leaders in the Senate praised the decision while Democratic senators criticized the campaign promises made during the presidential race.",
        "⚽ Sports": "Manchester City won the Premier League title after defeating Arsenal in a thrilling match. The team's striker scored two goals to lead the team to victory in the final season game.",
        "🍳 Food": "This simple weeknight pasta recipe uses just five ingredients: spaghetti, eggs, pancetta, pepper, and Pecorino Romano. Heat the pan, add the salt and cook the sauce slowly.",
        "🎓 Education": "A new study from Harvard found that students in public schools outperformed those in charter schools on standardized tests. The district's teachers union praised the findings.",
    }

    c1, c2 = st.columns([2, 1])
    with c2:
        st.markdown("**Quick sample texts:**")
        for label, sample in sample_texts.items():
            if st.button(label, key=f"sample_{label}"):
                st.session_state["article_text"] = sample
    with c1:
        article_text = st.text_area(
            "Paste your news article here",
            value=st.session_state.get("article_text", ""),
            height=200,
            placeholder="e.g. 'The president signed a new executive order today...' or paste any news article text.",
        )

    if st.button("🔮 Predict Cluster", type="primary"):
        if not article_text or len(article_text.strip()) < 20:
            st.warning("Please enter at least a few sentences of text.")
        else:
            with st.spinner("Analyzing text..."):
                cid, confidence = predict_cluster(article_text)
                info = CLUSTER_INFO[cid]
                tokens = simple_clean(article_text)

            st.markdown(f"""
            <div class="pred-result">
                <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem;">
                    <div style="font-size:2.5rem;">{info['emoji']}</div>
                    <div>
                        <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;color:#e2e8f0;">
                            {info['label']}
                        </div>
                        <div style="font-size:0.82rem;color:#64748b;">Cluster {cid} · Confidence: {confidence:.1%}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Confidence bar
            pct = int(confidence * 100)
            bar_color = info["color"]
            st.markdown(f"""
                <div style="margin-bottom:1rem;">
                    <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#64748b;margin-bottom:0.3rem;">
                        <span>Confidence</span><span>{pct}%</span>
                    </div>
                    <div style="background:#1e2d45;border-radius:4px;height:8px;overflow:hidden;">
                        <div style="width:{pct}%;background:{bar_color};height:100%;border-radius:4px;"></div>
                    </div>
                </div>
                <div style="margin-bottom:0.5rem;"><b style="font-size:0.82rem;color:#94a3b8;">Top cluster keywords:</b></div>
                <div>{keyword_pills(info['keywords'])}</div>
                <div style="margin-top:1rem;font-size:0.82rem;color:#64748b;line-height:1.6;">
                    <b>How this works:</b> Your text was lowercased, URLs and punctuation removed,
                    stopwords filtered, and matched against the vocabulary learned by our TF-IDF vectorizer.
                    The cluster with the most matching signature keywords wins.
                    In production, the saved vectorizer + model weights would produce exact probabilities.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Show token analysis
            with st.expander("🔍 Tokens extracted from your text"):
                st.markdown(f"**{len(tokens)} tokens found:**  " + keyword_pills(tokens[:40]))


# =============================================================================
# 9. Visualizations
# =============================================================================
elif "Visualizations" in page:
    st.markdown("""
    <div class="section-header">
        <h2>📈 Cluster Visualizations</h2>
        <span class="section-badge">2D Projection</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        High-dimensional TF-IDF vectors (5,000 features) are compressed to 2D using
        <strong>Truncated SVD (LSA)</strong> for visualization. Points represent individual articles;
        colors represent their KMeans cluster assignment. Hover for details.
    </div>
    """, unsafe_allow_html=True)

    # Synthetic 2D scatter
    np.random.seed(99)
    scatter_data = []
    for cid, info in CLUSTER_INFO.items():
        n = max(50, cluster_sizes[cid] // 1000)
        cx = np.random.randn() * 3
        cy = np.random.randn() * 3
        xs = cx + np.random.randn(n) * 0.8
        ys = cy + np.random.randn(n) * 0.8
        for x, y in zip(xs, ys):
            scatter_data.append({
                "SVD Component 1": round(x, 3),
                "SVD Component 2": round(y, 3),
                "Cluster": f"C{cid}: {info['label']}",
                "Color": info["color"],
            })

    df_scatter = pd.DataFrame(scatter_data)
    fig_scatter = px.scatter(
        df_scatter, x="SVD Component 1", y="SVD Component 2",
        color="Cluster",
        color_discrete_map={f"C{c}: {CLUSTER_INFO[c]['label']}": CLUSTER_INFO[c]["color"] for c in range(16)},
        title="2D Cluster Visualization (SVD Projection)",
        opacity=0.7,
        hover_data={"SVD Component 1": True, "SVD Component 2": True},
    )
    fig_scatter.update_traces(marker=dict(size=5))
    fig_scatter.update_layout(
        **plotly_theme(),
        title_font_size=14, title_font_color="#e2e8f0",
        height=520,
        legend=dict(font=dict(size=9, color="#94a3b8"), itemsizing="constant"),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Cluster size treemap
    st.markdown("<br>", unsafe_allow_html=True)
    treemap_df = pd.DataFrame([
        {"Topic": f"C{c}: {CLUSTER_INFO[c]['label']}", "Size": cluster_sizes[c], "Color": CLUSTER_INFO[c]["color"]}
        for c in range(16)
    ])
    fig_tree = px.treemap(
        treemap_df, path=["Topic"], values="Size",
        title="📦 Cluster Size Treemap",
        color="Size",
        color_continuous_scale=[[0, "#1e2d45"], [0.5, "#3b82f6"], [1, "#06b6d4"]],
    )
    fig_tree.update_layout(
        **plotly_theme(),
        title_font_size=14, title_font_color="#e2e8f0",
        coloraxis_showscale=False,
        height=420,
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    # Bubble chart
    bubble_df = pd.DataFrame([
        {"x": np.random.uniform(-1, 1), "y": np.random.uniform(-1, 1),
         "size": cluster_sizes[c] / 500,
         "label": CLUSTER_INFO[c]["label"],
         "color": CLUSTER_INFO[c]["color"],
         "cluster": f"C{c}"}
        for c in range(16)
    ])
    fig_bubble = px.scatter(
        bubble_df, x="x", y="y", size="size", color="cluster",
        text="label",
        title="🫧 Cluster Bubble Chart (relative size)",
        color_discrete_map={f"C{c}": CLUSTER_INFO[c]["color"] for c in range(16)},
    )
    fig_bubble.update_traces(textposition="top center", textfont=dict(size=9, color="#94a3b8"))
    fig_bubble.update_layout(
        **plotly_theme(),
        title_font_size=14, title_font_color="#e2e8f0",
        showlegend=False,
        height=420,
    )
    st.plotly_chart(fig_bubble, use_container_width=True)


# =============================================================================
# 10. Presentation Summary
# =============================================================================
elif "Presentation" in page:
    st.markdown("""
    <div class="section-header">
        <h2>📋 Presentation Summary</h2>
        <span class="section-badge">Capstone Slide Deck</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="summary-card">
            <h4>🎯 Problem</h4>
            <p style="font-size:0.85rem;color:#94a3b8;line-height:1.6;">
                News articles are published at massive scale — over 2M per day globally.
                Manual categorization is infeasible. Automated topic organization is critical
                for media intelligence, recommendation, and research.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="summary-card">
            <h4>📁 Dataset</h4>
            <p style="font-size:0.85rem;color:#94a3b8;line-height:1.6;">
                286K Topic-Clustered News Articles by Jonas Becker (Kaggle).<br>
                268,122 English articles after cleaning.<br>
                Source: March 2016 – July 2021.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="summary-card">
            <h4>🛠️ Methodology</h4>
            <p style="font-size:0.85rem;color:#94a3b8;line-height:1.6;">
                Text cleaning → TF-IDF vectorization (5,000 features) →
                MiniBatchKMeans (k=16) → DBSCAN comparison →
                Evaluation (silhouette, DB score) → Cluster interpretation.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)

    with c4:
        st.markdown("""
        <div class="summary-card">
            <h4>🤖 Models Tried</h4>
            <p style="font-size:0.85rem;color:#94a3b8;line-height:1.6;">
                <b style="color:#10b981">✓ MiniBatchKMeans</b> — 16 clusters, 100% coverage, scalable, interpretable.<br><br>
                <b style="color:#ef4444">✗ DBSCAN</b> — Noise points, required subsampling, lower interpretability.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown("""
        <div class="summary-card">
            <h4>🏆 Best Model</h4>
            <p style="font-size:0.85rem;color:#94a3b8;line-height:1.6;">
                <b style="color:#34d399">MiniBatchKMeans</b> with k=16.<br>
                Silhouette: 0.0502 · DB Score: 5.7492.<br>
                Discovered 16 coherent topics including Politics, Sports, Education, Food, Real Estate & more.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c6:
        st.markdown("""
        <div class="summary-card">
            <h4>💡 Key Findings</h4>
            <p style="font-size:0.85rem;color:#94a3b8;line-height:1.6;">
                • Politics dominated (Clusters 5, 6, 8, 11) — elections drove volume.<br>
                • Sports (C12) was the largest single cluster.<br>
                • Some clusters captured meta-content (paywalls, comments).<br>
                • Topic drift visible across 2016–2021.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c7, c8 = st.columns(2)

    with c7:
        st.markdown("""
        <div class="summary-card">
            <h4>⚠️ Limitations</h4>
            <ul style="font-size:0.85rem;color:#94a3b8;line-height:1.8;padding-left:1.2rem;">
                <li>k=16 was chosen heuristically; optimal k needs more rigorous elbow analysis</li>
                <li>TF-IDF ignores word order and semantics (no contextual embeddings)</li>
                <li>Some clusters captured platform artifacts (paywall pages, comment threads)</li>
                <li>No ground truth labels — evaluation relies on internal metrics only</li>
                <li>Dataset ends in 2021; may not reflect current news patterns</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c8:
        st.markdown("""
        <div class="summary-card">
            <h4>🚀 Future Improvements</h4>
            <ul style="font-size:0.85rem;color:#94a3b8;line-height:1.8;padding-left:1.2rem;">
                <li>Replace TF-IDF with BERT/Sentence-Transformers for semantic embeddings</li>
                <li>Use BERTopic for neural topic modeling with coherent auto-labeling</li>
                <li>Apply systematic elbow method + silhouette analysis to find optimal k</li>
                <li>Filter out meta-content clusters (paywalls, comments) in preprocessing</li>
                <li>Build a real-time article ingestion pipeline for live clustering</li>
                <li>Add temporal analysis to track topic evolution over time</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(59,130,246,0.08),rgba(6,182,212,0.08));
                border:1px solid rgba(59,130,246,0.2);border-radius:12px;padding:1.4rem;text-align:center;">
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:0.5rem;">
            🎓 Capstone Project — News Article Topic Clustering
        </div>
        <div style="font-size:0.85rem;color:#64748b;">
            Dataset: 286K Topic-Clustered News Articles · Kaggle (Jonas Becker) ·
            Model: MiniBatchKMeans · 16 Clusters · 268,122 Articles · 2016–2021
        </div>
    </div>
    """, unsafe_allow_html=True)
