import os
import html
import requests
import streamlit as st

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000/analyze"
)

st.set_page_config(
    page_title="IBM Stock Analysis Agent",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .main {
        background: linear-gradient(180deg, #0b1020 0%, #111827 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #f8fafc;
    }

    .hero-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 28px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.28);
        margin-bottom: 1.25rem;
    }

    .glass-card {
        background: rgba(17, 24, 39, 0.88);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.22);
        margin-bottom: 1rem;
    }

    .metric-pill {
        display: inline-block;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }

    .pill-green {
        background: rgba(34,197,94,0.16);
        color: #86efac;
        border: 1px solid rgba(34,197,94,0.35);
    }

    .pill-yellow {
        background: rgba(250,204,21,0.16);
        color: #fde68a;
        border: 1px solid rgba(250,204,21,0.35);
    }

    .pill-red {
        background: rgba(239,68,68,0.16);
        color: #fca5a5;
        border: 1px solid rgba(239,68,68,0.35);
    }

    .pill-gray {
        background: rgba(148,163,184,0.16);
        color: #cbd5e1;
        border: 1px solid rgba(148,163,184,0.35);
    }

    .pill-blue {
        background: rgba(59,130,246,0.16);
        color: #93c5fd;
        border: 1px solid rgba(59,130,246,0.35);
    }

    .recommendation-box {
        border-radius: 22px;
        padding: 26px;
        text-align: center;
        font-weight: 800;
        font-size: 1.9rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.22);
        margin-bottom: 1rem;
    }

    .buy-box {
        background: linear-gradient(135deg, rgba(22,163,74,0.28), rgba(34,197,94,0.14));
        border: 1px solid rgba(34,197,94,0.35);
        color: #bbf7d0;
    }

    .watch-box {
        background: linear-gradient(135deg, rgba(234,179,8,0.26), rgba(250,204,21,0.12));
        border: 1px solid rgba(250,204,21,0.35);
        color: #fde68a;
    }

    .avoid-box {
        background: linear-gradient(135deg, rgba(220,38,38,0.26), rgba(239,68,68,0.12));
        border: 1px solid rgba(239,68,68,0.35);
        color: #fecaca;
    }

    .neutral-box {
        background: linear-gradient(135deg, rgba(100,116,139,0.26), rgba(148,163,184,0.12));
        border: 1px solid rgba(148,163,184,0.35);
        color: #e2e8f0;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 800;
        margin-bottom: 0.8rem;
        color: #f8fafc;
    }

    .circle-wrap {
        text-align: center;
        margin-bottom: 1rem;
    }

    .circle-label {
        margin-top: 0.6rem;
        font-weight: 700;
        color: #cbd5e1;
    }

    .score-caption {
        color: #94a3b8;
        font-size: 0.92rem;
        text-align: center;
        margin-top: 0.35rem;
    }

    .note-item {
        border-left: 4px solid #facc15;
        padding: 0.8rem 1rem;
        border-radius: 12px;
        background: rgba(250,204,21,0.08);
        color: #fef3c7;
        margin-bottom: 0.75rem;
    }

    .rationale-box {
        line-height: 1.7;
        color: #e5e7eb;
        font-size: 1rem;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    div[data-testid="stTextInput"] input {
        border-radius: 14px;
    }

    div[data-testid="stButton"] > button {
        width: 100%;
        border-radius: 14px;
        font-weight: 700;
        padding: 0.7rem 1rem;
        background: linear-gradient(135deg, #2563eb, #3b82f6);
        color: white;
        border: none;
    }

    div[data-testid="stButton"] > button:hover {
        filter: brightness(1.05);
    }
</style>
""", unsafe_allow_html=True)


ORDERED_KEYS = [
    "Momentum",
    "Volatility",
    "News Sentiment",
    "Fundamentals",
    "Investor Fit",
]


def clamp_score(value) -> int:
    try:
        return max(0, min(5, int(value)))
    except (TypeError, ValueError):
        return 0


def pill_class_for_confidence(value: str) -> str:
    value = (value or "").strip().lower()
    if value == "high":
        return "pill-green"
    if value == "medium":
        return "pill-yellow"
    if value == "low":
        return "pill-red"
    return "pill-gray"


def box_class_for_recommendation(value: str) -> str:
    value = (value or "").strip().lower()
    if value == "buy":
        return "buy-box"
    if value == "watchlist":
        return "watch-box"
    if value == "avoid":
        return "avoid-box"
    return "neutral-box"


def safe_text(value: str) -> str:
    return html.escape(str(value))


def score_color(score: int) -> str:
    if score >= 4:
        return "#22c55e"
    if score >= 2:
        return "#facc15"
    if score == 1:
        return "#fb7185"
    return "#94a3b8"


def score_caption(score: int) -> str:
    if score == 0:
        return "No data"
    return f"{int((score / 5) * 100)}% strength"


def render_score_circle(label: str, score: int):
    score = clamp_score(score)
    percent = int((score / 5) * 100)
    color = score_color(score)
    safe_label = safe_text(label)

    st.markdown(
        f"""
        <div class="glass-card circle-wrap">
            <div style="
                width: 120px;
                height: 120px;
                margin: 0 auto;
                border-radius: 50%;
                background:
                    conic-gradient({color} 0% {percent}%,
                    rgba(255,255,255,0.10) {percent}% 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
            ">
                <div style="
                    width: 84px;
                    height: 84px;
                    background: #0f172a;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    flex-direction: column;
                    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06);
                ">
                    <div style="font-size: 1.5rem; font-weight: 800; color: {color};">{score}</div>
                    <div style="font-size: 0.75rem; color: #94a3b8;">/ 5</div>
                </div>
            </div>
            <div class="circle-label">{safe_label}</div>
            <div class="score-caption">{score_caption(score)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overall_circle(scores: dict):
    values = [clamp_score(scores.get(k, 0)) for k in ORDERED_KEYS]
    avg_score = sum(values) / len(values) if values else 0
    percent = int((avg_score / 5) * 100)

    if avg_score >= 4:
        color = "#22c55e"
    elif avg_score >= 2:
        color = "#facc15"
    elif avg_score > 0:
        color = "#fb7185"
    else:
        color = "#94a3b8"

    caption = "No data available" if avg_score == 0 else f"{percent}% composite score"

    st.markdown(
        f"""
        <div class="glass-card circle-wrap">
            <div class="section-title">Overall Score</div>
            <div style="
                width: 170px;
                height: 170px;
                margin: 0 auto;
                border-radius: 50%;
                background:
                    conic-gradient({color} 0% {percent}%,
                    rgba(255,255,255,0.10) {percent}% 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
            ">
                <div style="
                    width: 122px;
                    height: 122px;
                    background: #0f172a;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    flex-direction: column;
                    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06);
                ">
                    <div style="font-size: 2rem; font-weight: 800; color: {color};">{avg_score:.1f}</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">average / 5</div>
                </div>
            </div>
            <div class="score-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def call_backend(user_query: str):
    response = None
    try:
        response = requests.post(
            BACKEND_URL,
            json={"query": user_query},
            timeout=120
        )
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as e:
        detail = ""
        try:
            if response is not None:
                detail = response.text
        except Exception:
            pass
        return None, f"{e}\n{detail}"


def render_result(data: dict, query_text: str):
    recommendation = str(data.get("recommendation", "Cannot determine"))
    confidence = str(data.get("confidence", "Cannot determine"))
    scores = data.get("scores", {}) if isinstance(data.get("scores"), dict) else {}
    for score in scores:
        if score == 0:
            recommendation = "Cannot determine"
            confidence = "Cannot determine"
    rationale = str(data.get("rationale", "No rationale provided."))
    risk_notes = data.get("risk_note", [])
    disclaimer = str(
        data.get(
            "disclaimer",
            "This analysis is for educational purposes only and not financial advice."
        )
    )

    if not isinstance(risk_notes, list):
        risk_notes = []

    normalized_scores = {key: clamp_score(scores.get(key, 0)) for key in ORDERED_KEYS}

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="small-muted">Analyzed Query</div>
            <div style="font-size: 1.08rem; font-weight: 600; margin-top: 0.35rem;">
                {safe_text(query_text)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    top_col1, top_col2 = st.columns([1.25, 1])

    with top_col1:
        st.markdown(
            f"""
            <div class="recommendation-box {box_class_for_recommendation(recommendation)}">
                {safe_text(recommendation)}
            </div>
            <div>
                <span class="metric-pill pill-blue">Recommendation</span>
                <span class="metric-pill {pill_class_for_confidence(confidence)}">
                    Confidence: {safe_text(confidence)}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with top_col2:
        render_overall_circle(normalized_scores)

    st.markdown('<div class="section-title">Score Breakdown</div>', unsafe_allow_html=True)
    score_cols = st.columns(len(ORDERED_KEYS))
    for col, key in zip(score_cols, ORDERED_KEYS):
        with col:
            render_score_circle(key, normalized_scores.get(key, 0))

    lower_col1, lower_col2 = st.columns([1.5, 1])

    with lower_col1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="section-title">Rationale</div>
                <div class="rationale-box">{safe_text(rationale)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with lower_col2:
        notes_html = ""
        if risk_notes:
            for note in risk_notes:
                notes_html += f'<div class="note-item">{safe_text(note)}</div>'
        else:
            notes_html = '<div class="small-muted">No risk notes provided.</div>'

        st.markdown(
            f"""
            <div class="glass-card">
                <div class="section-title">Risk Notes</div>
                {notes_html}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
        <div class="glass-card">
            <div class="small-muted">{safe_text(disclaimer)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Debug JSON"):
        st.json(data)


st.markdown("""
<div class="hero-card">
    <div style="font-size: 2.2rem; font-weight: 900; margin-bottom: 0.4rem;">
        IBM Stock Analysis Agent
    </div>
    <div class="small-muted" style="font-size: 1rem;">
        Enter a stock prompt and get a cleaner, visual AI analysis with recommendation strength,
        confidence, score circles, rationale, and risk notes.
    </div>
</div>
""", unsafe_allow_html=True)

query = st.text_input(
    "Stock analysis prompt",
    placeholder="Analyze VOO for a cautious investor with a long-term investment horizon."
)

if st.button("Analyze"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Analyzing..."):
            result, error = call_backend(query.strip())

        if error:
            st.error("Backend request failed")
            st.code(error)
        elif result:
            render_result(result, query)