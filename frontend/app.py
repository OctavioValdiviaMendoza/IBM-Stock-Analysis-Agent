import os
import requests
import streamlit as st

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000/analyze"
)

st.set_page_config(page_title="IBM Stock Analysis Agent", layout="wide")

st.title("IBM Stock Analysis Agent")
st.write("Enter a stock analysis query and get AI-powered insights.")

query = st.text_input(
    "Enter your query",
    placeholder="Analyze VOO for a cautious investor with a long-term investment horizon."
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
    st.subheader("Query")
    st.info(query_text)

    col1, col2 = st.columns(2)

    with col1:
        recommendation = data.get("recommendation", "Watchlist")
        if recommendation == "Buy":
            st.success(f"Recommendation: {recommendation}")
        elif recommendation == "Watchlist":
            st.warning(f"Recommendation: {recommendation}")
        else:
            st.error(f"Recommendation: {recommendation}")

    with col2:
        st.markdown(f"**Confidence:** {data.get('confidence', 'Medium')}")

    st.subheader("Score Overview")
    scores = data.get("scores", {})
    ordered_keys = [
        "Momentum",
        "Volatility",
        "News Sentiment",
        "Fundamentals",
        "Investor Fit",
    ]

    for key in ordered_keys:
        value = int(scores.get(key, 1))
        st.progress(value / 5, text=f"{key}: {value}/5")

    st.subheader("Rationale")
    st.write(data.get("rationale", "No rationale provided."))

    st.subheader("Risk Notes")
    risk_notes = data.get("risk_note", [])
    if risk_notes:
        for note in risk_notes:
            st.warning(note)
    else:
        st.info("No risk notes provided.")

    st.caption(
        data.get(
            "disclaimer",
            "This analysis is for educational purposes only and not financial advice."
        )
    )

    with st.expander("Debug JSON"):
        st.json(data)

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