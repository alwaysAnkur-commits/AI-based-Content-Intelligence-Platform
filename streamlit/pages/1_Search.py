# streamlit/pages/1_Search.py
import streamlit as st
import requests

st.set_page_config(page_title="Search", layout="wide")
st.title("🔍 Semantic Search")

with st.sidebar:
    st.header("Filters")
    source_filter = st.selectbox("Source", ["All", "BBC", "TechCrunch", "Reuters", "NewsAPI", "Reddit"])
    top_k = st.slider("Number of results", 5, 30, 10)

query = st.text_input("Search query", placeholder="e.g. artificial intelligence funding")

if st.button("Search") and query:
    with st.spinner("Searching..."):
        try:
            response = requests.post("http://localhost:8000/search",
                                      json={"query": query, "top_k": top_k}, timeout=10)
            response.raise_for_status()
            data = response.json()
            st.success(f"Found {data['result_count']} results")

            for r in data["results"]:
                with st.expander(f"📄 {r['doc_id']} — relevance: {r['fused_score']:.3f}"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Fused Score", f"{r['fused_score']:.3f}")
                    col2.metric("Semantic", f"{r['dense_score']:.3f}")
                    col3.metric("Keyword", f"{r['sparse_score']:.3f}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the search API. Make sure it's running: `uvicorn api.search_api:app --reload`")