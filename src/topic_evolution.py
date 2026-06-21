# src/topic_evolution.py
import pandas as pd
import plotly.express as px

def track_topic_evolution(topics_df: pd.DataFrame, corpus_df: pd.DataFrame) -> pd.DataFrame:
    merged = topics_df.merge(corpus_df[["doc_id", "date"]], on="doc_id", how="left")
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.dropna(subset=["date"])
    merged["week"] = merged["date"].dt.to_period("W").astype(str)

    weekly_counts = merged.groupby(["week", "dominant_topic"]).size().reset_index(name="doc_count")
    weekly_counts.to_csv("data/processed/topic_evolution.csv", index=False)
    return weekly_counts

def identify_trending(weekly_counts: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    pivot = weekly_counts.pivot(index="week", columns="dominant_topic", values="doc_count").fillna(0)
    if len(pivot) < 2:
        print("Not enough weeks of data to compute trend direction.")
        return pd.DataFrame()
    growth = (pivot.iloc[-1] - pivot.iloc[0])
    trending = growth.sort_values(ascending=False).head(top_n)
    print(f"\nTop {top_n} rising topics:\n{trending}")
    return trending

def visualize(weekly_counts: pd.DataFrame):
    fig = px.line(weekly_counts, x="week", y="doc_count", color="dominant_topic",
                  title="Topic Prevalence Over Time")
    fig.write_html("data/processed/topic_evolution_chart.html")

if __name__ == "__main__":
    topics_df = pd.read_csv("data/processed/lda_topics.csv")
    corpus_df = pd.read_csv("data/processed/full_processed_corpus.csv").fillna("")
    weekly = track_topic_evolution(topics_df, corpus_df)
    if not weekly.empty:
        identify_trending(weekly)
        visualize(weekly)