# src/dashboard_widgets/sentiment_comparison.py
import plotly.express as px
import pandas as pd

def build_sentiment_comparison_chart(sentiment_df: pd.DataFrame, category_df: pd.DataFrame):
    """Returns a Plotly figure — embed directly via st.plotly_chart(fig) in Streamlit."""
    merged = sentiment_df.merge(category_df[["doc_id", "predicted_label"]], on="doc_id", how="left")
    summary = merged.groupby(["predicted_label", "sentiment_label"]).size().reset_index(name="count")

    fig = px.bar(summary, x="predicted_label", y="count", color="sentiment_label", barmode="group",
                  color_discrete_map={"POSITIVE": "#2ECC71", "NEGATIVE": "#E74C3C"},
                  title="Sentiment Breakdown by Document Category",
                  labels={"predicted_label": "Category", "count": "Documents"})
    fig.update_layout(legend_title_text="Sentiment", height=450)
    return fig

if __name__ == "__main__":
    sentiment_df = pd.read_csv("data/processed/sentiment_scores.csv")
    category_df = pd.read_csv("data/processed/classification_report.csv")
    fig = build_sentiment_comparison_chart(sentiment_df, category_df)
    fig.write_html("data/processed/sentiment_comparison_chart.html")