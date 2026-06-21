# src/sentiment_trends.py
import pandas as pd

def aggregate_sentiment_trends(sentiment_df: pd.DataFrame, metadata_df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    # ✅ Create numeric sentiment score from label + confidence
    sentiment_df["sentiment_numeric"] = sentiment_df["sentiment_label"].map({"POSITIVE": 1, "NEGATIVE": -1})
    sentiment_df["sentiment_score"] = sentiment_df["sentiment_numeric"] * sentiment_df["sentiment_confidence"]

    # ✅ Merge with metadata
    required_cols = ["doc_id", "date", "source"]
    available_cols = [c for c in required_cols if c in metadata_df.columns]
    if "doc_id" not in available_cols:
        raise ValueError("metadata_df must contain a 'doc_id' column for merging.")

    merged = sentiment_df.merge(metadata_df[available_cols], on="doc_id", how="left")

    # ✅ Handle date column if present
    if "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
        merged = merged.dropna(subset=["date"])

    # ✅ Resample trends by frequency (default daily)
    group_cols = [c for c in ["source"] if c in merged.columns]
    if "date" in merged.columns:
        trend = (merged.set_index("date")
                        .groupby(group_cols)
                        .resample(freq)["sentiment_score"]
                        .mean()
                        .reset_index())
    else:
        # fallback: overall average sentiment
        trend = pd.DataFrame({
            "overall_avg_sentiment": [merged["sentiment_score"].mean()]
        })

    trend.to_csv("data/processed/sentiment_trends.csv", index=False)
    print(f"✅ Sentiment trends saved: {trend.shape[0]} rows")
    return trend

def detect_anomalies(trend_df: pd.DataFrame, z_threshold: float = 2.5, window: int = 7) -> pd.DataFrame:
    """Rolling z-score anomaly detection — flags days deviating >z_threshold std devs from rolling mean."""
    anomalies = []
    if "date" not in trend_df.columns or "sentiment_score" not in trend_df.columns:
        print("⚠️ No date or sentiment_score column available for anomaly detection.")
        return pd.DataFrame()

    for source, group in trend_df.groupby("source"):
        group = group.sort_values("date").copy()
        group["rolling_mean"] = group["sentiment_score"].rolling(window, min_periods=3).mean()
        group["rolling_std"] = group["sentiment_score"].rolling(window, min_periods=3).std()
        group["z_score"] = (group["sentiment_score"] - group["rolling_mean"]) / group["rolling_std"]

        for _, row in group[group["z_score"].abs() > z_threshold].iterrows():
            anomalies.append({
                "source": source,
                "date": row["date"],
                "sentiment": row["sentiment_score"],
                "z_score": round(row["z_score"], 2),
                "type": "spike" if row["z_score"] > 0 else "drop"
            })

    anomaly_df = pd.DataFrame(anomalies)
    anomaly_df.to_csv("data/processed/sentiment_anomalies.csv", index=False)
    print(f"✅ Detected {len(anomaly_df)} anomaly days.")
    return anomaly_df

if __name__ == "__main__":
    sentiment_df = pd.read_csv("data/processed/sentiment_scores.csv")
    metadata_df = pd.read_csv("data/processed/full_processed_corpus.csv")

    trend_df = aggregate_sentiment_trends(sentiment_df, metadata_df)
    anomaly_df = detect_anomalies(trend_df)
