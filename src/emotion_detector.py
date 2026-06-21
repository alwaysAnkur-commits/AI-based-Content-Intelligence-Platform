# src/emotion_detector.py
from transformers import pipeline
import pandas as pd
from tqdm import tqdm

emotion_pipe = pipeline("text-classification",
                          model="j-hartmann/emotion-english-distilroberta-base", top_k=None, device=-1)

def detect_emotions_batch(records: list) -> pd.DataFrame:
    rows = []
    for rec in tqdm(records, desc="Emotion detection"):
        text = str(rec.get("body_clean", "") or rec.get("body", ""))[:512]
        if len(text.strip()) < 5:
            continue
        results = emotion_pipe(text)[0]
        distribution = {r["label"]: round(r["score"], 4) for r in results}
        top_3 = sorted(distribution.items(), key=lambda x: -x[1])[:3]

        row = {"doc_id": rec.get("doc_id", ""), "source": rec.get("source", "")}
        row.update({f"emotion_{k}": v for k, v in distribution.items()})
        for i, (label, _) in enumerate(top_3, start=1):
            row[f"top_emotion_{i}"] = label
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("data/processed/emotion_scores.csv", index=False)
    print(f"\n{df['top_emotion_1'].value_counts()}")
    return df

def visualize_emotion_radar(doc_emotions: dict, doc_id: str, output_path: str = None):
    import plotly.graph_objects as go
    fig = go.Figure(data=go.Scatterpolar(r=list(doc_emotions.values()), theta=list(doc_emotions.keys()), fill='toself'))
    fig.update_layout(title=f"Emotion Profile — {doc_id}", polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    if output_path:
        fig.write_html(output_path)
    return fig

if __name__ == "__main__":
    df = pd.read_csv("data/processed/full_processed_corpus.csv").fillna("")
    detect_emotions_batch(df.to_dict(orient="records"))