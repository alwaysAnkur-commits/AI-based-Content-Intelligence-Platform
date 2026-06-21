# src/embedding_generator.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np, pandas as pd
import plotly.express as px

model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(texts: list, batch_size: int = 64) -> np.ndarray:
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                               convert_to_numpy=True, normalize_embeddings=True)
    return embeddings

def benchmark_embedding_quality(embeddings: np.ndarray, n_clusters: int = 8) -> dict:
    """Silhouette score: how well-separated are natural clusters in the embedding space — proxy for embedding quality."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels, sample_size=min(1000, len(embeddings)))
    return {"silhouette_score": round(score, 4), "n_clusters_tested": n_clusters}

def visualize_tsne(embeddings: np.ndarray, doc_ids: list, output_path: str):
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    coords = tsne.fit_transform(embeddings)
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "doc_id": doc_ids})
    fig = px.scatter(df, x="x", y="y", hover_data=["doc_id"], title="t-SNE: Document Embedding Space")
    fig.write_html(output_path)

if __name__ == "__main__":
    df = pd.read_csv("data/processed/full_processed_corpus.csv").fillna("")
    texts = df["body_clean"].tolist()
    doc_ids = df["doc_id"].tolist()

    embeddings = generate_embeddings(texts)
    np.save("data/processed/embeddings/corpus_embeddings.npy", embeddings)
    np.save("data/processed/embeddings/doc_ids.npy", np.array(doc_ids))

    quality = benchmark_embedding_quality(embeddings)
    with open("data/processed/embeddings/embedding_quality_report.md", "w") as f:
        f.write(f"# Embedding Quality Report\n\nModel: all-MiniLM-L6-v2 (384-dim)\n"
                f"Documents: {len(embeddings)}\nSilhouette score: {quality['silhouette_score']}\n")

    visualize_tsne(embeddings, doc_ids, "data/processed/embeddings/tsne_plot.html")
    print(f"Generated {embeddings.shape} embedding matrix. Silhouette: {quality['silhouette_score']}")