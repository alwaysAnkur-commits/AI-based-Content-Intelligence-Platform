# src/topic_modeler.py
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import pyLDAvis, pyLDAvis.gensim_models
import pandas as pd

def build_lda(corpus_lemmas: list, k_range=(5, 8, 10, 12, 15)) -> dict:
    tokenized = [str(doc).split() for doc in corpus_lemmas if str(doc).strip()]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.95)
    bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized]

    best_k, best_score, best_model = None, -1, None
    coherence_log = []
    for k in k_range:
        model = LdaModel(corpus=bow_corpus, id2word=dictionary, num_topics=k,
                          passes=20, iterations=400, alpha="auto", eta="auto", random_state=42)
        coherence_model = CoherenceModel(model=model, texts=tokenized, dictionary=dictionary, coherence="c_v")
        score = coherence_model.get_coherence()
        coherence_log.append({"k": k, "coherence": round(score, 4)})
        print(f"K={k}: coherence={score:.4f}")
        if score > best_score:
            best_k, best_score, best_model = k, score, model

    return {"model": best_model, "dictionary": dictionary, "bow_corpus": bow_corpus,
            "best_k": best_k, "best_score": best_score, "coherence_log": coherence_log}

def assign_topics(model, bow_corpus, doc_ids: list) -> pd.DataFrame:
    rows = []
    for doc_id, bow in zip(doc_ids, bow_corpus):
        topic_dist = model.get_document_topics(bow)
        if not topic_dist:
            continue
        dominant_topic, dominant_score = max(topic_dist, key=lambda x: x[1])
        top_words = ", ".join([w for w, _ in model.show_topic(dominant_topic, topn=8)])
        rows.append({"doc_id": doc_id, "dominant_topic": dominant_topic,
                      "topic_confidence": round(dominant_score, 4), "topic_keywords": top_words})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = pd.read_csv("data/processed/tokenized_corpus.csv").fillna("")
    result = build_lda(df["lemmas"].tolist())
    print(f"\nBest K={result['best_k']} (coherence={result['best_score']:.4f})")

    result["model"].save("data/processed/lda_model.gensim")
    topics_df = assign_topics(result["model"], result["bow_corpus"], df["doc_id"].tolist())
    topics_df.to_csv("data/processed/lda_topics.csv", index=False)

    vis = pyLDAvis.gensim_models.prepare(result["model"], result["bow_corpus"], result["dictionary"])
    pyLDAvis.save_html(vis, "data/processed/lda_vis.html")
    print("Saved lda_vis.html — open in browser to explore topics interactively")