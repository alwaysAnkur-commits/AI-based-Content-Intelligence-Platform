# src/ner_benchmark.py
from transformers import pipeline as hf_pipeline
import spacy
import pandas as pd
import time
from seqeval.metrics import precision_score, recall_score, f1_score
from flair.data import Sentence
from flair.models import SequenceTagger

nlp_spacy = spacy.load("en_core_web_sm")
bert_ner = hf_pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
flair_tagger = SequenceTagger.load("flair/ner-english-large")  # FLERT-based

def run_spacy_ner(text): return [(e.text, e.label_) for e in nlp_spacy(text).ents]
def run_bert_ner(text): return [(r["word"], r["entity_group"]) for r in bert_ner(text)]
def run_flert_ner(text):
    s = Sentence(text)
    flair_tagger.predict(s)
    return [(span.text, span.tag) for span in s.get_spans("ner")]

def benchmark_models(test_set: list) -> pd.DataFrame:
    """test_set: [{'text': ..., 'gold_entities': [(text, label), ...]}, ...] — built from 50 annotated docs."""
    models = {"spaCy": run_spacy_ner, "BERT-NER": run_bert_ner, "FLERT": run_flert_ner}
    rows = []

    for name, fn in models.items():
        all_pred, all_gold = [], []
        start = time.perf_counter()
        for ex in test_set:
            pred_labels = [p[1] for p in fn(ex["text"])]
            gold_labels = [g[1] for g in ex["gold_entities"]]
            max_len = max(len(pred_labels), len(gold_labels), 1)
            pred_labels += ["O"] * (max_len - len(pred_labels))
            gold_labels += ["O"] * (max_len - len(gold_labels))
            all_pred.append(pred_labels)
            all_gold.append(gold_labels)
        elapsed = (time.perf_counter() - start) / len(test_set)

        rows.append({"model": name, "precision": round(precision_score(all_gold, all_pred), 3),
                      "recall": round(recall_score(all_gold, all_pred), 3),
                      "f1_score": round(f1_score(all_gold, all_pred), 3),
                      "avg_inference_time_sec": round(elapsed, 4)})

    df = pd.DataFrame(rows)
    df.to_csv("data/processed/ner_benchmark_results.csv", index=False)
    with open("docs/ner_benchmark.md", "w") as f:
        f.write("# NER Model Benchmark\n\n" + df.to_markdown(index=False))
    print(df)
    return df

if __name__ == "__main__":
    test_set = [{"text": "Tim Cook met officials in Berlin on March 5.",
                 "gold_entities": [("Tim Cook", "PER"), ("Berlin", "LOC"), ("March 5", "DATE")]}]
    benchmark_models(test_set)