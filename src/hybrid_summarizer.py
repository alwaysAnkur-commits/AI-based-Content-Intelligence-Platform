# src/hybrid_summarizer.py
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import pandas as pd
from summarizer import summarize_text, evaluate_summary
from tqdm import tqdm

def textrank_extract(text: str, num_sentences: int = 5) -> str:
    """Stage 1: pick the most important sentences via TextRank (graph-based, no model needed)."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    sentences = summarizer(parser.document, num_sentences)
    return " ".join(str(s) for s in sentences)

def hybrid_summarize(text: str) -> str:
    """Stage 2: feed TextRank's extracted sentences into BART for abstractive refinement."""
    extracted = textrank_extract(text, num_sentences=5)
    if len(extracted.split()) < 20:
        return extracted
    return summarize_text(extracted, max_len=100, min_len=25)

def compare_methods(records: list) -> pd.DataFrame:
    rows = []
    for rec in tqdm(records, desc="Hybrid summarization comparison"):
        text = str(rec.get("body_clean", "") or rec.get("body", ""))
        if len(text.split()) < 50:
            continue
        extractive_only = textrank_extract(text)
        abstractive_only = summarize_text(text)
        hybrid = hybrid_summarize(text)
        reference = ". ".join(text.split(". ")[:2])
        rows.append({"doc_id": rec.get("doc_id", ""),
                      "extractive_only": extractive_only, "abstractive_only": abstractive_only,
                      "hybrid": hybrid,
                      "extractive_rouge1": evaluate_summary(extractive_only, reference).get("rouge1", 0),
                      "abstractive_rouge1": evaluate_summary(abstractive_only, reference).get("rouge1", 0),
                      "hybrid_rouge1": evaluate_summary(hybrid, reference).get("rouge1", 0)})
    df = pd.DataFrame(rows)
    df.to_csv("data/processed/hybrid_summary_comparison.csv", index=False)
    print(df[["extractive_rouge1", "abstractive_rouge1", "hybrid_rouge1"]].mean())
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/full_processed_corpus.csv").fillna("")
    compare_methods(df.to_dict(orient="records")[:30])