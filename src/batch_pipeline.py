import time
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from text_cleaner import clean_text
from tokenizer import tokenize_and_lemmatize, detect_language
from doc_stats import compute_stats
import os
from datetime import datetime

def process_single_document(record: dict, run_id: str) -> dict:
    """
    Full pipeline for one document:
    clean → detect language → tokenize → lemmatize → compute stats
    """
    doc_id = record.get("doc_id", "")
    raw_body = str(record.get("body", "") or "")
    raw_title = str(record.get("title", "") or "")
    
    # Stage 1: Clean
    clean_body = clean_text(raw_body)
    clean_title = clean_text(raw_title, lowercase=False)
    
    # Stage 2: Tokenize + Lemmatize
    tok_result = tokenize_and_lemmatize(clean_body)
    
    # Stage 3: Doc stats
    stats = compute_stats(clean_body)
    
    return {
        "run_id": run_id,
        "doc_id": doc_id,
        "source": record.get("source", ""),
        "title_clean": clean_title,
        "body_clean": clean_body,
        "detected_lang": tok_result["lang"],
        "tokens": " ".join(tok_result["tokens"]),
        "lemmas": " ".join(tok_result["lemmas"]),
        "token_count": len(tok_result["tokens"]),
        **stats  # unpack all stats columns
    }

def run_batch_pipeline(input_csv: str, output_csv: str, max_workers: int = None) -> pd.DataFrame:
    """
    Run the full NLP preprocessing pipeline on all documents.
    Uses ProcessPoolExecutor for CPU-bound multiprocessing.
    Target: 500+ docs in under 10 minutes.
    """
    df = pd.read_csv(input_csv).fillna("")
    records = df.to_dict(orient="records")
    total = len(records)
    
    print(f"\nBatch Pipeline Starting: {total} documents")
    print(f"Workers: {max_workers or os.cpu_count()}")
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.perf_counter()
    results = []
    
    # Use ProcessPoolExecutor for true parallelism on CPU-bound NLP work
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_document, rec, run_id): i 
                   for i, rec in enumerate(records)}
        
        with tqdm(total=total, desc="Processing") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    idx = futures[future]
                    print(f"\nError on doc {idx}: {e}")
                    results.append({
                        "run_id": run_id,
                        "doc_id": records[idx].get("doc_id", ""),
                        "error": str(e)
                    })
                pbar.update(1)
    
    elapsed = time.perf_counter() - start_time
    docs_per_sec = total / elapsed
    
    print(f"\n✅ Pipeline complete!")
    print(f"   Time: {elapsed:.1f}s ({docs_per_sec:.1f} docs/sec)")
    print(f"   Estimated for 500 docs: {500/docs_per_sec:.1f}s")
    
    result_df = pd.DataFrame(results)
    
    # Append mode with deduplication by doc_id
    try:
        existing = pd.read_csv(output_csv)
        result_df = pd.concat([existing, result_df], ignore_index=True)
        result_df = result_df.drop_duplicates(subset=["doc_id"], keep="last")
    except FileNotFoundError:
        pass
    
    result_df.to_csv(output_csv, index=False)
    print(f"Updated {output_csv} with {len(result_df)} total documents.")
    
    # Timing log (append mode)
    with open("data/processed/batch_timing_log.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | run_id={run_id} | {total} docs | {elapsed:.2f}s | {docs_per_sec:.1f} docs/sec\n")
    
    return result_df

if __name__ == "__main__":
    run_batch_pipeline(
        input_csv="data/processed/deduped_articles.csv",
        output_csv="data/processed/full_processed_corpus.csv"
    )
