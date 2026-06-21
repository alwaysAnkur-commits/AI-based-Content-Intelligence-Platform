# src/multilingual_pipeline.py
import pandas as pd, time
from lang_detector import detect_language
from translator import translate_text
from summarizer import summarize_text
from tqdm import tqdm

def process_document(rec: dict) -> dict:
    doc_id = rec.get("doc_id", "")
    text = str(rec.get("body_clean", "") or rec.get("body", ""))
    log = {"doc_id": doc_id}
    t0 = time.perf_counter()

    detection = detect_language(text)
    log["detected_lang"] = detection["lang"]
    log["t_detect_sec"] = round(time.perf_counter() - t0, 3)

    t1 = time.perf_counter()
    if detection["lang"] != "en" and detection["confidence"] > 0.7:
        translated = translate_text(text, detection["lang"], "en")
    else:
        translated = text
    log["t_translate_sec"] = round(time.perf_counter() - t1, 3)

    t2 = time.perf_counter()
    summary = summarize_text(translated) if len(translated.split()) > 50 else translated
    log["t_summarize_sec"] = round(time.perf_counter() - t2, 3)

    log["summary"] = summary
    log["total_time_sec"] = round(time.perf_counter() - t0, 3)
    log["status"] = "success"
    return log

def run_multilingual_pipeline(records: list) -> pd.DataFrame:
    logs = []
    for rec in tqdm(records, desc="Multilingual pipeline"):
        try:
            logs.append(process_document(rec))
        except Exception as e:
            logs.append({"doc_id": rec.get("doc_id", ""), "status": "failed", "error": str(e)})
    df = pd.DataFrame(logs)
    df.to_csv("data/processed/multilingual_pipeline_log.csv", index=False)
    print(f"Processed {len(df)} docs. Success rate: {(df['status']=='success').mean()*100:.1f}%")
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/full_processed_corpus.csv").fillna("")