# src/extraction_pipeline.py
import json, fitz
from pathlib import Path
from datetime import datetime
from table_extractor import extract_tables_from_pdf
from ocr_extractor import ocr_pdf
from image_captioning import caption_batch

def extract_native_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text

def is_scanned_pdf(pdf_path: str, text_threshold: int = 50) -> bool:
    """If native extraction yields very little text per page, it's likely scanned."""
    text = extract_native_text(pdf_path)
    doc = fitz.open(pdf_path)
    avg_chars = len(text) / max(len(doc), 1)
    doc.close()
    return avg_chars < text_threshold

def extract_all_modalities(pdf_path: str, image_dir: str = None) -> dict:
    """Single function: auto-detects PDF type and runs text/table/image extraction."""
    doc_id = Path(pdf_path).stem
    result = {"doc_id": doc_id, "source_pdf": pdf_path,
               "processed_at": datetime.now().isoformat(), "extraction_methods_used": []}

    if is_scanned_pdf(pdf_path):
        ocr_pages = ocr_pdf(pdf_path)
        result["text"] = "\n".join(p["text"] for p in ocr_pages)
        result["text_extraction_method"] = "ocr_tesseract"
        result["ocr_avg_confidence"] = sum(p["avg_confidence"] for p in ocr_pages) / len(ocr_pages)
        result["extraction_methods_used"].append("ocr")
    else:
        result["text"] = extract_native_text(pdf_path)
        result["text_extraction_method"] = "native_pymupdf"
        result["extraction_methods_used"].append("native_text")

    try:
        tables = extract_tables_from_pdf(pdf_path)
        result["tables"] = [{"page": t["page"], "shape": t["shape"], "accuracy": t["accuracy"]} for t in tables]
        result["table_count"] = len(tables)
        if tables:
            result["extraction_methods_used"].append("table_extraction")
    except Exception as e:
        result["tables"], result["table_count"] = [], 0
        result["table_extraction_error"] = str(e)

    if image_dir and Path(image_dir).exists():
        captions = caption_batch(image_dir)
        result["images"], result["image_count"] = captions, len(captions)
        if captions:
            result["extraction_methods_used"].append("image_captioning")
    else:
        result["images"], result["image_count"] = [], 0

    return result

def run_unified_extraction(pdf_dir: str, output_path: str = "data/processed/consolidated_extraction.json"):
    all_results = [extract_all_modalities(str(p)) for p in Path(pdf_dir).glob("*.pdf")]
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nUnified extraction complete: {len(all_results)} documents.")
    return all_results

if __name__ == "__main__":
    run_unified_extraction("data/raw/pdfs")