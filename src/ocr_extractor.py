import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path
import pandas as pd
from pathlib import Path
import Levenshtein

# ✅ Save intermediate images in data/raw/extracted_images
EXTRACTED_DIR = Path("data/raw/extracted_images")
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Grayscale -> adaptive threshold -> deskew, in that order."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 10)
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = thresh.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def ocr_image(image_path: str) -> dict:
    img = cv2.imread(image_path)
    processed = preprocess_for_ocr(img)
    config = "--oem 1 --psm 3"  # OEM 1 = LSTM engine, PSM 3 = auto segmentation
    text = pytesseract.image_to_string(processed, config=config)
    data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
    confs = [int(c) for c in data["conf"] if c != "-1"]
    avg_conf = sum(confs) / len(confs) if confs else 0
    return {"text": text, "avg_confidence": round(avg_conf, 2), "word_count": len(text.split())}

def ocr_pdf(pdf_path: str, dpi: int = 300) -> list:
    """300 DPI is Tesseract's recommended minimum for reliable accuracy."""
    pages = convert_from_path(pdf_path, dpi=dpi)
    results = []
    for i, page in enumerate(pages):
        # ✅ Save each page image in extracted_images folder
        tmp_path = EXTRACTED_DIR / f"page_{i}.png"
        page.save(tmp_path)
        result = ocr_image(str(tmp_path))
        result["page_number"] = i + 1
        results.append(result)
    return results

def calculate_cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate via Levenshtein edit distance."""
    return Levenshtein.distance(hypothesis, reference) / max(len(reference), 1)

def benchmark_against_ground_truth(ocr_results: list, ground_truth_dir: str) -> pd.DataFrame:
    rows = []
    for r in ocr_results:
        gt_path = Path(ground_truth_dir) / f"page_{r['page_number']}.txt"
        if gt_path.exists():
            gt = gt_path.read_text(encoding="utf-8")
            cer = calculate_cer(r["text"], gt)
            rows.append({
                "page": r["page_number"],
                "cer": round(cer, 4),
                "accuracy_pct": round((1 - cer) * 100, 2),
                "ocr_confidence": r["avg_confidence"]
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    results = ocr_pdf("data/raw/sample_scanned.pdf")
    pd.DataFrame(results).to_csv("data/processed/ocr_extraction_report.csv", index=False)
    print(f"OCR complete: {len(results)} pages")
