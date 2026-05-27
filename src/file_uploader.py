import os, magic, chardet
from pathlib import Path

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".csv", ".txt", ".jpg", ".jpeg", ".png"}
MAX_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB

def validate_file(filepath):
    path = Path(filepath)
    errors = []
    if not path.exists():
        return False, ["File not found"]
    if path.stat().st_size > MAX_SIZE_BYTES:
        errors.append(f"File too large: {path.stat().st_size / 1e6:.1f} MB (max 25 MB)")
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        errors.append(f"Unsupported extension: {path.suffix}")
    # MIME check
    mime = magic.from_file(filepath, mime=True)
    allowed_mimes = {"application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                     "text/csv", "text/plain", "image/jpeg", "image/png"}
    if mime not in allowed_mimes:
        errors.append(f"Unexpected MIME type: {mime}")
    return len(errors) == 0, errors

def read_file(filepath):
    path = Path(filepath)
    ext = path.suffix.lower()
    if ext == ".txt" or ext == ".csv":
        raw = open(filepath, "rb").read()
        enc = chardet.detect(raw)["encoding"] or "utf-8"
        return open(filepath, encoding=enc, errors="replace").read()
    elif ext == ".pdf":
        import fitz  # PyMuPDF
        doc = fitz.open(filepath)
        return "\n".join(page.get_text() for page in doc)
    elif ext == ".docx":
        from docx import Document
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext in {".jpg", ".jpeg", ".png"}:
        return f"[IMAGE FILE: {path.name}]"  # OCR handled in Week 4
    return ""

def upload_file(filepath):
    valid, errors = validate_file(filepath)
    if not valid:
        print(f"Validation failed: {errors}")
        return None
    text = read_file(filepath)
    print(f"Successfully read {filepath} ({len(text)} chars)")
    return text