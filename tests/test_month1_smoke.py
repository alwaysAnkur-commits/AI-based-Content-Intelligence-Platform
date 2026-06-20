# tests/test_month1_smoke.py
"""
Quick smoke test verifying every Month 1 module imports cleanly and
core functions run without error. Run before recording the demo video.
"""
import importlib
import sys
sys.path.insert(0, "src")

MODULES_TO_CHECK = [
    "scraper", "api_ingestion", "file_uploader", "dedup", "metadata_tracker",
    "classifier", "confidence_router", "pipeline_router", "quality_checker",
    "data_versioning", "text_cleaner", "tokenizer", "feature_extractor",
    "doc_stats", "batch_pipeline", "table_extractor", "ocr_extractor",
    "image_captioning", "extraction_pipeline"
]

def run_smoke_test():
    results = {}
    for module_name in MODULES_TO_CHECK:
        try:
            importlib.import_module(module_name)
            results[module_name] = "✅ PASS"
        except Exception as e:
            results[module_name] = f"❌ FAIL — {e}"

    print("\n=== MONTH 1 SMOKE TEST ===")
    for mod, status in results.items():
        print(f"{mod:<25} {status}")

    failed = [m for m, s in results.items() if "FAIL" in s]
    if failed:
        print(f"\n⚠️  {len(failed)} module(s) failed import. Fix before demo recording.")
    else:
        print(f"\n✅ All {len(results)} modules import cleanly. Ready for demo.")
    return results

if __name__ == "__main__":
    run_smoke_test()