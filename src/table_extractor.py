import camelot
import pandas as pd
from pathlib import Path
import pdfplumber
import pikepdf

def extract_tables_from_pdf(pdf_path: str, flavor: str = "lattice") -> list:
    """
    Try extracting tables with Camelot. Fallback to stream flavor if lattice fails.
    If PDF is restricted, attempt to unlock with pikepdf. If still blocked, use pdfplumber.
    """
    results = []
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor=flavor)
        if len(tables) == 0 or tables[0].parsing_report.get("accuracy", 0) < 50:
            raise ValueError("Low accuracy, trying stream flavor")
    except Exception:
        try:
            tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        except Exception as e:
            # Fallback: try unlocking with pikepdf
            unlocked_path = pdf_path.replace(".pdf", "_unlocked.pdf")
            try:
                pdf = pikepdf.open(pdf_path)
                pdf.save(unlocked_path)
                tables = camelot.read_pdf(unlocked_path, pages="all", flavor="stream")
            except Exception:
                # Final fallback: pdfplumber text extraction
                tables = []
                with pdfplumber.open(pdf_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if text:
                            df = pd.DataFrame([line.split() for line in text.split("\n")])
                            results.append({
                                "table_index": i,
                                "page": page.page_number,
                                "dataframe": df,
                                "accuracy": 0,
                                "shape": df.shape
                            })
                return results

    # Collect Camelot results
    for i, table in enumerate(tables):
        results.append({
            "table_index": i,
            "page": table.page,
            "dataframe": table.df,
            "accuracy": table.parsing_report.get("accuracy", 0),
            "shape": table.df.shape
        })
    return results

def extract_tables_batch(pdf_dir: str, output_dir: str = "data/processed/tables") -> pd.DataFrame:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for pdf_file in Path(pdf_dir).glob("*.pdf"):
        tables = extract_tables_from_pdf(str(pdf_file))
        for t in tables:
            csv_name = f"{pdf_file.stem}_table{t['table_index']}_p{t['page']}.csv"
            t["dataframe"].to_csv(Path(output_dir) / csv_name, index=False)
            summary_rows.append({
                "source_pdf": pdf_file.name,
                "table_index": t["table_index"],
                "page": t["page"],
                "rows": t["shape"][0],
                "cols": t["shape"][1],
                "accuracy_pct": t["accuracy"],
                "output_file": csv_name
            })
        print(f"{pdf_file.name}: extracted {len(tables)} tables")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{output_dir}/extraction_summary.csv", index=False)
    avg_acc = summary_df["accuracy_pct"].mean() if not summary_df.empty else 0
    print(f"\nBatch complete. Average cell accuracy: {avg_acc:.1f}%")
    return summary_df

if __name__ == "__main__":
    extract_tables_batch("data/raw/pdfs")
