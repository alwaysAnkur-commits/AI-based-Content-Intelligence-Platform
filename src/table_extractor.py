# # src/table_extractor.py
import camelot
import pandas as pd
from pathlib import Path

def extract_tables_from_pdf(pdf_path: str, flavor: str = "lattice") -> list:
    """
    flavor='lattice' — works when tables have visible ruled borders (bank
    statements, formal invoices). flavor='stream' — works when columns are
    separated by whitespace with no visible lines (most exported reports).
    """
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor=flavor)
        if len(tables) == 0 or tables[0].parsing_report["accuracy"] < 50:
            raise ValueError("Low accuracy, trying stream flavor")
    except Exception:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")

    results = []
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
                "source_pdf": pdf_file.name, "table_index": t["table_index"],
                "page": t["page"], "rows": t["shape"][0], "cols": t["shape"][1],
                "accuracy_pct": t["accuracy"], "output_file": csv_name
            })
        print(f"{pdf_file.name}: extracted {len(tables)} tables")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{output_dir}/extraction_summary.csv", index=False)
    avg_acc = summary_df["accuracy_pct"].mean() if not summary_df.empty else 0
    print(f"\nBatch complete. Average cell accuracy: {avg_acc:.1f}%")
    return summary_df

if __name__ == "__main__":
    extract_tables_batch("data/raw/pdfs")


