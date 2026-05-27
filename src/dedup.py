import pandas as pd
import csv
from datasketch import MinHash, MinHashLSH

def get_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for word in text.lower().split():
        m.update(word.encode("utf8"))
    return m

def deduplicate(records, threshold=0.8):
    """
    records: list of dicts with 'title' and 'body' keys
    Returns: (deduplicated list, report list)
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    minhashes = {}
    for i, rec in enumerate(records):
        text = (rec.get("title", "") + " " + rec.get("body", "")).strip()
        m = get_minhash(text)
        minhashes[i] = m
        lsh.insert(str(i), m)

    duplicates = set()
    report = []
    for i, rec in enumerate(records):
        if i in duplicates:
            continue
        result = lsh.query(minhashes[i])
        for j in result:
            j = int(j)
            if j != i and j not in duplicates:
                sim = minhashes[i].jaccard(minhashes[j])
                report.append({"original_idx": i, "duplicate_idx": j, "similarity": round(sim, 3),
                                "original_title": records[i].get("title", ""),
                                "duplicate_title": records[j].get("title", "")})
                duplicates.add(j)

    deduped = [rec for i, rec in enumerate(records) if i not in duplicates]
    report_df = pd.DataFrame(report)
    report_df.to_csv("data/processed/dedup_report.csv", index=False)
    print(f"Original: {len(records)} | Duplicates removed: {len(duplicates)} | Remaining: {len(deduped)}")
    return deduped, report

if __name__ == "__main__":
    df = pd.read_csv("data/raw/scraped_articles.csv")
    records = df.to_dict(orient="records")
    deduped, report = deduplicate(records)
    pd.DataFrame(deduped).to_csv("data/processed/deduped_articles.csv", index=False)