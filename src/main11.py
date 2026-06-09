from scraper import run_all_scrapers
from api_ingestion import fetch_newsapi, fetch_pushshift, save_jsonl
from dedup import deduplicate
from metadata_tracker import bulk_register
import pandas as pd, json

print("=== STEP 1: Web Scraping ===")
df_scraped = run_all_scrapers()

print("\n=== STEP 2: API Ingestion ===")
api_records = fetch_newsapi() + fetch_pushshift()
save_jsonl(api_records, "data/raw/api_articles.jsonl")
df_api = pd.DataFrame(api_records)
df_api.to_csv("data/raw/api_articles.csv", index=False)

print("\n=== STEP 3: Combine & Deduplicate ===")
all_records = pd.concat([df_scraped, df_api]).to_dict(orient="records")
deduped, report = deduplicate(all_records)
pd.DataFrame(deduped).to_csv("data/processed/deduped_articles.csv", index=False)

print("\n=== STEP 4: Register Metadata ===")
bulk_register(deduped)

print("\n✅ Week 1 pipeline complete!")