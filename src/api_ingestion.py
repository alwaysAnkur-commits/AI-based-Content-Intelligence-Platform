import os
import json
import requests
from newsapi import NewsApiClient
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

UNIFIED_SCHEMA_KEYS = [
    "title", "body", "author", "date", "source", "url", "scrape_timestamp"
]

def normalize(record):
    """Ensure every record has all unified schema keys."""
    return {k: record.get(k, "") for k in UNIFIED_SCHEMA_KEYS}

def fetch_newsapi(query="technology", page_size=20):
    api = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))
    response = api.get_everything(
        q=query,
        language="en",
        page_size=page_size,
        sort_by="publishedAt"
    )
    articles = []
    for a in response.get("articles", []):
        articles.append(normalize({
            "title": a.get("title", ""),
            "body": a.get("content", "") or a.get("description", ""),
            "author": a.get("author", ""),
            "date": a.get("publishedAt", ""),
            "source": a.get("source", {}).get("name", "NewsAPI"),
            "url": a.get("url", ""),
            "scrape_timestamp": datetime.now().isoformat()
        }))
    return articles

def fetch_pushshift(subreddits=["technology", "worldnews"], size=10):
    url = "https://api.pushshift.io/reddit/search/submission/"
    articles = []
    for sub in subreddits:
        params = {
            "subreddit": sub,
            "size": size,
            "sort": "desc",
            "sort_type": "created_utc"
        }
        response = requests.get(url, params=params)
        data = response.json().get("data", [])
        for post in data:
            articles.append(normalize({
                "title": post.get("title", ""),
                "body": post.get("selftext", ""),
                "author": post.get("author", ""),
                "date": datetime.fromtimestamp(post.get("created_utc", 0)).isoformat(),
                "source": f"Pushshift/r/{sub}",
                "url": post.get("url", ""),
                "scrape_timestamp": datetime.now().isoformat()
            }))
    return articles

def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(records)} records to {path}")

if __name__ == "__main__":
    news = fetch_newsapi()
    reddit = fetch_pushshift()
    all_records = news + reddit
    save_jsonl(all_records, "data/raw/api_articles.jsonl")
