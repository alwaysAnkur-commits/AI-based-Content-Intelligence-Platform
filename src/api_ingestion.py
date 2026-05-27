import os, json, praw
from newsapi import NewsApiClient
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

UNIFIED_SCHEMA_KEYS = ["title", "body", "author", "date", "source", "url", "scrape_timestamp"]

def normalize(record):
    """Ensure every record has all unified schema keys."""
    return {k: record.get(k, "") for k in UNIFIED_SCHEMA_KEYS}

def fetch_newsapi(query="technology", page_size=20):
    api = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))
    response = api.get_everything(q=query, language="en", page_size=page_size, sort_by="publishedAt")
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

def fetch_reddit(subreddits=["technology", "worldnews"], limit=10):
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent="content-intel-bot/1.0"
    )
    articles = []
    for sub in subreddits:
        for post in reddit.subreddit(sub).hot(limit=limit):
            articles.append(normalize({
                "title": post.title,
                "body": post.selftext,
                "author": str(post.author),
                "date": datetime.fromtimestamp(post.created_utc).isoformat(),
                "source": f"Reddit/r/{sub}",
                "url": f"https://reddit.com{post.permalink}",
                "scrape_timestamp": datetime.now().isoformat()
            }))
    return articles

def save_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(records)} records to {path}")

if __name__ == "__main__":
    news = fetch_newsapi()
    reddit = fetch_reddit()
    all_records = news + reddit
    save_jsonl(all_records, "data/raw/api_articles.jsonl")