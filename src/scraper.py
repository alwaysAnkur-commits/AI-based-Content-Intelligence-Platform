import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os
from datetime import datetime

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}

def fetch_with_retry(url, retries=3, backoff=2):
    """Fetch URL with exponential backoff on failure."""
    for attempt in range(retries):
        try:
            time.sleep(random.uniform(1, 2))  # polite rate limiting
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            wait = backoff ** attempt
            print(f"Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    return None

def scrape_bbc(max_articles=10):
    articles = []
    url = "https://www.bbc.com/news"
    resp = fetch_with_retry(url)
    if not resp:
        return articles
    soup = BeautifulSoup(resp.text, "lxml")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/news/") and len(href) > 10:
            links.add("https://www.bbc.com" + href)
        if len(links) >= max_articles:
            break
    for link in list(links)[:max_articles]:
        resp = fetch_with_retry(link)
        if not resp:
            continue
        s = BeautifulSoup(resp.text, "lxml")
        title = s.find("h1")
        body_tags = s.find_all("p")
        body = " ".join(p.get_text() for p in body_tags[:10])
        articles.append({
            "title": title.get_text() if title else "",
            "body": body,
            "author": "",
            "date": datetime.now().isoformat(),
            "source": "BBC",
            "url": link,
            "scrape_timestamp": datetime.now().isoformat()
        })
    return articles

def scrape_techcrunch(max_articles=10):
    """Use RSS feed for reliable scraping."""
    articles = []
    feed_url = "https://techcrunch.com/feed/"
    resp = fetch_with_retry(feed_url)
    if not resp:
        return articles
    soup = BeautifulSoup(resp.text, "xml")
    items = soup.find_all("item")[:max_articles]
    for item in items:
        articles.append({
            "title": item.find("title").get_text() if item.find("title") else "",
            "body": item.find("description").get_text() if item.find("description") else "",
            "author": item.find("dc:creator").get_text() if item.find("dc:creator") else "",
            "date": item.find("pubDate").get_text() if item.find("pubDate") else "",
            "source": "TechCrunch",
            "url": item.find("link").get_text() if item.find("link") else "",
            "scrape_timestamp": datetime.now().isoformat()
        })
    return articles

def scrape_reuters(max_articles=10):
    """Use topNews RSS feed for reliability."""
    articles = []
    feed_url = "https://feeds.reuters.com/reuters/topNews"
    resp = fetch_with_retry(feed_url)
    if not resp:
        return articles
    soup = BeautifulSoup(resp.text, "xml")
    items = soup.find_all("item")[:max_articles]
    for item in items:
        articles.append({
            "title": item.find("title").get_text() if item.find("title") else "",
            "body": item.find("description").get_text() if item.find("description") else "",
            "author": "",
            "date": item.find("pubDate").get_text() if item.find("pubDate") else "",
            "source": "Reuters",
            "url": item.find("link").get_text() if item.find("link") else "",
            "scrape_timestamp": datetime.now().isoformat()
        })
    return articles

def run_all_scrapers():
    all_articles = []
    print("Scraping BBC...")
    all_articles.extend(scrape_bbc())
    print("Scraping TechCrunch...")
    all_articles.extend(scrape_techcrunch())
    print("Scraping Reuters...")
    all_articles.extend(scrape_reuters())

    df = pd.DataFrame(all_articles)
    os.makedirs("data/raw", exist_ok=True)
    file_path = "data/raw/scraped_articles.csv"

    # Append if file exists, else create new
    if os.path.exists(file_path):
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, index=False)

    print(f"Saved {len(df)} new articles to {file_path}")
    return df

if __name__ == "__main__":
    run_all_scrapers()

print("Run Successful!")
