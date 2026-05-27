import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
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
    articles = []
    url = "https://techcrunch.com"
    resp = fetch_with_retry(url)
    if not resp:
        return articles
    soup = BeautifulSoup(resp.text, "lxml")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "techcrunch.com/20" in href:
            links.add(href)
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
            "source": "TechCrunch",
            "url": link,
            "scrape_timestamp": datetime.now().isoformat()
        })
    return articles

def scrape_reuters(max_articles=10):
    # Reuters blocks scrapers heavily — use their public RSS feed instead
    feed_url = "https://feeds.reuters.com/reuters/technologyNews"
    resp = fetch_with_retry(feed_url)
    articles = []
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
    df.to_csv("data/raw/scraped_articles.csv", index=False)
    print(f"Saved {len(df)} articles to data/raw/scraped_articles.csv")
    return df

if __name__ == "__main__":
    run_all_scrapers()

print("Run Successful!")