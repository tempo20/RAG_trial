import json
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import requests
import feedparser
from bs4 import BeautifulSoup
from tqdm import tqdm

RSS_SOURCES = [
    {
        "name": "CNBC",
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362",
    },
    {
        "name": "BBC",
        "url": "https://feeds.bbci.co.uk/news/world/rss.xml",
    },
    {
        "name": "Nasdaq",
        "url": "https://www.nasdaq.com/feed/rssoutbound?category=Markets",
    },
    {
        "name": "Nasdaq",
        "url": "https://www.nasdaq.com/feed/rssoutbound?category=Investing",
    },
    # {
    #     "name": "Stockbiz",
    #     "url": "http://en.stockbiz.vn/RSS/News/Financial.ashx",
    # },
    # {
    #     "name": "Stockbiz",
    #     "url": "http://en.stockbiz.vn/RSS/News/Market.ashx",
    # },
    {
        "name": "CBS MoneyWatch",
        "url": "https://www.cbsnews.com/latest/rss/moneywatch",
    },
]
OUTPUT_JSON = "cnbc_articles.json"
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_REQUESTS = 0.8  # be polite to server


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # remove scripts/styles/nav/noisy elements
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    # CNBC pages often carry text in article-related containers; fallback to all <p>
    candidates = []
    for selector in [
        "div.ArticleBody-articleBody",
        "div.group",
        "article",
        "div[data-module='ArticleBody']",
    ]:
        candidates.extend(soup.select(selector))

    text_parts = []
    if candidates:
        for c in candidates:
            ps = c.find_all("p")
            for p in ps:
                t = p.get_text(" ", strip=True)
                if t:
                    text_parts.append(t)
    else:
        for p in soup.find_all("p"):
            t = p.get_text(" ", strip=True)
            if t:
                text_parts.append(t)

    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for t in text_parts:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    text = "\n".join(deduped).strip()

    # Strip trailing boilerplate some sites append
    noise_markers = [
        "This data feed is not available",
    ]
    for marker in noise_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].strip()

    return text


def scrape_article(url: str, session: requests.Session) -> Dict[str, Any]:
    result = {
        "url": url,
        "title": None,
        "published": None,
        "text": None,
        "status": "ok",
        "error": None,
    }
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # title
        title = None
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
        if not title and soup.title:
            title = soup.title.get_text(strip=True)

        # published date
        published = None
        pub_meta = (
            soup.find("meta", property="article:published_time")
            or soup.find("meta", attrs={"name": "date"})
            or soup.find("time")
        )
        if pub_meta:
            if pub_meta.name == "time":
                published = pub_meta.get("datetime") or pub_meta.get_text(strip=True)
            else:
                published = pub_meta.get("content")

        text = extract_main_text(resp.text)

        result["title"] = title
        result["published"] = published
        result["text"] = text
        if not text:
            result["status"] = "empty_text"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def main():
    links = []
    seen = set()
    source_by_link = {}
    for source in RSS_SOURCES:
        try:
            resp = requests.get(
                source["url"],
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
        except Exception as e:
            print(f"Warning: failed to fetch RSS from {source['name']}: {e}")
            continue
        # RSS entries -> unique links across all sources
        for entry in feed.entries:
            link = entry.get("link")
            if link and link not in seen:
                seen.add(link)
                links.append(link)
                source_by_link[link] = source

    print(f"Found {len(links)} unique links in {len(RSS_SOURCES)} RSS feeds.")

    session = requests.Session()
    articles: List[Dict[str, Any]] = []

    for link in tqdm(links, desc="Scraping articles"):
        article = scrape_article(link, session)
        source_meta = source_by_link.get(link, {})
        article["source"] = source_meta.get("name")
        article["source_rss"] = source_meta.get("url")
        articles.append(article)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    payload = {
        "source_rss": [source["url"] for source in RSS_SOURCES],
        "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
        "count": len(articles),
        "articles": articles,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(articles)} articles to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()