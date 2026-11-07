"""Download public speeches listed in data/sources.csv into data/raw/*.txt.

Each output file starts with a small header (TITLE / YEAR / URL) that ingest.py
turns into metadata, so every answer can cite its source.

Usage:  python -m src.scrape_speeches
Be polite: only scrape pages you are allowed to, and respect robots.txt / terms.
"""
import csv, io, re, time, pathlib
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

OUT = pathlib.Path("data/raw")
HEADERS = {"User-Agent": "lky-chatbot-research/1.0 (educational project)"}


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:80]


def extract_html(content: bytes) -> str:
    soup = BeautifulSoup(content, "html.parser")
    for t in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        t.decompose()
    node = soup.find("article") or soup.find("main") or soup.body or soup
    text = node.get_text("\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_pdf(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    return "\n\n".join((p.extract_text() or "") for p in reader.pages).strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open("data/sources.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(l for l in f if not l.lstrip().startswith("#")))
    if not rows:
        print("data/sources.csv kosong. Isi dulu dengan URL pidato publik.")
        return
    for r in rows:
        title, year, url = r["title"].strip(), r["year"].strip(), r["url"].strip()
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            is_pdf = url.lower().endswith(".pdf") or "pdf" in resp.headers.get("content-type", "")
            text = extract_pdf(resp.content) if is_pdf else extract_html(resp.content)
            if len(text) < 500:
                print(f"[skip] terlalu pendek: {title}")
                continue
            path = OUT / f"{year}-{slugify(title)}.txt"
            path.write_text(f"TITLE: {title}\nYEAR: {year}\nURL: {url}\n\n{text}", encoding="utf-8")
            print(f"[ok] {path.name} ({len(text)} chars)")
        except Exception as e:
            print(f"[fail] {title}: {e}")
        time.sleep(1.5)


if __name__ == "__main__":
    main()
