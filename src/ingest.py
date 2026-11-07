"""Chunk + embed every .txt in data/raw (and data/sample) into ChromaDB.

Usage:  python -m src.ingest
"""
import glob, os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src import config


def parse_doc(path: str):
    raw = open(path, encoding="utf-8").read()
    meta = {"title": os.path.splitext(os.path.basename(path))[0], "year": "", "url": ""}
    head, sep, body = raw.partition("\n\n")
    if sep and head.startswith("TITLE:"):
        for line in head.splitlines():
            k, _, v = line.partition(":")
            if k.strip().lower() in meta:
                meta[k.strip().lower()] = v.strip()
        raw = body
    return meta, raw


def main():
    paths = sorted(glob.glob("data/raw/*.txt") + glob.glob("data/sample/*.txt"))
    if not paths:
        print("Tidak ada file .txt. Jalankan `python -m src.scrape_speeches` atau taruh teks di data/raw/.")
        return
    model = SentenceTransformer(config.EMBED_MODEL)
    client = chromadb.PersistentClient(config.DB_PATH)
    try:
        client.delete_collection(config.COLLECTION)  # idempotent re-ingest
    except Exception:
        pass
    col = client.create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)

    total = 0
    for path in paths:
        meta, text = parse_doc(path)
        chunks = splitter.split_text(text)
        if not chunks:
            continue
        base = os.path.basename(path)
        for i in range(0, len(chunks), 64):
            batch = chunks[i:i + 64]
            col.add(
                ids=[f"{base}-{i + j}" for j in range(len(batch))],
                documents=batch,
                embeddings=model.encode(batch).tolist(),
                metadatas=[{**meta, "file": base}] * len(batch),
            )
        total += len(chunks)
        print(f"{base}: {len(chunks)} chunks")
    print(f"Selesai. {total} chunks dari {len(paths)} dokumen.")


if __name__ == "__main__":
    main()
