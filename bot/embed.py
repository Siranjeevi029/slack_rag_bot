"""Build FAISS vector index from messages.md using thread-based chunks.

Each chunk = top-level message + replies (or standalone message) = one vector.
Metadata stored in chunks_meta.json alongside the index.

Run:  python -m bot.embed
"""
import json
import sys

import numpy as np

from bot import config
from bot.chunk import parse_chunks

EMBED_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
FAISS_INDEX = config.DATA_DIR / "faiss.index"
CHUNKS_META = config.DATA_DIR / "chunks_meta.json"
TOP_K = 200  # chunks retrieved per query — accuracy over token cost


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL}...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True,
                              normalize_embeddings=True)
    return embeddings.astype("float32")


def build_index(embeddings: np.ndarray):
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine sim (embeddings normalized)
    index.add(embeddings)
    return index


def main():
    if not config.MESSAGES_MD.exists():
        sys.exit("messages.md not found — run `python -m bot.scrape` first")

    chunks = parse_chunks(config.MESSAGES_MD)
    threads = sum(1 for c in chunks if c["is_thread"])
    print(f"Parsed {len(chunks)} chunks ({threads} threads, "
          f"{len(chunks) - threads} standalone)")

    embeddings = embed_chunks(chunks)
    index = build_index(embeddings)

    import faiss
    faiss.write_index(index, str(FAISS_INDEX))
    CHUNKS_META.write_text(json.dumps(chunks, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"Saved: {FAISS_INDEX}  ({len(chunks)} vectors, dim={embeddings.shape[1]})")
    print(f"Saved: {CHUNKS_META}")


if __name__ == "__main__":
    main()
