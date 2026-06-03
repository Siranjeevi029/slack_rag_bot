"""Hybrid RAG: vector (FAISS) + keyword (BM25) fused via Reciprocal Rank Fusion.

Index built by:  python -m bot.embed
"""
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from bot import config
from bot.llm import complete_ex
from bot.embed import FAISS_INDEX, CHUNKS_META, EMBED_MODEL, TOP_K

IST = timezone(timedelta(hours=5, minutes=30))
OUTPUT_FILE = config.ROOT / "output.txt"
RRF_K = 60  # standard RRF constant


def load_prompts() -> dict:
    text = config.PROMPT_FILE.read_text(encoding="utf-8")
    parts = re.split(r"^=== (\w+) ===$", text, flags=re.M)
    prompts = {}
    for i in range(1, len(parts), 2):
        prompts[parts[i]] = parts[i + 1].strip()
    for required in ("ANSWER", "ANSWER_THREAD"):
        if required not in prompts:
            raise ValueError(f"prompt file missing '=== {required} ===' section")
    return prompts


def _now_ist() -> str:
    return datetime.now(tz=IST).strftime("%Y-%m-%d %H:%M IST (%A)")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class RAGBot:
    def __init__(self):
        import faiss
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        if not FAISS_INDEX.exists():
            raise FileNotFoundError(
                f"{FAISS_INDEX} not found — run `python -m bot.embed` first")
        if not CHUNKS_META.exists():
            raise FileNotFoundError(
                f"{CHUNKS_META} not found — run `python -m bot.embed` first")

        print("Loading FAISS index, BM25, and embedding model...")
        self.index = faiss.read_index(str(FAISS_INDEX))
        self.chunks: list[dict] = json.loads(
            CHUNKS_META.read_text(encoding="utf-8"))
        self.model = SentenceTransformer(EMBED_MODEL)
        self.prompts = load_prompts()

        # BM25 built in-memory from chunk texts (no persistence needed, fast to rebuild)
        corpus = [_tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)

        # Roster: actual channel member names → token index for case-insensitive,
        # capitalization-independent, fuzzy name matching in queries.
        self.roster: set[str] = set()
        for c in self.chunks:
            for u in c.get("users", []):
                self.roster.add(u)
        self._name_tokens: dict[str, set[str]] = {}
        for name in self.roster:
            if "bot" in name.lower():  # skip Status Bot etc.
                continue
            for tok in re.findall(r"\w+", name.lower()):
                if len(tok) >= 3:
                    self._name_tokens.setdefault(tok, set()).add(name)

        print(f"Index ready: {self.index.ntotal} vectors, "
              f"{len(self.chunks)} chunks, {len(self.roster)} members")

    def _vector_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return (idx, score) list, sorted by score desc."""
        vec = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(vec, k)
        return [(int(idx), float(score))
                for score, idx in zip(scores[0], indices[0])
                if idx >= 0]

    def _bm25_search(self, query: str, k: int) -> list[int]:
        """Return chunk indices sorted by BM25 score desc."""
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:k]

    def _rrf_merge(self, vec_hits: list[tuple[int, float]],
                   bm25_ranked: list[int]) -> list[dict]:
        """Reciprocal Rank Fusion → chronological sort → return chunk dicts."""
        vec_rank = {idx: rank for rank, (idx, _) in enumerate(vec_hits)}
        bm25_rank = {idx: rank for rank, idx in enumerate(bm25_ranked)}

        all_idx = set(vec_rank) | set(bm25_rank)
        rrf_scores: dict[int, float] = {}
        for idx in all_idx:
            score = 0.0
            if idx in vec_rank:
                score += 1.0 / (RRF_K + vec_rank[idx])
            if idx in bm25_rank:
                score += 1.0 / (RRF_K + bm25_rank[idx])
            rrf_scores[idx] = score

        # sort by RRF score
        sorted_idx = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

        results = []
        for idx in sorted_idx:
            entry = dict(self.chunks[idx])
            entry["_idx"] = idx
            entry["score"] = rrf_scores[idx]
            results.append(entry)

        # final: chronological order so Gemini sees timeline
        results.sort(key=lambda h: (h["day"], h["start_time"]))
        return results

    def _extract_names(self, query: str) -> set[str]:
        """Resolve query words to actual channel member names.
        Case-insensitive + fuzzy (handles 'divaesh nandha' → 'Divaesh nandaa').
        Does NOT rely on user capitalizing the name."""
        import difflib
        qtokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) >= 3]
        all_tokens = list(self._name_tokens.keys())
        names: set[str] = set()
        for qt in qtokens:
            if qt in self._name_tokens:
                names |= self._name_tokens[qt]
            else:
                # fuzzy match for misspellings (high cutoff → avoid false hits)
                for m in difflib.get_close_matches(qt, all_tokens, n=2, cutoff=0.82):
                    names |= self._name_tokens[m]
        return names

    def _person_view(self, chunk: dict, names: set) -> str:
        """Build a denoised view of a chunk for a person query:
        keep thread parent (context) + only messages authored by or @mentioning
        the person. Drops other people's standup-thread lines. Case-insensitive."""
        msgs = chunk.get("messages")
        if not msgs:
            return chunk["text"]
        lowered = [n.lower() for n in names]
        day = chunk["day"]
        kept = []
        for j, m in enumerate(msgs):
            u, t = m["user"].lower(), m["text"].lower()
            authored = any(n in u for n in lowered)
            mentioned = any(n in t for n in lowered)
            if authored or mentioned or j == 0:
                kept.append(m)
        relevant = [m for m in kept
                    if any(n in m["user"].lower() for n in lowered)
                    or any(n in m["text"].lower() for n in lowered)]
        if not relevant:
            return ""
        lines = []
        for m in kept:
            prefix = "  ↳ " if m["is_reply"] else ""
            lines.append(f"{prefix}**{m['user']}** [{day} {m['time']}]: {m['text']}")
        return "\n".join(lines)

    def _name_supplement(self, query: str, existing: list[dict]) -> list[dict]:
        """Guarantee every chunk where a queried person appears is included,
        but DENOISED to that person's own lines (+ thread parent for context)."""
        candidates = self._extract_names(query)
        if not candidates:
            return existing
        lowered = [n.lower() for n in candidates]
        by_idx = {h["_idx"]: h for h in existing}
        for i, chunk in enumerate(self.chunks):
            hay = (" ".join(chunk.get("users", [])) + " " + chunk.get("text", "")).lower()
            if not any(n in hay for n in lowered):
                continue
            view = self._person_view(chunk, candidates)
            if not view:
                continue
            if i in by_idx:
                # replace full chunk text with denoised view to cut noise
                by_idx[i] = {**by_idx[i], "text": view}
            else:
                by_idx[i] = {**chunk, "_idx": i, "score": 0.0, "text": view}
        results = list(by_idx.values())
        results.sort(key=lambda h: (h["day"], h["start_time"]))
        return results

    def _retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        vec_hits = self._vector_search(query, k)
        bm25_ranked = self._bm25_search(query, k)
        results = self._rrf_merge(vec_hits, bm25_ranked)
        return self._name_supplement(query, results)

    def _format_hits(self, hits: list[dict]) -> str:
        """Group chunks by day with clear date headers → prevents Gemini date confusion."""
        from collections import defaultdict
        by_day: dict[str, list[str]] = defaultdict(list)
        for h in hits:
            by_day[h["day"]].append(h["text"])
        sections = []
        for day in sorted(by_day):
            body = "\n\n".join(by_day[day])
            sections.append(f"=== {day} ===\n{body}")
        return "\n\n".join(sections)

    def _log(self, question: str, hits: list[dict],
             answer: str, usage: dict) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        block = [
            "=" * 70,
            f"[{ts}] query: {question}",
            f"  key #{usage.get('key_index')} | "
            f"input={usage.get('prompt_tokens')} "
            f"output={usage.get('completion_tokens')} "
            f"total={usage.get('total_tokens')} tokens",
            f"  top-{len(hits)} chunks retrieved",
            "  answer:",
            answer,
            "",
            "",
        ]
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write("\n".join(block))

    def ask(self, question: str, thread_text: str | None = None) -> str:
        retrieval_query = question
        if thread_text:
            retrieval_query = f"{thread_text[-800:]}\n\nLatest: {question}"

        hits = self._retrieve(retrieval_query)
        context = self._format_hits(hits)
        now = _now_ist()

        if thread_text:
            prompt = self.prompts["ANSWER_THREAD"].format(
                channel_context=context or "(no relevant messages found)",
                thread=thread_text,
                question=question or "(no new text)",
                current_datetime=now,
            )
        else:
            prompt = self.prompts["ANSWER"].format(
                context=context or "(no relevant messages found)",
                question=question,
                current_datetime=now,
            )

        answer, usage = complete_ex(prompt)
        self._log(question, hits, answer, usage)
        return answer


if __name__ == "__main__":
    bot = RAGBot()
    print("Hybrid RAG ready. Ask questions (Ctrl+C to quit).\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q:
            print("\n" + bot.ask(q) + "\n")
