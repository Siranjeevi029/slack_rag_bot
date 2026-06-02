"""Reasoning-based retrieval over the PageIndex tree, answered by Groq.

Flow:
  1. Load tree (node_id, date, summary) + node_id -> line_num map.
  2. Groq picks relevant node_ids from the tree (reasoning over summaries).
  3. Fetch those nodes' raw text via get_page_content (line numbers).
  4. Groq answers the question grounded in the fetched text.

Prompts live in prompt.txt. LLM calls + key rotation live in bot/llm.py.
"""
import json
import re
import sys
from datetime import datetime

from datetime import timedelta, timezone

from bot import config
from bot.llm import complete_ex

IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist() -> str:
    return datetime.now(tz=IST).strftime("%Y-%m-%d %H:%M IST (%A)")

sys.path.insert(0, str(config.ROOT / "PageIndex"))
from pageindex.utils import structure_to_list  # noqa: E402

OUTPUT_FILE = config.ROOT / "output.txt"   # per-query log of selected node ids


def load_prompts() -> dict:
    """Parse prompt.txt into {'SELECT': ..., 'ANSWER': ...}."""
    text = config.PROMPT_FILE.read_text(encoding="utf-8")
    parts = re.split(r"^=== (\w+) ===$", text, flags=re.M)
    prompts = {}
    for i in range(1, len(parts), 2):
        prompts[parts[i]] = parts[i + 1].strip()
    for required in ("SELECT", "ANSWER", "ANSWER_THREAD"):
        if required not in prompts:
            raise ValueError(f"prompt.txt missing '=== {required} ===' section")
    return prompts


class RAGBot:
    def __init__(self):
        from bot.index import get_client
        self.client = get_client()
        self.doc_id = config.DOC_ID_FILE.read_text(encoding="utf-8").strip()
        structure = json.loads(self.client.get_document_structure(self.doc_id))
        self.nodes = structure_to_list(structure)
        self.line_by_id = {n["node_id"]: n.get("line_num")
                           for n in self.nodes if n.get("node_id")}
        self.title_by_id = {n["node_id"]: n.get("title", "")
                            for n in self.nodes if n.get("node_id")}
        self.prompts = load_prompts()

    @staticmethod
    def _fmt_usage(usage: dict) -> str:
        return (f"key #{usage.get('key_index')} | "
                f"input={usage.get('prompt_tokens')} "
                f"output={usage.get('completion_tokens')} "
                f"total={usage.get('total_tokens')} tokens")

    def _log(self, question: str, kind: str,
             select_ids: list[str], select_usage: dict,
             answer_out: str, answer_usage: dict) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ids = ", ".join(select_ids) if select_ids else "(none)"
        block = [
            "=" * 70,
            f"[{ts}] query: {question}",
            f"mode: {kind}",
            "",
            "--- CALL 1: node selection ---",
            f"  {self._fmt_usage(select_usage)}",
            f"  nodes selected ({len(select_ids)}): {ids}",
            "",
            "--- CALL 2: answer ---",
            f"  {self._fmt_usage(answer_usage)}",
            f"  nodes passed as context ({len(select_ids)}): {ids}",
            "  output:",
            answer_out,
            "",
            "",
        ]
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write("\n".join(block))

    def _tree_digest(self) -> str:
        lines = []
        for n in self.nodes:
            nid = n.get("node_id")
            if not nid:
                continue
            summ = (n.get("summary") or n.get("prefix_summary") or "").strip()
            lines.append(f"[{nid}] {n.get('title', '')}: {summ}")
        return "\n".join(lines)

    def _select_nodes(self, question: str):
        # No artificial cap: pass ALL relevant nodes the query needs. The ceiling
        # is just the total number of nodes available. Accuracy > token cost.
        prompt = self.prompts["SELECT"].format(
            sections=self._tree_digest(), question=question,
            max_nodes=len(self.line_by_id),
            current_datetime=_now_ist())
        raw, usage = complete_ex(prompt)
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end == -1:
            return [], usage
        try:
            ids = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            return [], usage
        return [i for i in ids if i in self.line_by_id], usage

    def _fetch(self, node_ids: list[str]) -> str:
        line_nums = [str(self.line_by_id[i]) for i in node_ids if self.line_by_id.get(i)]
        if not line_nums:
            return ""
        content = json.loads(self.client.get_page_content(self.doc_id, ",".join(line_nums)))
        return "\n\n".join(c.get("content", "") for c in content)

    def ask(self, question: str, thread_text: str | None = None) -> str:
        # In a thread, fold recent thread text into the retrieval query so
        # follow-ups ("what about the second one?") still route correctly.
        retrieval_query = question
        if thread_text:
            retrieval_query = f"{thread_text[-800:]}\n\nLatest: {question}"

        node_ids, select_usage = self._select_nodes(retrieval_query)
        context = self._fetch(node_ids) if node_ids else ""

        now = _now_ist()
        if thread_text:
            kind = "ANSWER_THREAD"
            prompt = self.prompts["ANSWER_THREAD"].format(
                channel_context=context or "(no relevant channel history found)",
                thread=thread_text,
                question=question or "(no new text — respond to the thread)",
                current_datetime=now,
            )
        else:
            kind = "ANSWER"
            prompt = self.prompts["ANSWER"].format(
                context=context or "(no channel history retrieved — likely small talk)",
                question=question,
                current_datetime=now,
            )
        answer, answer_usage = complete_ex(prompt)
        self._log(question, kind, node_ids, select_usage, answer, answer_usage)
        return answer


if __name__ == "__main__":
    bot = RAGBot()
    print("RAG ready. Ask questions (Ctrl+C to quit).\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q:
            print("\n" + bot.ask(q) + "\n")
