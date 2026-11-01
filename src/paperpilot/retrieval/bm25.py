import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from paperpilot.models import Chunk

# keeps "self-attention", "gpt-3", "t5" as single tokens
TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or "
    "that the this to was we were what which with our does do".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.bm25 = BM25Okapi([tokenize(c.indexed_text) for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = scores.argsort()[::-1][:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top]

    def save(self, path: Path) -> None:
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "BM25Index":
        return pickle.loads(path.read_bytes())  # our own file, built locally
