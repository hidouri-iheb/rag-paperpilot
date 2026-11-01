from collections import defaultdict


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion over lists of chunk ids, best first."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    # ties broken by id so results are deterministic
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
