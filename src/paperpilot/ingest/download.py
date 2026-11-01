"""Fetch metadata and PDFs from arXiv for every paper in data/papers.yaml.

Safe to re-run: PDFs already on disk are skipped.
"""

import logging
import time
from pathlib import Path

import arxiv
import httpx
import yaml

from paperpilot.config import get_settings
from paperpilot.models import Paper

# arXiv asks scripts to identify themselves
log = logging.getLogger(__name__)
USER_AGENT = "paperpilot/0.1 (https://github.com/hidouri-iheb/rag-paperpilot)"


def load_paper_list(path: Path) -> dict[str, str]:
    """Return {arxiv_id: area}, in file order."""
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))["papers"]
    return {str(entry["id"]): entry["area"] for entry in entries}


def fetch_metadata(id_to_area: dict[str, str], delay: float) -> list[Paper]:
    """One batched query for all ids. Logs any id arXiv doesn't know."""
    client = arxiv.Client(page_size=100, delay_seconds=delay, num_retries=3)
    search = arxiv.Search(id_list=list(id_to_area), max_results=len(id_to_area))

    found: dict[str, Paper] = {}
    for result in client.results(search):
        arxiv_id = result.get_short_id().split("v")[0]  # 1706.03762v7 -> 1706.03762
        found[arxiv_id] = Paper(
            arxiv_id=arxiv_id,
            title=" ".join(result.title.split()),
            authors=[author.name for author in result.authors],
            year=result.published.year,
            abstract=" ".join(result.summary.split()),
            area=id_to_area.get(arxiv_id, "unknown"),
        )

    for arxiv_id in id_to_area:
        if arxiv_id in found:
            paper = found[arxiv_id]
            log.info("%s | %d | %s", arxiv_id, paper.year, paper.title)
        else:
            log.error("%s | no metadata, check the id", arxiv_id)
    return [found[i] for i in id_to_area if i in found]


def download_pdf(arxiv_id: str, dest: Path) -> None:
    """Write to a .part file first so a crash never leaves a half PDF behind."""
    response = httpx.get(
        f"https://arxiv.org/pdf/{arxiv_id}",
        headers={"User-Agent": USER_AGENT},
        timeout=60,
        follow_redirects=True,  # arxiv redirects to the latest version
    )
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError("response is not a PDF")
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(response.content)
    tmp.replace(dest)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    settings.pdf_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    id_to_area = load_paper_list(settings.papers_yaml)
    papers = fetch_metadata(id_to_area, settings.download_delay_seconds)

    downloaded = skipped = failed = 0
    for paper in papers:
        dest = settings.pdf_dir / f"{paper.arxiv_id}.pdf"
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue
        try:
            download_pdf(paper.arxiv_id, dest)
            downloaded += 1
            log.info("downloaded %s", paper.arxiv_id)
        except Exception as exc:  # keep going, report at the end
            failed += 1
            log.error("failed %s: %s", paper.arxiv_id, exc)
        time.sleep(settings.download_delay_seconds)

    out = settings.processed_dir / "papers.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for paper in papers:
            f.write(paper.model_dump_json() + "\n")

    missing = len(id_to_area) - len(papers)
    log.info(
        "done: %d downloaded, %d skipped, %d failed, %d missing metadata",
        downloaded,
        skipped,
        failed,
        missing,
    )
    if failed or missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
