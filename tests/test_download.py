from paperpilot.config import PROJECT_ROOT
from paperpilot.ingest.download import load_paper_list

AREAS = {"transformers", "rag-retrieval", "fine-tuning", "vision", "rl"}


def test_trailing_zero_in_id_survives(tmp_path):
    yaml_file = tmp_path / "papers.yaml"
    yaml_file.write_text('papers:\n  - { id: "1801.01290", area: rl }\n', encoding="utf-8")
    assert list(load_paper_list(yaml_file)) == ["1801.01290"]


def test_curated_list_is_valid():
    papers = load_paper_list(PROJECT_ROOT / "data" / "papers.yaml")
    assert len(papers) == 50  # also catches duplicates, the dict would merge them
    assert set(papers.values()) <= AREAS
