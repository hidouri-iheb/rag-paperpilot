from paperpilot.ingest.parse import clean_text, process_paper


def test_dehyphenation_and_ligatures():
    assert clean_text("retriev-\nal of ﬁles") == "retrieval of files"


def test_references_dropped_but_appendix_kept():
    raw = [
        (1, "# 1 Introduction\nWe study retrieval."),
        (2, "# References\n[1] Smith. A paper.\n[2] Doe. Another."),
        (3, "[3] Lee. More refs.\n# A Proof Details\nThe proof goes here."),
    ]
    pages = process_paper("x", raw)
    text = "\n".join(p.text for p in pages)
    assert "Smith" not in text and "Lee" not in text
    assert "The proof goes here." in text
    assert [p.page for p in pages] == [1, 3]


def test_section_carries_over_to_next_page():
    pages = process_paper("x", [(1, "# 2 Method\nText."), (2, "More text.")])
    assert pages[1].section == "2 Method"


def test_repeated_footer_removed():
    raw = [(i, f"Body {i}.\nPreprint under review") for i in range(1, 6)]
    assert all("Preprint" not in p.text for p in process_paper("x", raw))
