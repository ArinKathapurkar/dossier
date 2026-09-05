"""Chunking: page boundaries are never crossed and ids are stable."""

from dossier.ingest.chunk import Chunk, WhitespaceTokenizer, chunk_page, chunk_pages, make_chunk_id


def test_short_page_yields_one_chunk():
    tok = WhitespaceTokenizer()
    assert chunk_page("revenue grew twelve percent", tok, window=350, overlap=50) == [
        "revenue grew twelve percent"
    ]


def test_empty_page_yields_nothing():
    assert chunk_page("   \n\n  ", WhitespaceTokenizer()) == []


def test_long_page_windows_overlap():
    tok = WhitespaceTokenizer()
    text = " ".join(f"w{i}" for i in range(120))
    pieces = chunk_page(text, tok, window=50, overlap=10)
    assert len(pieces) > 1
    first, second = pieces[0].split(), pieces[1].split()
    assert len(first) == 50
    # step = window - overlap = 40, so the second window starts at token 40 and the last
    # 10 tokens of the first window reappear at the head of the second.
    assert first[-10:] == second[:10]


def test_overlap_must_be_smaller_than_window():
    import pytest

    with pytest.raises(ValueError):
        chunk_page("a b c", WhitespaceTokenizer(), window=10, overlap=10)


def test_chunks_never_cross_a_page_boundary():
    tok = WhitespaceTokenizer()
    pages = [("DOC", 1, " ".join(f"a{i}" for i in range(200))), ("DOC", 2, " ".join(f"b{i}" for i in range(200)))]
    meta = {"DOC": {"company": "3M", "doc_type": "10K", "doc_period": 2022}}
    chunks = chunk_pages(pages, meta, tok, window=50, overlap=10)
    for c in chunks:
        tokens = c.text.split()
        prefixes = {t[0] for t in tokens}
        assert len(prefixes) == 1, "a chunk mixed tokens from two pages"
        assert (c.page_num == 1) == (prefixes == {"a"})


def test_chunk_ids_are_deterministic():
    assert make_chunk_id("DOC", 3, 0) == make_chunk_id("DOC", 3, 0)
    assert make_chunk_id("DOC", 3, 0) != make_chunk_id("DOC", 3, 1)
    assert make_chunk_id("DOC", 3, 0) != make_chunk_id("OTHER", 3, 0)


def test_chunk_carries_document_metadata():
    tok = WhitespaceTokenizer()
    meta = {"DOC": {"company": "Costco", "doc_type": "10K", "doc_period": 2021}}
    chunks = chunk_pages([("DOC", 5, "membership fee income rose")], meta, tok)
    assert isinstance(chunks[0], Chunk)
    assert (chunks[0].company, chunks[0].doc_type, chunks[0].fiscal_period, chunks[0].page_num) == (
        "Costco",
        "10K",
        "2021",
        5,
    )
