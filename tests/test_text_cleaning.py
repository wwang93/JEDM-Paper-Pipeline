from jedm_pipeline.text_cleaning import chunk_paragraphs, srt_to_text


def test_srt_to_text_basic() -> None:
    srt = "1\n00:00:00,000 --> 00:00:01,000\nHello\n\n2\n00:00:01,100 --> 00:00:02,000\nWorld\n"
    assert srt_to_text(srt) == "Hello World"


def test_chunk_paragraphs_splits_long_text() -> None:
    text = "a " * 3000
    chunks = chunk_paragraphs(text, max_chars=1000)
    assert len(chunks) > 1
