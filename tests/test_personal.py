import sqlite3

import pytest

from lipflow import personal, vocab


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(personal, "DIR", str(tmp_path))
    monkeypatch.setattr(personal, "PHRASES", str(tmp_path / "phrases.txt"))
    monkeypatch.setattr(vocab, "PATH", str(tmp_path / "words.txt"))
    return tmp_path


def fake_wispr(folder):
    """A Wispr-like DB: ids, JSON blobs, a raw ASR column and a formatted-text column."""
    folder.mkdir()
    con = sqlite3.connect(folder / "flow.sqlite")
    con.execute("create table History (id text, app text, asrText text, formattedText text, meta text)")
    rows = ["Hey Miguel, can you send me the deck before the review?",
            "I'm sending you a message with my new tool.",
            "Let's ship Lipflow to Miguel tomorrow.",
            "Can you ask Miguel about the Vizcom demo?",
            "Ping Miguel when the Vizcom build is green.",
            "The Vizcom standup moved to ten."] * 2
    for i, r in enumerate(rows):
        con.execute("insert into History values (?,?,?,?,?)",
                    (f"id-{i}", "com.tinyspeck.slackmacgap", r.lower().replace(",", ""), r, '{"x": 1}'))
    con.commit()
    con.close()


def test_import_finds_formatted_text_and_names(home):
    fake_wispr(home / "Wispr Flow")
    stats = personal.import_wispr(str(home / "Wispr Flow"))
    assert stats["source"].endswith("History.formattedText")
    assert stats["phrases"] == 6
    assert "Vizcom" in stats["new_names"] and "Miguel" in stats["new_names"]
    assert "Lipflow" not in stats["new_names"]  # already listed
    assert "Miguel" in vocab.load()


def test_personal_rerank_and_similar(home):
    fake_wispr(home / "Wispr Flow")
    personal.import_wispr(str(home / "Wispr Flow"))
    p = personal.Personal(personal.PHRASES)
    guesses = ["HELLO MIGUEL I AM SENDING YOU A BASIN WITH MY NEW SCHOOL",
               "HELLO MIGUEL I AM SENDING YOU A MESSAGE WITH MY NEW TOOL"]
    assert p.rerank(guesses)[0].endswith("MESSAGE WITH MY NEW TOOL")
    assert p.rerank(["WHAT IS FOR LUNCH", "WHAT IS FOR BRUNCH"]) == ["WHAT IS FOR LUNCH", "WHAT IS FOR BRUNCH"]
    assert "I'm sending you a message with my new tool." in p.similar(guesses[0])
