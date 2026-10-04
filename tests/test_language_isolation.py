"""Language identity comes from the chosen runtime, including mixed text."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from lipflow import corrections, dictation, paths, personal, practice, vocab
from lipflow.cleanup import Cleaner


@pytest.fixture
def homes(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "HOME", str(tmp_path))
    monkeypatch.setattr(vocab, "PATH", str(tmp_path / "words.txt"))
    monkeypatch.setattr(personal, "PHRASES", str(tmp_path / "phrases.txt"))
    monkeypatch.setattr(dictation, "SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setattr(dictation, "HISTORY", str(tmp_path / "history.jsonl"))
    monkeypatch.setattr(corrections, "DIR", str(tmp_path / "clips/corrections"))
    monkeypatch.setattr(practice, "CLIPS", str(tmp_path / "clips/onboarding"))
    return tmp_path


def test_vocab_and_personal_phrases_stay_in_selected_language(homes):
    homes.joinpath("words.txt").write_text("EnglishName\n", encoding="utf-8")
    homes.joinpath("phrases.txt").write_text("English private phrase\n", encoding="utf-8")
    zh = homes / "languages/zh"
    zh.mkdir(parents=True)
    zh.joinpath("words.txt").write_text("中文 Zoom\n", encoding="utf-8")
    zh.joinpath("phrases.txt").write_text("中文 Zoom 会议\n", encoding="utf-8")
    en, cn = Cleaner("basic", language="en"), Cleaner("basic", language="zh")
    assert vocab.load("en") == ["EnglishName"]
    assert vocab.load("zh") == ["中文 Zoom"]
    assert en.personal.phrases == ["English private phrase"]
    assert cn.personal.phrases == ["中文 Zoom 会议"]
    assert en.guard_sensitive is False and cn.guard_sensitive is True


def test_settings_and_language_switch_do_not_copy_preferences(homes):
    dictation.save_settings({"language": "en", "cleanup": "codex", "input_mode": "whisper"})
    dictation.save_settings({"language": "zh", "cleanup": "basic", "input_mode": "silent"})
    paths.select_language("zh")
    assert paths.default_language() == "zh"
    assert dictation.load_settings("en")["cleanup"] == "codex"
    assert dictation.load_settings("zh")["input_mode"] == "silent"
    paths.select_language("en")
    assert dictation.load_settings("zh")["cleanup"] == "basic"
    with pytest.raises(ValueError):
        paths.language_home("../../other")


def test_mixed_text_corrections_use_explicit_language_not_han_detection(homes):
    rois = np.zeros((3, 96, 96), dtype=np.uint8)
    corrections.save(rois, "Zoom meeting", "Zoom 中文 meeting", [], language="en")
    corrections.save(rois, "开会", "Zoom meeting", [], language="zh")
    assert [x["text"] for x in corrections.load_all("en")] == ["Zoom 中文 meeting"]
    assert [x["text"] for x in corrections.load_all("zh")] == ["Zoom meeting"]
    assert corrections.count("en") == corrections.count("zh") == 1


def test_history_and_clip_retention_are_separate(homes, monkeypatch):
    monkeypatch.setattr(dictation, "KEEP_CLIPS", 1)
    rois = np.zeros((3, 96, 96), dtype=np.uint8)
    rec = SimpleNamespace(duration=1)
    for language, text in [("en", "English 中文"), ("zh", "中文 English")]:
        dictation.log_history(rec, [text], text, 1, "basic", language=language)
        dictation.keep_clip(rois, [text], text, {"language": language})
        practice.save_clip(rois, text, language=language)
    en = json.loads(homes.joinpath("history.jsonl").read_text())
    zh = json.loads(homes.joinpath("languages/zh/history.jsonl").read_text())
    assert en["text"] == "English 中文" and zh["text"] == "中文 English"
    assert len(practice.saved_clips("en")) == len(practice.saved_clips("zh")) == 1
    assert len(list(homes.glob("clips/dictations/*.npz"))) == 1
    assert len(list(homes.glob("languages/zh/clips/dictations/*.npz"))) == 1
