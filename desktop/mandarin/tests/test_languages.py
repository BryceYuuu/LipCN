"""Explicit language routes and real offline OpenCC conversion; no model/device."""
import sys
from types import SimpleNamespace

import pytest

import languages


def test_choices_default_and_recognition_routes_are_explicit():
    assert languages.LANGUAGE_CHOICES == (
        ('zh-Hans', '简体中文'), ('zh-Hant', '繁體中文'), ('en', 'English'))
    assert languages.DEFAULT_LANGUAGE == 'zh-Hans'
    assert languages.recognition_language('zh-Hans') == 'zh'
    assert languages.recognition_language('zh-Hant') == 'zh'
    assert languages.recognition_language('en') == 'en'


@pytest.mark.parametrize('language', ['zh', 'fr', '', 'EN', 'zh-TW', None, 0, ['en']])
def test_unknown_ids_never_silently_select_a_recognizer(language):
    with pytest.raises(ValueError, match='不支持的语言'):
        languages.recognition_language(language)


def test_real_opencc_preserves_facts_and_uses_script_not_regional_terms():
    original = '我明天3点不能去开会，软件和硬盘，Alex API v2.1，费用100元。'
    traditional = languages.to_traditional(original)
    assert traditional == '我明天3點不能去開會，軟件和硬盤，Alex API v2.1，費用100元。'
    assert languages.to_simplified(traditional) == original
    assert '軟體' not in traditional and '硬碟' not in traditional


def test_real_opencc_resolves_common_phrase_context():
    assert languages.to_traditional('头发发展，皇后后来。') == '頭髮發展，皇后後來。'


def test_missing_converter_fails_explicitly_instead_of_returning_wrong_script(monkeypatch):
    languages._converter.cache_clear()
    monkeypatch.setitem(sys.modules, 'opencc', None)
    try:
        with pytest.raises(languages.LanguageConversionError, match='opencc-python-reimplemented'):
            languages.to_traditional('不可以支付100元')
    finally:
        languages._converter.cache_clear()


def test_converter_failures_never_expose_input_or_backend_exception(monkeypatch):
    def fail(text):
        raise RuntimeError('PRIVATE TRANSCRIPT ' + text)
    monkeypatch.setattr(languages, '_converter', lambda mode: SimpleNamespace(convert=fail))
    with pytest.raises(languages.LanguageConversionError) as error:
        languages.to_simplified('PRIVATE NAME')
    assert 'PRIVATE' not in str(error.value)
    assert error.value.__suppress_context__


@pytest.mark.parametrize('result', [None, '', 12])
def test_invalid_conversion_is_not_accepted(monkeypatch, result):
    monkeypatch.setattr(languages, '_converter', lambda mode: SimpleNamespace(convert=lambda text: result))
    with pytest.raises(languages.LanguageConversionError):
        languages.to_traditional('我们不要取消')


def test_converter_is_loaded_once_per_direction(monkeypatch):
    calls = []
    def construct(mode):
        calls.append(mode)
        return SimpleNamespace(convert=lambda text: text)
    languages._converter.cache_clear()
    monkeypatch.setitem(sys.modules, 'opencc', SimpleNamespace(OpenCC=construct))
    try:
        languages.to_simplified('你好')
        languages.to_simplified('明天见')
        languages.to_traditional('你好')
        languages.to_traditional('明天见')
        assert calls == ['t2s', 's2t']
    finally:
        languages._converter.cache_clear()
