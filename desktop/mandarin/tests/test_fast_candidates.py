"""Fast formatting uses real language rules/OpenCC, never a model or device."""
from unittest.mock import Mock

import pytest

import hybrid_candidates as hc


@pytest.mark.parametrize('language', ['zh', 'zh-Hans', 'zh-Hant', 'en'])
@pytest.mark.parametrize('source', ['speech', 'lips'])
@pytest.mark.parametrize('candidates', [
    '嗯，明天3點我不能付款', ['我找張桓', '我已经付款了'],
    '你咋看這個軟件', 'UM, I DO NOT OWE ALICE $100', 'Meet 张三 at 3 PM',
    'OpenAI API v2.1', [], ['', '  ', None, 3], ['', '  明天見Alex  '],
])
def test_fast_has_three_choices_and_exactly_matches_initial_first(language, source, candidates):
    formatter = hc.CandidateFormatter()
    kwargs = dict(source=source, protected_names=['張桓'], language=language)
    initial = formatter.initial(candidates, **kwargs)
    fast = formatter.format_fast(candidates, **kwargs)
    assert fast.raw == initial.raw
    assert fast.variants[:1] == initial.variants
    assert fast.low_consistency == initial.low_consistency
    assert len(fast.variants) == (3 if initial.variants else 0)
    assert fast.backend == 'fast-rules' and fast.elapsed_seconds >= 0


@pytest.mark.parametrize('language', ['zh-Hans', 'zh-Hant', 'en'])
def test_fast_does_not_load_generate_inspect_model_path_or_take_model_lock(monkeypatch, language):
    def forbidden(*args, **kwargs):
        pytest.fail('fast formatting must never access a language model')
    class ForbiddenLock:
        def __enter__(self):
            forbidden()
        def __exit__(self, *args):
            pass
    formatter = hc.CandidateFormatter(generator=forbidden)
    formatter.model_path = Mock()
    formatter.model_path.is_dir.side_effect = forbidden
    formatter._lock = ForbiddenLock()
    monkeypatch.setattr(formatter, 'load', forbidden)
    monkeypatch.setattr(formatter, '_generate', forbidden)
    raw = 'Um, Alice cannot pay $100' if language == 'en' else '嗯，明天3点我不付款'
    result = formatter.format_fast(raw, language=language)
    assert len(result.variants) == 3 and formatter._model is None and not formatter._warmed
    formatter.model_path.is_dir.assert_not_called()


def test_fast_uses_real_opencc_without_regional_terminology_changes():
    formatter = hc.CandidateFormatter()
    simplified = formatter.format_fast('你咋看這個軟件，頭髮不能剪', language='zh-Hans')
    traditional = formatter.format_fast('你咋看这个软件，头发不能剪', language='zh-Hant')
    assert [v.text for v in simplified.variants] == [
        '你咋看这个软件，头发不能剪。', '你怎么看这个软件，头发不能剪。', '你咋看这个软件，头发不能剪。']
    assert [v.text for v in traditional.variants] == [
        '你咋看這個軟件，頭髮不能剪。', '你怎麼看這個軟件，頭髮不能剪。', '你咋看這個軟件，頭髮不能剪。']
    assert all('軟體' not in v.text for v in traditional.variants)


@pytest.mark.parametrize('raw,protected', [
    ('你咋看明天3点不用交的100元', ()),
    ('嗯，我不会向Alex付款1.50元', ('Alex',)),
    ('这个项目叫“咋办”，明天不能改', ('咋办',)),
    ('呃，明天我们不去', ('呃',)),
    ('我們不會向張桓付款100元', ('張桓',)),
])
def test_fast_chinese_edits_keep_existing_content_guards(raw, protected):
    formatter = hc.CandidateFormatter()
    result = formatter.format_fast(raw, protected_names=protected, language='zh-Hans')
    names = tuple(hc.to_simplified(name) for name in protected)
    assert all(not hc.guard_proposal(result.raw, variant.text, names) for variant in result.variants)
    if protected == ('呃',):
        assert all('呃' in variant.text for variant in result.variants)
    if '咋办' in protected:
        assert all('咋办' in variant.text and '怎么办' not in variant.text for variant in result.variants)


def test_fast_chinese_removes_only_isolated_fillers():
    result = hc.CandidateFormatter().format_fast('嗯，明天我不去，呃，改天见', source='speech')
    assert result.variants[0].text == '嗯，明天我不去，呃，改天见。'
    assert result.variants[2].text == '明天我不去，改天见。'
    assert result.variants[2].changed


@pytest.mark.parametrize('raw,names', [
    ("Um, Alice can't pay $100 at 3:15 PM", ('Alice',)),
    ('Uh, I do not owe 1.50 USD', ()),
    ('The aluminum umbrella is not red', ()),
    ('Um, this name is Um', ('Um',)),
    ('Visit https://example.com/x and email alice@example.com', ()),
    ('Are you sure?', ()), ('um', ()),
])
def test_fast_english_keeps_negation_numbers_names_tokens_and_question_intent(raw, names):
    result = hc.CandidateFormatter().format_fast(raw, language='en', protected_names=names)
    assert all(not hc.guard_english_proposal(raw, variant.text, names) for variant in result.variants)
    assert result.variants[0].text == raw
    if raw.startswith("Um, Alice"):
        assert result.variants[1].text == raw + '.'
        assert result.variants[2].text == "Alice can't pay $100 at 3:15 PM."
    if names == ('Um',):
        assert all(variant.text.count('Um') == 2 for variant in result.variants)


@pytest.mark.parametrize('language,first,other', [
    ('zh-Hans', '我明天不会付款', '我已经给张桓付款100元'),
    ('zh-Hant', '我明天不會付款', '我已经给张桓付款100元'),
    ('en', 'I will not pay Alice', 'I paid Bob 100 dollars'),
    ('en', '我不会付100元 to Alice', 'Pay Bob instead'),
    ('zh-Hans', 'Alice API v2.1', '我今天已经付款'),
])
def test_fast_never_mixes_another_beam_or_translates_mixed_content(language, first, other):
    formatter = hc.CandidateFormatter()
    single = formatter.format_fast(first, language=language)
    multiple = formatter.format_fast([first, other], language=language)
    assert multiple.variants == single.variants
    if language == 'en' and hc._HAN.search(first):
        assert all(v.text == first for v in multiple.variants)


@pytest.mark.parametrize('language,raw', [('zh-Hans', '嗯，你咋看' * 70), ('en', 'Um, no payment. ' * 30)])
def test_long_input_is_preserved_without_truncation_or_rules(language, raw):
    formatter = hc.CandidateFormatter()
    result = formatter.format_fast(raw, language=language)
    assert all(v.text == formatter.initial(raw, language=language).variants[0].text for v in result.variants)


@pytest.mark.parametrize('language', ['zh-Hans', 'zh-Hant'])
def test_fast_script_conversion_failure_is_explicit_not_wrong_script(monkeypatch, language):
    def fail(text):
        raise hc.LanguageConversionError('简繁转换组件不可用')
    monkeypatch.setattr(hc, 'to_simplified' if language == 'zh-Hans' else 'to_traditional', fail)
    with pytest.raises(hc.LanguageConversionError, match='简繁转换组件不可用'):
        hc.CandidateFormatter().format_fast('明天我不付款', language=language)


def test_fast_rejects_unknown_language_without_model_work():
    with pytest.raises(ValueError, match='不支持的语言'):
        hc.CandidateFormatter().format_fast('hello', language='automatic')
