"""Distinct decoder hypotheses stay separate; no rewriting model is involved."""
from unittest.mock import Mock

import pytest

import hybrid_candidates as hc


def texts(result):
    return [variant.text for variant in result.variants]


def test_actual_hypotheses_preserve_model_order_and_do_not_rewrite_the_first():
    candidates = ['嗯，我明天不去上海', '我明天要去伤害', '明天我不去商海', '我昨天去了上海']
    result = hc.CandidateFormatter().recognition_choices(candidates)
    assert texts(result) == [text + '。' for text in candidates[:3]]
    assert [variant.label for variant in result.variants] == ['方案一', '方案二', '方案三']
    assert result.raw == candidates[0] and result.backend == 'recognition-candidates'
    assert all(not variant.changed for variant in result.variants)
    assert result.elapsed_seconds >= 0


@pytest.mark.parametrize('language,candidates,expected', [
    ('zh-Hans', ['明天見', '明天见。', ' 明 天 见！', '明年见', '后天见'],
     ['明天见。', '明年见。', '后天见。']),
    ('zh-Hant', ['软件会发展', '軟件會發展。', '軟 件 會 發 展', '软件不会发展'],
     ['軟件會發展。', '軟件不會發展。']),
    ('en', ["I CAN'T PAY ALICE", 'i can’t pay alice.', "I can't pay Alice!", 'I can pay Alice',
            'I cannot pay Alex'],
     ["I CAN'T PAY ALICE", 'I can pay Alice', 'I cannot pay Alex']),
])
def test_deduplicate_in_target_script_ignoring_punctuation_spaces_and_case(language, candidates, expected):
    result = hc.CandidateFormatter().recognition_choices(candidates, language=language)
    assert texts(result) == expected


def test_traditional_conversion_does_not_replace_regional_terms():
    result = hc.CandidateFormatter().recognition_choices(['这个软件很好', '这个软件不好'], language='zh-Hant')
    assert texts(result) == ['這個軟件很好。', '這個軟件不好。']


@pytest.mark.parametrize('candidates,expected', [
    ([], []), (['', ' ', None, 42, '。！？'], []),
    (['我不去'], ['我不去。']),
    (['我不去', '我不去。', '我 不 去'], ['我不去。']),
    (['我不去', '我去', '我去。'], ['我不去。', '我去。']),
])
def test_never_pads_missing_choices_with_repetition_or_invented_text(candidates, expected):
    result = hc.CandidateFormatter().recognition_choices(candidates)
    assert texts(result) == expected
    if not expected:
        assert result.raw == '' and result.warnings == () and not result.low_consistency


def test_candidates_after_duplicates_are_considered_until_three_distinct_are_found():
    result = hc.CandidateFormatter().recognition_choices(
        ['第一项', '第一项。', '第 一 项', '第二项', '第二项！', '第三项', '第四项'])
    assert texts(result) == ['第一项。', '第二项。', '第三项。']


@pytest.mark.parametrize('language,candidates', [
    ('zh-Hans', ['张桓明天3点不会支付100元', '张恒明天4点会支付200元', '张桓明天3点会支付100元']),
    ('en', ["Alice won’t pay $1.50 at 3:15 PM", "Alex won't pay $2.50 at 3:15 PM", 'Alice will pay $1.50']),
    ('en', ['Meet 张桓 at 3 PM', 'Meet 张恒 at 4 PM', 'Do not meet 张桓']),
])
def test_names_numbers_negation_and_mixed_language_are_not_altered(language, candidates):
    result = hc.CandidateFormatter().recognition_choices(candidates, language=language)
    assert texts(result) == [hc.punctuate(text) if language != 'en' else text for text in candidates]


def test_english_first_apostrophe_and_capitalization_are_preserved_in_display():
    result = hc.CandidateFormatter().recognition_choices(
        ['Alice’s API won’t change', "ALICE'S API WON'T CHANGE", 'Alice’s API will change'], language='en')
    assert texts(result) == ['Alice’s API won’t change', 'Alice’s API will change']


def test_model_ranks_are_not_accepted_as_probabilities_or_extra_text():
    result = hc.CandidateFormatter().recognition_choices(
        [{'text': '伪候选', 'score': 0.99}, ('伪候选', 0.95), '实际候选'])
    assert texts(result) == ['实际候选。']
    assert all('score' not in variant and 'probability' not in variant for variant in result.as_dict()['variants'])


@pytest.mark.parametrize('language', ['zh-Hans', 'zh-Hant', 'en'])
def test_no_model_access_locks_rewrites_or_external_generation(monkeypatch, language):
    def forbidden(*args, **kwargs):
        pytest.fail('recognition choices must only present actual hypotheses')
    class ForbiddenLock:
        def __enter__(self):
            forbidden()
        def __exit__(self, *args):
            pass
    formatter = hc.CandidateFormatter(generator=forbidden)
    formatter._lock = ForbiddenLock()
    formatter.model_path = Mock()
    formatter.model_path.is_dir.side_effect = forbidden
    for name in ('load', '_generate', 'format', 'format_fast', 'initial'):
        monkeypatch.setattr(formatter, name, forbidden)
    monkeypatch.setattr(hc, 'style_fallback', forbidden)
    result = formatter.recognition_choices(['我不去', '我去'] if language != 'en' else ['I will not go', 'I will go'],
                                           language=language)
    assert len(result.variants) == 2 and formatter._model is None
    formatter.model_path.is_dir.assert_not_called()


@pytest.mark.parametrize('language', ['zh-Hans', 'zh-Hant'])
def test_missing_conversion_is_explicit_instead_of_wrong_script(monkeypatch, language):
    def unavailable(text):
        raise hc.LanguageConversionError('简繁转换组件不可用')
    monkeypatch.setattr(hc, 'to_simplified' if language == 'zh-Hans' else 'to_traditional', unavailable)
    with pytest.raises(hc.LanguageConversionError):
        hc.CandidateFormatter().recognition_choices(['不能付款'], language=language)


@pytest.mark.parametrize('language', ['zh', '', 'automatic', None])
def test_new_api_requires_an_explicit_supported_language(language):
    with pytest.raises(ValueError, match='不支持的语言'):
        hc.CandidateFormatter().recognition_choices(['hello'], language=language)


@pytest.mark.parametrize('language,candidates', [
    ('zh-Hans', ['支付3.5元', '支付35元']),
    ('zh-Hant', ['支付3.5元', '支付35元']),
    ('en', ['PAY $1.50', 'PAY $150']),
    ('en', ['PAY -5', 'PAY 5', 'PAY5']),
    ('en', ["WE'RE READY", 'WERE READY']),
    ('en', ['CALL JO ANN', 'CALL JOANN']),
    ('zh-Hans', ['请PAY -5元', '请PAY5元', '请PAY +5元']),
    ('zh-Hant', ['請給JO ANN支付$1.50', '請給JOANN支付$1.50', '請給JO ANN支付$150']),
    ('en', ["WE'RE支付3.5元", 'WERE支付3.5元', "WE'RE支付35元"]),
    ('en', ['meet at 3:15', 'meet at 315', 'meet on 3/15']),
    ('en', ['use ID-5', 'use ID5', 'use ID_5']),
    ('en', ['CALL MASSE', 'CALL MAẞE']),
])
def test_meaningful_numeric_notation_word_boundaries_and_contractions_remain_distinct(language, candidates):
    result = hc.CandidateFormatter().recognition_choices(candidates, language=language)
    if language == 'zh-Hans':
        expected = [hc.punctuate(hc.to_simplified(text)) for text in candidates]
    elif language == 'zh-Hant':
        expected = [hc.punctuate(hc.to_traditional(text)) for text in candidates]
    else:
        expected = candidates
    assert texts(result) == expected


@pytest.mark.parametrize('language,candidates,expected', [
    ('en', ['Call  Jo\tAnn!', 'call jo ann.', 'CALL JOANN'], ['Call  Jo\tAnn!', 'CALL JOANN']),
    ('en', ["WE'RE READY", 'we’re ready.', 'weʼre ready!'], ["WE'RE READY"]),
    ('zh-Hans', ['请 给 JO ANN 支付 $1.50', '请给 jo  ann支付$1.50。'], ['请 给 JO ANN 支付 $1.50。']),
    ('zh-Hant', ['軟 件 要 3.5 元', '软件要3.5元。'], ['軟 件 要 3.5 元。']),
])
def test_cosmetic_duplicates_still_collapse_without_rewriting_display(language, candidates, expected):
    result = hc.CandidateFormatter().recognition_choices(candidates, language=language)
    assert texts(result) == expected
