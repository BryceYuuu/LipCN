import json
from hybrid_candidates import CandidateFormatter


def echo_model(messages):
    raw = json.loads(messages[-1]['content'])['原文']
    return json.dumps({'natural': raw, 'concise': raw}, ensure_ascii=False)


def test_clear_register_and_filler_edits_offer_three_same_meaning_forms():
    result = CandidateFormatter(generator=echo_model).format(['嗯，我想知道这是啥意思'], source='speech')
    assert [v.text for v in result.variants] == [
        '嗯，我想知道这是啥意思。', '嗯，我想知道这是什么意思。', '我想知道这是啥意思。']


def test_explicit_name_is_not_rewritten_as_a_colloquial_word():
    result = CandidateFormatter(generator=echo_model).format(['请把文件发给啥啥'], protected_names=['啥啥'])
    assert all('啥啥' in variant.text for variant in result.variants)


def test_already_fluent_sentence_is_not_forced_into_three_claims():
    result = CandidateFormatter(generator=echo_model).format(['明天下午三点我不去上海开会'], source='speech')
    assert len(set(v.text for v in result.variants)) == 1
    assert any('部分方案相同' in text for text in result.warnings)
