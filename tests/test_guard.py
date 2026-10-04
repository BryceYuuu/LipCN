import pytest
from lipflow.guard import check
from lipflow.cleanup import Cleaner
from lipflow.personal import Personal


@pytest.mark.parametrize('original,proposed', [
    ("DON'T SEND IT TOMORROW", 'Send it tomorrow.'),
    ('Pay $100', 'Pay 100'), ('Pay 100 yuan', 'Pay 1000 yuan'),
    ('Meet tomorrow at 4:30', 'Meet today at 4:30'),
    ('不要明天发送', '明天发送'), ('明天给张三转账五百元', '明天给张三转账五千元'),
    ('会议在十月二日', '会议在十月三日'), ('我没有同意这个方案', '我同意这个方案'),
])
def test_sensitive_changes_require_review(original, proposed):
    assert check(original, proposed, [proposed])


def test_name_change_requires_review_even_if_it_is_a_decoder_alternative():
    assert check('Hi Miguel', 'Hi Priya', ['Hi Priya'], ['Miguel', 'Priya'])
    assert check('发给张三', '发给李四', ['发给李四'], ['张三', '李四'])


def test_formatting_and_equivalent_english_numbers_are_allowed():
    assert not check('HELLO WORLD', 'Hello, world.')
    assert not check('你好世界', '你好，世界。')
    assert not check('IN NINETEEN FORTY THREE', 'In 1943.')


def test_polish_always_reviews_changed_wording():
    assert check('I like the plan', 'The plan looks good', mode='polish')


@pytest.mark.parametrize('backend', ['local', 'claude', 'ollama'])
def test_every_backend_applies_same_guard_without_network(backend, monkeypatch):
    cleaner = Cleaner('basic', language='zh')
    cleaner.personal = Personal('/nonexistent')
    cleaner.backend = backend
    monkeypatch.setattr(cleaner, '_'+backend, lambda *args: '明天发送。')
    result = cleaner.process(['不要明天发送'], words=[])
    assert result.needs_review
    assert result.raw == '不要明天发送' and result.text == '不要明天发送。'
    assert result.proposed == '明天发送。'


def test_empty_gibberish_and_large_edits_fall_back_to_raw(monkeypatch):
    c = Cleaner('basic', guard_sensitive=True)
    c.backend = 'ollama'
    monkeypatch.setattr(c, '_ollama', lambda *args: 'Here is a completely unrelated answer')
    r = c.process(['THE PLAN IS READY'], words=[])
    assert r.needs_review and r.text == 'The plan is ready.'


@pytest.mark.parametrize('backend', ['local', 'claude', 'ollama'])
@pytest.mark.parametrize('raw,proposed', [
    ('WE WALK IN THE BARK', 'We walk in the park.'),
    ('SEND FIVE COPIES', 'Send 5 copies.'),
])
def test_default_english_does_not_revert_backend_corrections(backend, raw, proposed, monkeypatch):
    cleaner = Cleaner('basic')
    cleaner.personal = Personal('/nonexistent')
    cleaner.backend = backend
    monkeypatch.setattr(cleaner, '_' + backend, lambda *args: proposed)
    result = cleaner.process([raw], words=[])
    assert result.raw == raw
    assert result.text == proposed == result.proposed
    assert not result.needs_review


@pytest.mark.parametrize('backend', ['local', 'claude', 'ollama'])
def test_english_explicit_opt_in_applies_guard_to_every_backend(backend, monkeypatch):
    cleaner = Cleaner('basic', guard_sensitive=True)
    cleaner.personal = Personal('/nonexistent')
    cleaner.backend = backend
    monkeypatch.setattr(cleaner, '_' + backend, lambda *args: 'Send it tomorrow.')
    result = cleaner.process(["DON'T SEND IT TOMORROW"], words=[])
    assert result.needs_review
    assert result.text == "Don't send it tomorrow."
    assert result.proposed == 'Send it tomorrow.'


def test_english_opt_in_retains_name_proposal_for_review():
    cleaner = Cleaner('basic', guard_sensitive=True)
    cleaner.personal = Personal('/nonexistent')
    result = cleaner.process(['HELLO MCCALL I AM SENDING YOU A MESSAGE'], words=[], names=['Miguel'])
    assert result.needs_review
    assert result.raw == 'HELLO MCCALL I AM SENDING YOU A MESSAGE'
    assert result.text == 'Hello mccall I am sending you a message.'
    assert result.proposed == 'Hello Miguel I am sending you a message.'


@pytest.mark.parametrize('backend', ['local', 'claude', 'ollama'])
@pytest.mark.parametrize('guard_sensitive', [None, False])
def test_english_polish_retains_raw_and_requires_review_across_backends(backend, guard_sensitive, monkeypatch):
    cleaner = Cleaner('basic', mode='polish', guard_sensitive=guard_sensitive)
    cleaner.personal = Personal('/nonexistent')
    cleaner.backend = backend
    monkeypatch.setattr(cleaner, '_' + backend, lambda *args: 'The plan looks good.')
    result = cleaner.process(['I LIKE THE PLAN'], words=[])
    assert cleaner.guard_sensitive
    assert result.needs_review
    assert result.raw == 'I LIKE THE PLAN'
    assert result.text == 'I like the plan.'
    assert result.proposed == 'The plan looks good.'
    assert 'Polished wording requires review / 润色结果需确认' in result.warnings


@pytest.mark.parametrize('backend', ['local', 'claude', 'ollama'])
def test_explicit_false_cannot_disable_mandarin_sensitive_guards(backend, monkeypatch):
    cleaner = Cleaner('basic', language='zh', guard_sensitive=False)
    cleaner.personal = Personal('/nonexistent')
    cleaner.backend = backend
    monkeypatch.setattr(cleaner, '_' + backend, lambda *args: '明天发送。')
    result = cleaner.process(['不要明天发送'], words=[])
    assert cleaner.guard_sensitive
    assert result.needs_review
    assert result.text == '不要明天发送。'
    assert result.proposed == '明天发送。'


def test_chinese_custom_terms_are_not_snapped_with_english_visemes():
    c = Cleaner('basic', language='zh')
    c.personal = Personal('/nonexistent')
    r = c.process(['请发给张三', '请发给李四'], words=['张三'])
    assert r.text == '请发给张三。'
