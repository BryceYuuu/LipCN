from lipflow.corrections import find_correction, inserted_span

P = "Hello Miguel, I am sending you a message with my new school."


def fix(before, after, pasted=P):
    return find_correction(pasted, inserted_span(before, after))


def test_small_fix_is_learned():
    b = "Chat so far. "
    assert fix(b, b + P, P) is None  # untouched
    assert fix(b, b + P.replace("school", "tool"), P) == "Hello Miguel, I am sending you a message with my new tool."


def test_typing_more_after_is_not_part_of_the_label():
    b = ""
    after = P.replace("school", "tool") + " Also can you review it by Friday?"
    assert fix(b, after) == "Hello Miguel, I am sending you a message with my new tool."


def test_text_after_the_cursor_is_ignored():
    b = "Before. After text"
    after = "Before. " + P.replace("school", "tool") + " After text"
    assert fix(b, after) == "Hello Miguel, I am sending you a message with my new tool."


def test_rewrite_or_deletion_is_not_a_correction():
    assert fix("", "Totally different words that I wrote instead of it") is None
    assert fix("", "") is None
