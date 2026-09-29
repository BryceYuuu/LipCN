from lipflow.cleanup import basic_cleanup, numbers_to_digits, _user_prompt


def test_years_and_numbers():
    assert numbers_to_digits("married in nineteen fifty two") == "married in 1952"
    assert numbers_to_digits("it is twenty twenty six") == "it is 2026"
    assert numbers_to_digits("appeared in eleven films") == "appeared in 11 films"
    assert numbers_to_digits("one dog and two cats") == "one dog and two cats"
    assert numbers_to_digits("forty two") == "42"


def test_basic_cleanup():
    assert basic_cleanup("I THINK I'LL GO") == "I think I'll go."
    assert basic_cleanup("WHAT TIME IS IT") == "What time is it?"
    assert basic_cleanup("") == ""


def test_prompt_lists_candidates_and_context():
    p = _user_prompt(["A B", "A C"], "earlier text")
    assert "1. A B" in p and "2. A C" in p and "earlier text" in p
