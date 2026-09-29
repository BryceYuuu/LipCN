"""Live typing picks only words two reads agree on, and the final swap keeps the shared start."""
from lipflow import app as A


class Fake:
    session = 1
    last_paste_at = 0.0
    live_typed = ""
    prev_partial: list = []


def run_live(reads, monkeypatch):
    typed = []
    monkeypatch.setattr(A, "type_text", lambda t: typed.append(t))
    f = Fake()
    f.prev_partial = []
    for r in reads:
        A.Lipflow._live_type.__wrapped__(f, 1, r.split()) if hasattr(A.Lipflow._live_type, "__wrapped__") \
            else A.Lipflow._live_type(f, 1, r.split())
    return f, typed


def test_types_only_stable_words(monkeypatch):
    f, typed = run_live(["hello mig", "hello miguel i", "hello miguel i am"], monkeypatch)
    assert "".join(typed) == "hello miguel i"
    assert f.live_typed == "hello miguel i"


def test_final_swap_keeps_common_prefix(monkeypatch):
    ops = []
    monkeypatch.setattr(A, "backspace", lambda n: ops.append(("bs", n)))
    monkeypatch.setattr(A, "type_text", lambda t: ops.append(("type", t)))
    f = Fake()
    f.live_typed = "hello miguel i am sending"
    A.Lipflow._insert_final(f, "Hello Miguel, I am sending you a message.")
    assert ops == [("bs", len("hello miguel i am sending")), ("type", "Hello Miguel, I am sending you a message.")]
    f.live_typed = "Hi there my"
    ops.clear()
    A.Lipflow._insert_final(f, "Hi there, my friend.")
    assert ops == [("bs", 3), ("type", ", my friend.")]
