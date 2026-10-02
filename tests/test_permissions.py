from lipflow.onboarding import grant_covers


def test_grant_covers_only_an_allow_for_this_binary():
    cd = "9e8234086188f550281367c2f8d27aabe79baa8a"
    blob = bytes.fromhex("aabb") + bytes.fromhex(cd) + bytes.fromhex("ccdd")
    assert grant_covers(cd.upper(), 2, blob)
    assert not grant_covers("ab" * 20, 2, blob)
    assert not grant_covers(cd, 0, blob)  # denied
    assert not grant_covers(cd, 2, None)
    assert not grant_covers("", 2, blob)
