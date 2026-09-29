"""Stand-ins for the distutils helpers ESPnet used (distutils is gone in Python 3.12+)."""
import re


def strtobool(val):
    val = str(val).lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    if val in ("n", "no", "f", "false", "off", "0"):
        return 0
    raise ValueError(f"invalid truth value {val!r}")


class LooseVersion:
    def __init__(self, v):
        self.parts = tuple(int(p) for p in re.findall(r"\d+", str(v).split("+")[0])[:3])

    def __lt__(self, o): return self.parts < o.parts
    def __le__(self, o): return self.parts <= o.parts
    def __gt__(self, o): return self.parts > o.parts
    def __ge__(self, o): return self.parts >= o.parts
    def __eq__(self, o): return self.parts == o.parts
