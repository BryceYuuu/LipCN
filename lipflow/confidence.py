"""Conservative review routing. Scores and margins are evidence, NOT probabilities.

No camera-domain calibration is shipped. Auto routing is opt-in; default is review.
"""
from dataclasses import dataclass, asdict
import math

from .text import tokens


@dataclass(frozen=True)
class Hypothesis:
    text: str
    score: float
    token_count: int

    @property
    def normalized_score(self):
        return self.score / max(self.token_count, 1)


@dataclass(frozen=True)
class Quality:
    face_ratio: float = 1.0
    brightness: float = 128.0
    contrast: float = 30.0
    mouth_pixels: float = 100.0

    def problem(self):
        if not all(math.isfinite(v) for v in (self.face_ratio, self.brightness, self.contrast, self.mouth_pixels)):
            return "Invalid camera quality / 画面质量无法确认，请重说"
        if self.face_ratio < 0.7:
            return "Keep your mouth facing the camera / 请正对摄像头"
        if self.mouth_pixels < 35:
            return "Move closer to the camera / 请靠近摄像头"
        if self.brightness < 35 or self.brightness > 225 or self.contrast < 8:
            return "Improve the lighting / 请改善光线"
        return ""


@dataclass(frozen=True)
class Assessment:
    action: str  # retry / review / auto
    reason: str
    margin: float | None
    agreement: bool

    def asdict(self):
        return asdict(self)


def assess(hypotheses, greedy, quality: Quality, policy='review', min_margin=0.5):
    problem = quality.problem()
    if problem:
        return Assessment('retry', problem, None, False)
    if not hypotheses or not tokens(hypotheses[0].text):
        return Assessment('retry', 'Please repeat more slowly / 请慢一点重说', None, False)
    text_tokens = tokens(hypotheses[0].text)
    if '<unk>' in hypotheses[0].text or (len(text_tokens) >= 6 and len(set(text_tokens)) <= 2):
        return Assessment('retry', 'Unclear or repetitive output / 识别不清，请重说', None, False)
    valid = all(math.isfinite(h.score) and h.token_count > 0 for h in hypotheses)
    agreement = tokens(greedy) == tokens(hypotheses[0].text)
    margin = (hypotheses[0].normalized_score - hypotheses[1].normalized_score
              if len(hypotheses) > 1 and valid else None)
    # Missing scores/alternatives, non-finite values, or decoder disagreement must never auto-paste.
    if policy == 'auto' and margin is not None and margin >= min_margin and agreement:
        return Assessment('auto', 'Heuristic agreement (not calibrated)', margin, True)
    return Assessment('review', 'Choose a candidate / 请选择候选（未校准）', margin, agreement)
