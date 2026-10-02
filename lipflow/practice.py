"""Practice sentences and the clips you record for them, shared by the macOS and Windows setup.

Clips are saved with their known text under <Lipflow home>/clips/onboarding. Training holds
N_HELD_OUT of them out and only keeps the face model if it reads those better than the stock model.
"""
from __future__ import annotations

import glob
import os
import random
import re
import time

import numpy as np

from .paths import HOME as DIR

CLIPS = os.path.join(DIR, "clips", "onboarding")
N_SENTENCES = 24
N_HELD_OUT = 6

# Harvard sentences (IEEE 1969 "Recommended Practice for Speech Quality Measurements", lists 1-6):
# short, phonetically balanced sentences, so practice covers every lip shape evenly. Used when there's
# no Wispr Flow history to practice your own sentences from.
HARVARD = [
    "The birch canoe slid on the smooth planks", "Glue the sheet to the dark blue background",
    "It's easy to tell the depth of a well", "These days a chicken leg is a rare dish",
    "Rice is often served in round bowls", "The juice of lemons makes fine punch",
    "The box was thrown beside the parked truck", "The hogs were fed chopped corn and garbage",
    "Four hours of steady work faced us", "A large size in stockings is hard to sell",
    "The boy was there when the sun rose", "A rod is used to catch pink salmon",
    "The source of the huge river is the clear spring", "Kick the ball straight and follow through",
    "Help the woman get back to her feet", "A pot of tea helps to pass the evening",
    "Smoky fires lack flame and heat", "The soft cushion broke the man's fall",
    "The salt breeze came across from the sea", "The girl at the booth sold fifty bonds",
    "The small pup gnawed a hole in the sock", "The fish twisted and turned on the bent hook",
    "Press the pants and sew a button on the vest", "The swan dive was far short of perfect",
    "The beauty of the view stunned the young boy", "Two blue fish swam in the tank",
    "Her purse was full of useless trash", "The colt reared and threw the tall rider",
    "It snowed rained and hailed the same morning", "Read verse out loud for pleasure",
    "Hoist the load to your left shoulder", "Take the winding path to reach the lake",
    "Note closely the size of the gas tank", "Wipe the grease off his dirty face",
    "Mend the coat before you go out", "The wrist was badly strained and hung limp",
    "The stray cat gave birth to kittens", "The young girl gave no clear response",
    "The meal was cooked before the bell rang", "What joy there is in living",
    "A king ruled the state in the early days", "The ship was torn apart on the sharp reef",
    "Sickness kept him home the third week", "The wide road shimmered in the hot sun",
    "The lazy cow lay in the cool grass", "Lift the square stone over the fence",
    "The rope will bind the seven books at once", "Hop over the fence and plunge in",
    "The friendly gang left the drug store", "Mesh wire keeps chicks inside",
    "The frosty air passed through the coat", "The crooked maze failed to fool the mouse",
    "Adding fast leads to wrong sums", "The show was a flop from the very start",
    "A saw is a tool used for making boards", "The wagon moved on well oiled wheels",
    "March the soldiers past the next hill", "A cup of sugar makes sweet fudge",
    "Place a rosebush near the porch steps", "Both lost their lives in the raging storm",
]


def practice_sentences(n: int = N_SENTENCES) -> list[str]:
    """Half your own everyday sentences (from an imported Wispr Flow history: 5–12 words, no digits)
    for your real vocabulary, half Harvard sentences for even coverage of lip shapes; all Harvard
    if there's no history. Shuffled together."""
    from .personal import PHRASES
    mine = []
    if os.path.exists(PHRASES):
        for line in open(PHRASES, encoding="utf-8"):
            for s in re.split(r"(?<=[.!?])\s+", line.strip()):
                w = s.split()
                if 5 <= len(w) <= 12 and not re.search(r"\d|http|@|/", s):
                    mine.append(s.rstrip(".!?,"))
    random.shuffle(mine)
    own = list(dict.fromkeys(mine))[:n // 2]
    harvard = random.sample(HARVARD, n - len(own))
    out = own + harvard
    random.shuffle(out)
    return out


def saved_clips() -> list[dict]:
    items = []
    for p in sorted(glob.glob(os.path.join(CLIPS, "*.npz"))):
        d = np.load(p, allow_pickle=True)
        items.append({"rois": d["rois"], "text": str(d["text"]), "path": p})
    return items


def save_clip(rois, text: str, raw: str = "") -> str:
    os.makedirs(CLIPS, exist_ok=True)
    path = os.path.join(CLIPS, f"{int(time.time() * 1000)}.npz")
    np.savez_compressed(path, rois=rois, text=text, raw=raw or "")
    return path
