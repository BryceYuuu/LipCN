# Lipflow 👄

**Wispr Flow for your lips.** Hold a key, silently mouth what you want to say, let go, and the
text shows up at your cursor in whatever app you're in. No microphone and no sound, just your webcam.

Everything runs locally on your Mac. An optional LLM pass fixes the words lip reading gets wrong.

```
 hold ⌥ (right)  ──►  webcam  ──►  face landmarks (live)  ──►  mouth crops, 25 fps
                                                                   │
   paste at cursor  ◄──  LLM cleanup  ◄──  beam search + LM  ◄──  VSR encoder (Apple GPU)
                                       ▲
                     live preview: greedy CTC every 0.45 s while you talk
```

## Setup

```sh
./setup.sh            # uv sync + ~1.2 GB of models
uv run lipflow        # starts the menu-bar app (👄 in the menu bar)
```

The first time you use it, macOS asks for three permissions for whichever app launched
Lipflow (your terminal):

| Permission | Why |
|---|---|
| **Camera** | to see your mouth |
| **Input Monitoring** | to notice the push-to-talk key anywhere |
| **Accessibility** | to press ⌘V into the focused app |

Run `uv run lipflow doctor` to check them all. After granting Input Monitoring, restart the terminal.

### Better accuracy: turn on LLM cleanup

Lip reading can't tell apart words that look the same on the lips (p/b/m, f/v, t/d/n…), so the raw
model output reads like "WALLET OFFICER" when you said "while in office". Lipflow sends the model's
top-3 guesses plus your last few dictations to an LLM, which picks the sentence you meant and
fixes casing, punctuation and numbers. The first backend that's available is used:

1. **Claude**: `export ANTHROPIC_API_KEY=…` (model `claude-opus-5-5` at low effort; override with
   `LIPFLOW_MODEL`, e.g. `LIPFLOW_MODEL=claude-haiku-4-5` for lower latency). Best at fixing badly
   mis-read sentences.
2. **Local** (the default without a key): Qwen3-0.6B 4-bit running in-process on Apple Silicon
   via MLX. About 350 MB, downloaded on first launch, and about 0.2 s per sentence, fully offline.
   Tiny models copy the formatting they're shown, so this one gets lowercase guesses and a few
   worked examples (`SMALL_SHOTS` in `cleanup.py`). Override with `LIPFLOW_LOCAL_MODEL`.
3. **Ollama**: `--cleanup ollama` with `ollama pull qwen3:4b` (override with `LIPFLOW_OLLAMA_MODEL`).
4. **Offline rules**: sentence case, "I", end punctuation, "nineteen forty three" → 1943.

**Custom words.** Names are the hardest thing to lip-read (a name is just lip shapes). Put yours
in 👄 → *Edit custom words*, one per line (`~/Library/Application Support/Lipflow/words.txt`).
A guess that contains one of your words wins over the others and gets your capitalization, and
the LLM is told about them.

## Using it

| Do this | To |
|---|---|
| Hold **Right Option**, mouth the words, release | dictate |
| Double-tap **Right Option** … tap again | hands-free (up to 60 s) |
| **Esc** while listening | cancel |
| 👄 menu → Copy last dictation / Open history | get text back |

Lipflow keeps filming for 0.4 s after you release the key, because the model needs the frames
after the last word to read it.

Options: `uv run lipflow --help`

```
--key {right_option,left_option,right_command,right_control,fn}
--cleanup {auto,claude,local,ollama,basic}
--beam N          beam size (default 10)
--copy-only       copy to the clipboard instead of pasting
--camera N|FILE   camera index, or a video file to stand in for the webcam
```

Lip-read a video file: `uv run lipflow file talk.mp4 --start 10 --end 20`

## How it works

- **Model:** [Auto-AVSR](https://github.com/mpc001/auto_avsr) visual-only speech recognition trained
  on LRS3 (19.1% WER on the benchmark). A 3D-conv ResNet front end and a Conformer encoder feed a
  Transformer decoder plus CTC, with a subword Transformer language model in the beam search.
- **Preprocessing:** MediaPipe FaceLandmarker runs on every frame *while you're recording*, so
  there's no second detection pass afterwards. Eye, nose-base and mouth anchors are aligned to the
  training mean face, then 96×96 grayscale mouth crops are resampled to the 25 fps the model expects.
- **Speed (M4 Pro, 9 s utterance):** encoder 0.16 s on the Apple GPU (it's 1.7 s on CPU), beam
  search about 0.8–1.6 s on CPU (faster than MPS for thousands of tiny ops), for about 1–2 s from
  release to text. Two patches to the vendored ESPnet help: cross-attention keys/values are
  projected once per utterance instead of per hypothesis per step (25% faster beam search), and CTC
  scoring works on any device.

Accuracy on held-out news footage (public-domain White House addresses), raw model output:

| Said | Read |
|---|---|
| Born in New York City, and raised mostly in Chicago, Nancy Davis graduated from Smith College in 1943. | BORN IN NEW YORK CITY AND RAISED MOSTLY IN CHICAGO NANCY DAVIS GRADUATED FROM SMITH COLLEGE IN NINETEEN FORTY THREE |
| …a real-life Hollywood romance with the love of her life, Ronald Reagan, whom she married in 1952. | ROMANCE WITH THE LOVE OF HER LIFE RONALD REAGAN WHOM SHE MARRIED IN NINETEEN FIFTY TWO |
| Presidents have delivered some form of final message while in office - a farewell address to the American people. | PRESIDENTS HAVE DELIVERED SOME FORM OF FINAL MESSAGE WHILE IN OFFICE FAREWELL ADDRESS TO THE AMERICAN PEOPLE |

Silently mouthed speech is harder than filmed speech (smaller lip movements), so expect more
errors on your own webcam. That's what the LLM cleanup is for.

## Development

```sh
./setup.sh --samples      # also fetch the public-domain test clips
uv run pytest             # the paste test is opt-in: LIPFLOW_TEST_PASTE=1
```

Code map: `lipflow/face.py` (landmarks → mouth crops), `vsr.py` (model), `camera.py`, `hotkey.py`
(Quartz event tap; pynput's macOS listener crashes on recent macOS), `paste.py`, `hud.py`,
`cleanup.py`, `app.py` (wiring + menu bar). See `NOTICE` for bundled code and model licensing.
The LRS3-trained weights are for non-commercial research use.
