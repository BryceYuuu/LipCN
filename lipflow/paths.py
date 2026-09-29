"""Where each person's data lives. Shared model weights stay in the repo's models/; everything
learned from *you* (clips, phrases, personal models, settings) goes here.

Override with LIPFLOW_HOME (tests use a temp dir so they never touch your real data)."""
import os

HOME = os.environ.get("LIPFLOW_HOME") or os.path.expanduser("~/Library/Application Support/Lipflow")
PERSONAL_MODELS = os.path.join(HOME, "models")
PERSONAL_VSR = os.path.join(PERSONAL_MODELS, "vsr_face.pth")
PERSONAL_LM = os.path.join(PERSONAL_MODELS, "lm_phrasing.pth")

# The app bundle's launcher sets LIPFLOW_APP=1: permissions then belong to "Lipflow", not the terminal.
WHO = "Lipflow" if os.environ.get("LIPFLOW_APP") else "your terminal"
