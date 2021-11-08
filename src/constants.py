from pathlib import Path
from typing import Union

PATHLIKE = Union[str, Path]
FEATURE_MODEL_FILENAME = "feature_model.pth"
PHONEMES_FILENAME = "phonemes.json"
SPEAKERS_FILENAME = "speakers.json"
CHECKPOINT_DIR = Path("checkpoints")
DATA_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
MODEL_DIR = Path("models")
