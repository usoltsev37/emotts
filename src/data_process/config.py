from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatasetParams:

    text_dir: str
    mels_dir: str
    wav_dir: str
    feature_dir: str
    mels_fastspeech2_dir: str
    duration_dir: str
    pitch_dir: str
    energy_dir: str
    phones_dir: str
    ignore_speakers: List[str]
    text_ext: str = field(default=".TextGrid")
    mels_ext: str = field(default=".pkl")
    fastspeech2_ext: str = field(default=".npy")
    phones_ext: str = field(default=".txt")
    finetune_speakers: List[str] = field(
        default_factory=lambda: [f"00{i}" for i in range(11, 21)]
    )
    speaker_emb_dir: Optional[str] = field(default=None)
