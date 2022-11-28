import json
from pathlib import Path
from typing import Dict

import numpy as np
import tgt
import torch
from tqdm import tqdm

from src.constants import (
    CHECKPOINT_DIR, FEATURE_CHECKPOINT_NAME, FEATURE_MODEL_FILENAME,
    MELS_MEAN_FILENAME, MELS_STD_FILENAME, PHONEMES_FILENAME, REMOVE_SPEAKERS,
    SPEAKERS_FILENAME,
)
from src.data_process.voiceprint_variance_adaptor_dataset import VoicePrintVarianceBatch
from src.models.feature_models.non_attentive_tacotron import (
    NonAttentiveTacotronVoicePrintVarianceAdaptor,
)
from src.train_config import load_config


class Inferencer:

    PAD_TOKEN = "<PAD>"
    PHONES_TIER = "phones"
    LEXICON_OOV_TOKEN = "spn"
    MEL_EXT = "pth"
    _speaker_emb_ext = ".npy"

    def __init__(
        self, config_path: str
    ):
        config = load_config(config_path)
        checkpoint_path = CHECKPOINT_DIR / config.checkpoint_name / FEATURE_CHECKPOINT_NAME
        with open(checkpoint_path / PHONEMES_FILENAME) as f:
            self.phonemes_to_idx: Dict[str, int] = json.load(f)
        with open(checkpoint_path / SPEAKERS_FILENAME) as f:
            self.speakers_to_idx: Dict[str, int] = json.load(f)
        self.sample_rate = config.sample_rate
        self.hop_size = config.hop_size
        self.device = torch.device(config.device)
        self.feature_model: NonAttentiveTacotronVoicePrintVarianceAdaptor = torch.load(
            checkpoint_path / FEATURE_MODEL_FILENAME, map_location=config.device
        )
        if isinstance(self.feature_model.attention.eps, float):
            self.feature_model.attention.eps = torch.Tensor([self.feature_model.attention.eps])
        self._mels_dir = Path(config.data.mels_dir)
        self._duration_dir = Path(config.data.duration_dir)
        self._phonemes_dir = Path(config.data.phones_dir)
        self._phones_ext = config.data.phones_ext
        self._energy_dir = Path(config.data.energy_dir)
        self._pitch_dir = Path(config.data.pitch_dir)
        self._speaker_emb_dir = Path(config.data.speaker_emb_dir)
        self._mels_ext = config.data.mels_ext
        self.feature_model_mels_path = Path(config.data.feature_dir)
        self.feature_model_mels_path.mkdir(parents=True, exist_ok=True)
        self.mels_mean = torch.load(checkpoint_path / MELS_MEAN_FILENAME)
        self.mels_std = torch.load(checkpoint_path / MELS_STD_FILENAME)
        if config.finetune:
            self.speaker_to_use = config.data.finetune_speakers
        else:
            self.speaker_to_use = [
                speaker.name
                for speaker in self._mels_dir.iterdir()
                if speaker not in config.data.finetune_speakers
            ]


    def proceed_data(self) -> None:
        mels_set = {
            Path(x.parent.name) / x.stem
            for x in self._mels_dir.rglob(f"*{self._mels_ext}")
        }
        speaker_emb_set = {
            Path(x.parent.name) / x.stem
            for x in self._speaker_emb_dir.rglob(f"*{self._speaker_emb_ext}")
        }
        duration_set = {
            Path(x.parent.name) / x.stem
            for x in self._duration_dir.rglob(f"*{self._mels_ext}")
        }
        phones_set = {
            Path(x.parent.name) / x.stem
            for x in self._phonemes_dir.rglob(f"*{self._phones_ext}")
        }
        enegry_set = {
            Path(x.parent.name) / x.stem
            for x in self._energy_dir.rglob(f"*{self._mels_ext}")
        }
        pitch_set = {
            Path(x.parent.name) / x.stem
            for x in self._pitch_dir.rglob(f"*{self._mels_ext}")
        }
        samples = list(mels_set & duration_set & speaker_emb_set & phones_set & enegry_set & pitch_set)
        for sample in tqdm(samples):
            if sample.parent.name in REMOVE_SPEAKERS:
                continue
            elif sample.parent.name not in self.speaker_to_use:
                continue

            

            save_dir = self.feature_model_mels_path / sample.parent.name
            save_dir.mkdir(exist_ok=True)
            filepath = save_dir / f"{sample.name}.{self.MEL_EXT}"
            if filepath.exists():
                continue

            duration_path = (self._duration_dir / sample).with_suffix(self._mels_ext)
            phonemes_path = (self._phonemes_dir / sample).with_suffix(self._phones_ext)

            mels_path = (self._mels_dir / sample).with_suffix(self._mels_ext)
            speaker_emb_path = (self._speaker_emb_dir / sample).with_suffix(self._speaker_emb_ext)

            energy_path = (self._energy_dir / sample).with_suffix(self._mels_ext)
            pitch_path = (self._pitch_dir / sample).with_suffix(self._mels_ext)

            phonemes = open(phonemes_path).read().split(" ")

            if len(phonemes) == 0:
                continue

            if self.LEXICON_OOV_TOKEN in phonemes:
                continue

            speaker_id = self.speakers_to_idx[sample.parent.name]

            phoneme_ids = []
            for phoneme in phonemes:
                phoneme_ids.append(self.phonemes_to_idx[phoneme])

            durations = np.load(duration_path)
            energy = np.load(energy_path)
            pitch = np.load(pitch_path)

            mels: torch.Tensor = torch.Tensor(np.load(mels_path)).unsqueeze(0)
            mels = (mels - self.mels_mean) / self.mels_std



            speaker_emb_path = (self._speaker_emb_dir / sample).with_suffix(self._speaker_emb_ext)
            speaker_emb_array = np.load(str(speaker_emb_path)).astype(np.float32)
            speaker_emb_tensor = torch.from_numpy(speaker_emb_array).unsqueeze(0)

            with torch.no_grad():
                batch = VoicePrintVarianceBatch(
                    phonemes=torch.LongTensor([phoneme_ids]).to(self.device),
                    num_phonemes=torch.LongTensor([len(phoneme_ids)]),
                    speaker_ids=torch.LongTensor([speaker_id]).to(self.device),
                    speaker_embs=speaker_emb_tensor.to(self.device),
                    durations=torch.FloatTensor([durations]).to(self.device),
                    mels=mels.permute(0, 2, 1).float().to(self.device),
                    energies=torch.FloatTensor([energy]).to(self.device),
                    pitches=torch.FloatTensor([pitch]).to(self.device),

                )
                _, output, _, _, _, _ = self.feature_model(batch)
                output = output.permute(0, 2, 1).squeeze(0)
                output = output * self.mels_std.to(self.device) + self.mels_mean.to(self.device)

            torch.save(output.float(), filepath)
