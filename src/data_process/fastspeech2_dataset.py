import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data_process.config import DatasetParams
from src.constants import REMOVE_SPEAKERS

NUMBER = Union[int, float]
PHONES_TIER = "phones"
PAD_TOKEN = "<PAD>"


@dataclass
class FastSpeech2Sample:

    phonemes: List[int]
    num_phonemes: int
    speaker_id: int
    duration: np.array
    mel: torch.Tensor
    energy: np.array
    pitch: np.array



@dataclass
class FastSpeech2Info:

    phonemes_path: Path
    mel_path: Path
    energy_path: Path
    duration_path: Path
    pitch_path: Path
    speaker_id: int
    phonemes_length: int


@dataclass
class FastSpeech2Batch:
    
    speaker_ids: torch.Tensor
    phonemes: torch.Tensor
    num_phonemes: torch.Tensor
    max_phonemes_len: int
    mels: torch.Tensor
    max_mels_len: int
    energies: torch.Tensor
    pitches: torch.Tensor
    durations: torch.Tensor


class FastSpeech2Dataset(Dataset[FastSpeech2Sample]):
    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        mels_mean: torch.Tensor,
        mels_std: torch.Tensor,
        energy_mean: float,
        energy_std: float,
        pitch_mean: float,
        pitch_std: float,
        phoneme_to_ids: Dict[str, int],
        data: List[FastSpeech2Info],
    ):
        self._phoneme_to_id = phoneme_to_ids
        self._dataset = data
        self._dataset.sort(key=lambda x: x.phonemes_length)
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.mels_mean = mels_mean
        self.mels_std = mels_std
        self.energy_mean=energy_mean,
        self.energy_std=energy_std,
        self.pitch_mean=pitch_mean,
        self.pitch_std=pitch_std,

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> FastSpeech2Sample:

        info = self._dataset[idx]
        phoneme_collection = open(info.phonemes_path).read().split(" ")
        phoneme_ids = [
            self._phoneme_to_id[phoneme] for phoneme in phoneme_collection
        ]

        duration: np.array = np.load(info.duration_path)

        mels: torch.Tensor = torch.Tensor(np.load(info.mel_path))
        mels = (mels - self.mels_mean) / self.mels_std
        energy = np.load(info.energy_path)
        energy = (energy - self.energy_mean) / self.energy_std
        pitch = np.load(info.pitch_path)
        pitch = (pitch - self.pitch_mean) / self.pitch_std


        return FastSpeech2Sample(
            phonemes=phoneme_ids,
            num_phonemes=len(phoneme_ids),
            speaker_id=info.speaker_id,
            mel=mels,
            duration=duration,
            energy=energy,
            pitch=pitch
        )



class FastSpeech2Factory:

    """Create VCTK Dataset

    Note:
        * All the speeches from speaker ``p315`` will be skipped due to the lack of the corresponding text files.
        * All the speeches from ``p280`` will be skipped for ``mic_id="mic2"`` due to the lack of the audio files.
        * Some of the speeches from speaker ``p362`` will be skipped due to the lack of  the audio files.
        * See Also: https://datashare.is.ed.ac.uk/handle/10283/3443
        * Make sure to put the files as the following structure:
            text
            ├── p225
            |   ├──p225_001.TextGrid
            |   ├──p225_002.TextGrid
            |   └──...
            └── pXXX
                ├──pXXX_YYY.TextGrid
                └──...
            mels
            ├── p225
            |   ├──p225_001.pkl
            |   ├──p225_002.pkl
            |   └──...
            └── pXXX
                ├──pXXX_YYY.pkl
                └──...
    """

    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        n_mels: int,
        config: DatasetParams,
        phonemes_to_id: Dict[str, int],
        speakers_to_id: Dict[str, int],
        ignore_speakers: List[str],
        finetune: bool,
    ):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.finetune = finetune
        self._mels_dir = Path(config.mels_fastspeech2_dir)
        self._pitch_dir = Path(config.pitch_dir)
        self._phones_dir = Path(config.phones_dir)
        self._energy_dir = Path(config.energy_dir)
        self._duration_dir = Path(config.duration_dir)
        self._fastspeech2_ext = config.fastspeech2_ext
        self._phones_ext = config.phones_ext
        self.phoneme_to_id: Dict[str, int] = phonemes_to_id
        self.phoneme_to_id[PAD_TOKEN] = 0
        self.speaker_to_id: Dict[str, int] = speakers_to_id
        self.ignore_speakers = ignore_speakers
        if finetune:
            self.speaker_to_use = config.finetune_speakers
        else:
            self.speaker_to_use = [
                speaker.name
                for speaker in self._mels_dir.iterdir()
                if speaker not in config.finetune_speakers
            ]
        self._dataset: List[FastSpeech2Info] = self._build_dataset()
        self.mels_mean, self.mels_std = self._get_mean_and_std_mels()
        self.energy_mean, self.energy_std = self._get_mean_and_std_scalar(self._energy_dir, self._fastspeech2_ext)
        self.pitch_mean, self.pitch_std = self._get_mean_and_std_scalar(self._pitch_dir, self._fastspeech2_ext)

    @staticmethod
    def add_to_mapping(mapping: Dict[str, int], token: str) -> None:
        if token not in mapping:
            mapping[token] = len(mapping)

    def split_train_valid(
        self, test_fraction: float
    ) -> Tuple[FastSpeech2Dataset, FastSpeech2Dataset]:
        speakers_to_data_id: Dict[int, List[int]] = defaultdict(list)
        ignore_speaker_ids = {
            self.speaker_to_id[speaker] for speaker in self.ignore_speakers
        }
        for i, sample in enumerate(self._dataset):
            speakers_to_data_id[sample.speaker_id].append(i)
        test_ids: List[int] = []
        for speaker, ids in speakers_to_data_id.items():
            test_size = int(len(ids) * test_fraction)
            if test_size > 0 and speaker not in ignore_speaker_ids:
                test_indexes = random.choices(ids, k=test_size)
                test_ids.extend(test_indexes)

        train_data = []
        test_data = []
        for i in range(len(self._dataset)):
            if i in test_ids:
                test_data.append(self._dataset[i])
            else:
                train_data.append(self._dataset[i])
        train_dataset = FastSpeech2Dataset(
            sample_rate=self.sample_rate,
            hop_size=self.hop_size,
            mels_mean=self.mels_mean,
            mels_std=self.mels_std,
            energy_mean=self.energy_mean,
            energy_std=self.energy_std,
            pitch_mean=self.pitch_mean,
            pitch_std=self.pitch_std,
            phoneme_to_ids=self.phoneme_to_id,
            data=train_data,
        )
        test_dataset = FastSpeech2Dataset(
            sample_rate=self.sample_rate,
            hop_size=self.hop_size,
            mels_mean=self.mels_mean,
            mels_std=self.mels_std,
            energy_mean=self.energy_mean,
            energy_std=self.energy_std,
            pitch_mean=self.pitch_mean,
            pitch_std=self.pitch_std,
            phoneme_to_ids=self.phoneme_to_id,
            data=test_data,
        )
        return train_dataset, test_dataset

    def _build_dataset(self) -> List[FastSpeech2Info]:

        dataset: List[FastSpeech2Info] = []
        mels_set = {
            Path(x.parent.name) / x.stem
            for x in self._mels_dir.rglob(f"*{self._fastspeech2_ext}")
        }
        enegry_set = {
            Path(x.parent.name) / x.stem
            for x in self._energy_dir.rglob(f"*{self._fastspeech2_ext}")
        }
        pitch_set = {
            Path(x.parent.name) / x.stem
            for x in self._pitch_dir.rglob(f"*{self._fastspeech2_ext}")
        }
        duration_set = {
            Path(x.parent.name) / x.stem
            for x in self._duration_dir.rglob(f"*{self._fastspeech2_ext}")
        }
        phones_set = {
            Path(x.parent.name) / x.stem
            for x in self._phones_dir.rglob(f"*{self._phones_ext}")
        }           
        samples = list(mels_set & duration_set & pitch_set & enegry_set & phones_set)
        for sample in tqdm(samples):
            if sample.parent.name in REMOVE_SPEAKERS:
                continue

            mels_path = (self._mels_dir / sample).with_suffix(self._fastspeech2_ext)
            energy_path = (self._energy_dir / sample).with_suffix(self._fastspeech2_ext)
            pitch_path = (self._pitch_dir / sample).with_suffix(self._fastspeech2_ext)
            duration_path = (self._duration_dir / sample).with_suffix(self._fastspeech2_ext)
            phonemes_path = (self._phones_dir / sample).with_suffix(self._phones_ext)
            self.add_to_mapping(self.speaker_to_id, sample.parent.name)
            
            speaker_id = self.speaker_to_id[sample.parent.name]
            phonemes = open(phonemes_path).read().split(" ")
            if len(phonemes) > 0:

                for phoneme in phonemes:
                    self.add_to_mapping(self.phoneme_to_id, phoneme)

                if sample.parent.name in self.speaker_to_use:
                    dataset.append(
                        FastSpeech2Info(
                            phonemes_length=len(phonemes),
                            phonemes_path=phonemes_path,
                            energy_path=energy_path,
                            duration_path=duration_path,
                            pitch_path=pitch_path,
                            mel_path=mels_path,
                            speaker_id=speaker_id
                        )
                    )

        return dataset

    def _get_mean_and_std_mels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        mel_sum = torch.zeros(self.n_mels, dtype=torch.float64)
        mel_squared_sum = torch.zeros(self.n_mels, dtype=torch.float64)
        counts = 0

        for mel_path in self._mels_dir.rglob(f"*{self._fastspeech2_ext}"):
            if mel_path.parent.name in REMOVE_SPEAKERS:
                continue
            mels: torch.Tensor = torch.Tensor(np.load(mel_path))
            mel_sum += mels.sum(dim=-1)
            mel_squared_sum += (mels ** 2).sum(dim=-1)
            counts += mels.shape[-1]

        mels_mean: torch.Tensor = mel_sum / counts
        mels_std: torch.Tensor = torch.sqrt(
            (mel_squared_sum - mel_sum * mel_sum / counts) / counts
        )

        return mels_mean.view(-1, 1), mels_std.view(-1, 1)
    
    def _get_mean_and_std_scalar(self, scalars_path: Path, scalar_exts: str) -> Tuple[float, float]:
        scalar_sum = 0.0
        scalar_squared_sum = 0.0
        counts = 0

        for scalar_path in scalars_path.rglob(f"*{scalar_exts}"):
            if scalar_path.parent.name in REMOVE_SPEAKERS:
                continue
            scalar: torch.Tensor = torch.Tensor(np.load(scalar_path))
            scalar_sum += scalar.sum(dim=-1)
            scalar_squared_sum += (scalar ** 2).sum(dim=-1)
            counts += scalar.shape[-1]

        scalar_mean: torch.Tensor = scalar_sum / counts
        scalar_std: torch.Tensor = torch.sqrt(
            (scalar_squared_sum - scalar_sum * scalar_sum / counts) / counts
        )

        return scalar_mean, scalar_std




class FastSpeech2Collate:
    """
    Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, n_frames_per_step: int = 1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch: List[FastSpeech2Sample]) -> FastSpeech2Batch:
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [{...}, {...}, ...]
        """
        # Right zero-pad all one-hot text sequences to max input length
        batch_size = len(batch)
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x.phonemes) for x in batch]), dim=0, descending=True,
        )
        max_input_len = int(input_lengths[0])

        input_speaker_ids = torch.LongTensor(
            [batch[i].speaker_id for i in ids_sorted_decreasing]
        )

        text_padded = torch.zeros((batch_size, max_input_len), dtype=torch.long)
        durations_padded = torch.zeros((batch_size, max_input_len), dtype=torch.float)
        energy_padded = torch.zeros((batch_size, max_input_len), dtype=torch.float)
        pitch_padded = torch.zeros((batch_size, max_input_len), dtype=torch.float)
        
        for i, idx in enumerate(ids_sorted_decreasing):
            text = batch[idx].phonemes
            text_padded[i, : len(text)] = torch.LongTensor(text)
            durations = batch[idx].duration
            durations_padded[i, : len(durations)] = torch.FloatTensor(durations)
            energy = batch[idx].energy
            energy_padded[i, : len(energy)] = torch.FloatTensor(energy)
            pitch = batch[idx].pitch
            pitch_padded[i, : len(pitch)] = torch.FloatTensor(pitch)

        num_mels = batch[0].mel.size(0)
        max_target_len = max([x.mel.size(1) for x in batch])


        # include mel padded and gate padded
        mel_padded = torch.zeros(
            (batch_size, num_mels, max_target_len), dtype=torch.float
        )
        for i, idx in enumerate(ids_sorted_decreasing):
            mel: torch.Tensor = batch[idx].mel
            mel_padded[i, :, : mel.shape[1]] = mel
        mel_padded = mel_padded.permute(0, 2, 1)

        return FastSpeech2Batch(
            speaker_ids=input_speaker_ids,
            phonemes=text_padded,
            num_phonemes=input_lengths,
            max_phonemes_len=max_input_len,
            mels=mel_padded,
            max_mels_len=max_target_len,
            energies=energy_padded,
            pitches=pitch_padded,
            durations=durations_padded,
        )
