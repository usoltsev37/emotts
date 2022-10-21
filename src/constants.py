import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union


PATHLIKE = Union[str, Path]
FEATURE_MODEL_FILENAME = "feature_model.pth"
FASTSPEECH2_MODEL_FILENAME = "fastspeech2_model.pth"
MELS_MEAN_FILENAME = "mels_mean.pth"
MELS_STD_FILENAME = "mels_std.pth"
ENERGY_MEAN_FILENAME = "energy_mean.pth"
ENERGY_STD_FILENAME = "energy_std.pth"
ENERGY_MIN_FILENAME = "energy_min.pth"
ENERGY_MAX_FILENAME = "energy_max.pth"
PITCH_MEAN_FILENAME = "pitch_mean.pth"
PITCH_STD_FILENAME = "pitch_std.pth"
PITCH_MIN_FILENAME = "pitch_min.pth"
PITCH_MAX_FILENAME = "pitch_max.pth"
PHONEMES_FILENAME = "phonemes.json"
SPEAKERS_FILENAME = "speakers.json"
CHECKPOINT_DIR = Path("checkpoints")
HIFI_CHECKPOINT_NAME = "hifi"
FEATURE_CHECKPOINT_NAME = "fastspeech2"
FASTSPEECH2_CHECKPOINT_NAME = "fastspeech2"
DATA_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
MODEL_DIR = Path("models")
REFERENCE_PATH = Path("references")
SPEAKER_PRINT_DIR = Path("speakers_prints")
REMOVE_SPEAKERS = ["p280", "p315", "0019"]

PHONEMES_ENG = [
    [
        "AY1",
        "K", "AE1", "N",
        "T",
        "B", "IH0", "L", "IY1", "V",
        "HH",
        "D", "IH0", "D",
        "IH0 T",
    ],
    [
        "HH",
        "HH", "AE1", "Z",
        "AH0", "B", "AE1", "N", "D", "AH0", "N", "D",
        "AH0", "L",
        "TH",
        "T", "R", "AH0", "D", "IH1", "SH", "AH0", "N", "Z",
        "HH", "IH1", "R",
    ],
    [
        "AY1",
        "M", "IY1", "N",
        "K", "AE1", "N",
        "Y", "UW1",
        "IH0", "M", "AE1", "JH", "AH0", "N",
        "EH1", "N", "IY0", "TH", "IH2", "NG",
        "M", "AO0", "R",
        "IH2", "N", "AH0", "P", "R", "OW1", "P", "R", "IY0", "AH0", "T",
    ],
    [
        "DH", "AE1", "T",
        "EH1", "S",
        "HH", "W", "AH1", "T",
        "TH",
        "K", "R", "AW1", "D",
        "N", "EH1", "V", "ER0",
        "IH0", "K", "S", "P", "EH1", "K", "T", "IH0", "D",
        "AH0", "L",
        "AA1", "F",
        "Y", "UW1", "EH1", "S",
        "W", "ER1",
        "AH0", "S", "T", "AA1", "N", "IH0", "SH", "T",
        "T", "OW0",
        "S", "IY1",
        "IH0", "T",
    ],
    [
        "N", "OW0",
        "HH",
        "W", "AH1", "N", "D", "ER0", "Z",
        "W", "AY1",
        "P", "IY1", "P", "AH0", "L",
        "TH", "IH1", "NG", "K",
        "IH1", "S",
        "AH0",
        "B", "IH1", "T",
        "AA1", "D",
    ],
    [
        "DH", "AE1", "T",
        "IH1", "S",
        "IH0", "N", "S", "EY1", "N",
    ]
]

PHONEMES_CHI = [
    ['q', 'i1', 'ng', 'ch', 'u1', 'v2', 'l', 'a2', 'n', 'e2', 'r', 'sh', 'e4', 'ng', 'v2', 'l', 'a2', 'n'],
    ['t', 'ia1', 'n', 'd', 'ao4', 'ch', 'ou2', 'q', 'i2', 'n'],
    ['j', 'iou3', 't', 'ia1', 'n', 'l', 'a3', 'n', 've4'],
    ['s', 'ai1', 'ue1', 'n', 'sh', 'ii1', 'm', 'a3', 'ia1', 'n', 'zh', 'ii1', 'f', 'ei1', 'f', 'u2'],
    ['i1', 'm', 'i2', 'ng', 'j', 'i1', 'ng', 'r', 'e2', 'n'],
    ['i1', 's', 'ii1', 'b', 'u4', 'g', 'ou3'],
    ['i1', 'j', 'ia4', 'n', 'sh', 'ua1', 'ng', 'd', 'iao1'],
    ['sh', 'a1', 'n', 'v3', 'v4', 'l', 'ai2', 'f', 'e1', 'ng', 'm', 'a3', 'n', 'l', 'ou2'],
    ['m', 'a2', 'q', 've4', 's', 'uei1', 'x', 'iao3', 'u3', 'z', 'a4', 'ng', 'j', 'v4', 'q', 'va2', 'n'],
    ['q', 'ia2', 'ng', 'l', 'o2', 'ng', 'n', 'a2', 'n', 'ia1', 'd', 'i4', 't', 'ou2', 'sh', 'e2'],
    ['q', 'ia2', 'n', 'p', 'a4', 'l', 'a2', 'ng', 'h', 'ou4', 'p', 'a4', 'h', 'u3'],
    ['d', 'a4', 'zh', 'ii4', 'r', 'uo4', 'v2']
]

RUSSIAN_SPEAKERS = {0: "Игорина"}
try:
    with open("models/en/tacotron/speakers.json", "r") as json_file:
        ENGLISH_SPEAKERS = json.load(json_file)
except FileNotFoundError:
    ENGLISH_SPEAKERS = {0: "Speakers Loading Error"}


@dataclass
class TacoTronCheckpoint:
    path: Path
    model_file_name: str = FEATURE_MODEL_FILENAME
    phonemes_file_name: str = PHONEMES_FILENAME
    speakers_file_name: str = SPEAKERS_FILENAME
    mels_mean_filename: str = MELS_MEAN_FILENAME
    mels_std_filename: str = MELS_STD_FILENAME


#
# @dataclass
# class Emotion:
#     name: str
#     api_name: str
#     reference_mels_path: PATHLIKE
#     ru_speaker_id: int
#
#
# @dataclass
# class SupportedEmotions:
#     angry: Emotion = Emotion(
#         name="angry",
#         api_name="angry",
#         reference_mels_path="Angry.pkl",
#         ru_speaker_id=10,
#     )
#     happy: Emotion = Emotion(
#         name="happy",
#         api_name="happy",
#         reference_mels_path="Happy.pkl",
#         ru_speaker_id=21,
#     )
#     neutral: Emotion = Emotion(
#         name="neutral",
#         api_name="neutral",
#         reference_mels_path="Neutral.pkl",
#         ru_speaker_id=13,
#     )
#     sad: Emotion = Emotion(
#         name="sad", api_name="sad", reference_mels_path="Sad.pkl", ru_speaker_id=40
#     )
#     surprise: Emotion = Emotion(
#         name="surprise",
#         api_name="surprise",
#         reference_mels_path="Surprise.pkl",
#         ru_speaker_id=0,
#     )
#     very_angry: Emotion = Emotion(
#         name="very_angry",
#         api_name="veryangry",
#         reference_mels_path="Very_angry.pkl",
#         ru_speaker_id=41,
#     )
#     very_happy: Emotion = Emotion(
#         name="very_happy",
#         api_name="veryhappy",
#         reference_mels_path="Very_happy.pkl",
#         ru_speaker_id=12,
#     )
#
#
# @dataclass
# class Language:
#     name: str
#     api_name: str
#     emo_reference_dir: Path
#     emo_selector: dict
#     speaker_selector: dict
#     g2p_model_path: Path
#     tacotron_checkpoint: TacoTronCheckpoint
#     hifi_params: HIFIParams
#     test_phrase: str
#
#
# @dataclass
# class SupportedLanguages:
#     english: Language = Language(
#         name="English (en-EN)",
#         api_name="en",
#         emo_reference_dir=Path("models/en/emo_reference"),
#         emo_selector={
#             "🙂 happy": SupportedEmotions.happy,
#             "😲 surprise": SupportedEmotions.surprise,
#             "😐 neutral": SupportedEmotions.neutral,
#             "😞 sad": SupportedEmotions.sad,
#             "😡 angry": SupportedEmotions.angry,
#         },
#         speaker_selector=ENGLISH_SPEAKERS,
#         g2p_model_path=Path("models/en/g2p/english_g2p.zip"),
#         tacotron_checkpoint=TacoTronCheckpoint(path=Path("models/en/tacotron")),
#         hifi_params=HIFIParams(
#             dir_path="en/hifi", config_name="config.json", model_name="generator.hifi"
#         ),
#         test_phrase="How to fit linear regression?",
#     )
#     russian: Language = Language(
#         name="Russian (ru-RU)",
#         api_name="ru",
#         emo_reference_dir=Path("models/ru/emo_reference/mels"),
#         emo_selector={
#             "😃 happy+": SupportedEmotions.very_happy,
#             "🙂 happy": SupportedEmotions.happy,
#             "😐 neutral": SupportedEmotions.neutral,
#             "😞 sad": SupportedEmotions.sad,
#             "😒 angry": SupportedEmotions.angry,
#             "😡 angry+": SupportedEmotions.very_angry,
#         },
#         speaker_selector=RUSSIAN_SPEAKERS,
#         g2p_model_path=Path("models/ru/g2p/russian_g2p.zip"),
#         tacotron_checkpoint=TacoTronCheckpoint(path=Path("models/ru/tacotron")),
#         hifi_params=HIFIParams(
#             dir_path="ru/hifi", config_name="config.json", model_name="generator.hifi"
#         ),
#         test_phrase="Я усиленно обогреваю серверную в эти холодные зимние дни",
#     )
