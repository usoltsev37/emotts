#!/usr/bin/env python
from zipfile import PyZipFile
from pathlib import Path
import os
import shutil
from typing import Dict

import click
from tqdm import tqdm

encodings_eng = {
    "0011": "ascii",
    "0012": "utf16",
    "0013": "utf16",
    "0014": "utf16",
    "0015": "ascii",
    "0016": "iso8859",
    "0017": "iso8859",
    "0018": "utf16",
    "0019": "utf16",
    "0020": "ascii",
}
encodings_chi = {
    "0001": "gb2312",
    "0002": "gb2312",
    "0003": "utf16",
    "0004": "gb2312",
    "0005": "gb2312",
    "0006": "utf16",
    "0007": "utf16",
    "0008": "utf16",
    "0009": "gb2312",
    "0010": "utf16",
}

bug_lines = ["0015_000217", "0015_000175", "0015_000138", 
             "0015_000137", "0013_000431", "0012_001355", 
             "0013_000914", "0015_000187", "0015_000668",
            "0015_000690", "0015_000875", "0015_001015",
            "0015_001531", "0015_001578"]

@click.command()
@click.option("--input-esd-zip", type=Path,
              help="Path to esd dataset in zip.")
@click.option("--output-dir", type=Path, default="trimmed",
              help="outdir")
def main(input_esd_zip: Path, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    eng_output_dir = output_dir / Path("english")
    chi_output_dir = output_dir / Path("chinese")

    eng_output_dir.mkdir(exist_ok=True, parents=True)
    chi_output_dir.mkdir(exist_ok=True, parents=True)

    emmots_dict = {}
    pzf = PyZipFile(input_esd_zip)
    pzf.extractall()
    basedir_with_data = Path(input_esd_zip.stem)
    print(basedir_with_data)
    for path_to_dir_speaker in basedir_with_data.iterdir():
        if path_to_dir_speaker.is_dir():
            if int(path_to_dir_speaker.stem) > 10: # if indx speaker more then ten, that english speaker else chinese speaker
                process_one_speacker(path_to_dir_speaker, eng_output_dir / Path("text"), eng_output_dir / Path("audio"), emmots_dict)
            else:
                process_one_speacker(path_to_dir_speaker, chi_output_dir / Path("text"), chi_output_dir / Path("audio"), emmots_dict)
    
    shutil.rmtree(basedir_with_data)

    for speaker_dir in (eng_output_dir / Path("text")).iterdir():
        text_file = speaker_dir / Path(speaker_dir.stem + ".txt")
        process_file_with_text(text_file, speaker_dir, encodings_eng)
        os.remove(text_file)

    for speaker_dir in (chi_output_dir / Path("text")).iterdir():
        text_file = speaker_dir / Path(speaker_dir.stem + ".txt")
        process_file_with_text(text_file, speaker_dir, encodings_chi)
        os.remove(text_file)




def process_one_speacker(path_dir_speaker: Path, output_text_dir: Path, output_audio_dir: Path, emmots_dict: Dict):
    speacker_id = Path(path_dir_speaker.stem)
    output_audio_dir_cur_speacker = output_audio_dir / Path(speacker_id)
    output_audio_dir_cur_speacker.mkdir(exist_ok=True, parents=True)
    
    output_text_dir_cur_speacker = output_text_dir / Path(speacker_id)
    output_text_dir_cur_speacker.mkdir(exist_ok=True, parents=True)
    
    for emmots_dir_path in path_dir_speaker.iterdir():
        if emmots_dir_path.is_dir():
            emmots = emmots_dir_path.stem
            for subset in emmots_dir_path.iterdir():
                for audio_file in subset.iterdir():
                    emmots_dict[audio_file.name] = emmots
                    audio_file.rename(output_audio_dir_cur_speacker.joinpath(audio_file.name))
                    
        else:
            emmots_dir_path.rename(output_text_dir_cur_speacker.joinpath(emmots_dir_path.name))    

def create_file(filename: str, text: str):
    with open(filename, 'w') as fil:
        fil.write(text)

def process_file_with_text(text_file: Path, speaker_dir: Path, encodings: Dict):
    with open(text_file, encoding=encodings[text_file.stem]) as fil:
        for line in fil.readlines():
            for bug in bug_lines:
                if bug in line:
                    line = line.replace(" ", "\t", 1)
            if line.rstrip() == "":
                continue
            data = line.split('\t')
            filename, text, emmot = data[0], data[1], data[2]
            filename = filename +'.txt'
            create_file(speaker_dir / filename, text)

if __name__ == "__main__":
    main()


