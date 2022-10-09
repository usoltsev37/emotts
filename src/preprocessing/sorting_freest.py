#!/usr/bin/env python
from zipfile import PyZipFile
from pathlib import Path
import os
import shutil
from typing import List

import click
from tqdm import tqdm


@click.command()
@click.option("--input-dir", type=Path,
              help="Path to esd dataset in zip.")
@click.option("--output-dir", type=Path, default="trimmed",
              help="outdir")

def main(input_dir: Path, output_dir: Path) -> None:

    audio_output_dir = output_dir / Path("audio")
    text_output_dir = output_dir / Path("text")
    audio_output_dir.mkdir(exist_ok=True, parents=True)
    text_output_dir.mkdir(exist_ok=True, parents=True)

    print("Move *.wav files")
    move_file(audio_output_dir, list(input_dir.rglob("*.wav")))
    print("Move *.txt files")
    move_file(text_output_dir, list(input_dir.rglob("*.txt")))

    shutil.rmtree(input_dir)


def move_file(output_dir: Path, path_list: List) -> None:
    for path in tqdm(path_list):
        speacker_name = path.name[8:8 + 7]
        number_utter = path.name[15: -4]
        new_path = output_dir / Path(speacker_name) / Path(speacker_name + "_" + number_utter + path.suffix)
        new_path.parent.mkdir(exist_ok=True, parents=True)
        path.rename(new_path)

if __name__ == "__main__":
    main()


