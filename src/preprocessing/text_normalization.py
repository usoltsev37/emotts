#!/usr/bin/env python
from pathlib import Path

import click
from text.cleaners import english_cleaners
from tqdm import tqdm
import subprocess as sub

@click.command()
@click.option("--input-dir", type=Path, required=True,
              help="Directory with texts to process.")
@click.option("--output-dir", type=Path, required=True,
              help="Directory for normalized texts.")
@click.option("--language", type=click.Choice(['english', 'chinese'], case_sensitive=True), default="english",
              help="choice language english/chinese")
def main(input_dir: Path, output_dir: Path, language: str) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)

    filepath_list = list(input_dir.rglob("*.txt"))
    print(f"Number of text files found: {len(filepath_list)}")
    print("Normalizing texts...")

    for filepath in tqdm(filepath_list):
        new_dir = output_dir / filepath.parent.name
        new_dir.mkdir(exist_ok=True)
        new_file = new_dir / filepath.name
        if language == "english":
            with open(filepath, "r") as fin, open(new_file, "w") as fout:
                content = fin.read()
                normalized_content = english_cleaners(content)
                fout.write(normalized_content)
        if language == "chinese":
            sub.run(["python", "src/preprocessing/text/cn_tn.py", filepath, new_file],stdout=sub.DEVNULL,
    stderr=sub.STDOUT)
            data = open(new_file, 'r').read()
            with open(new_file, 'w') as fin:
                for i, sym in enumerate(data):
                    if i == len(data) - 1:
                        fin.write(sym)
                    else:
                        fin.write(sym + " ")



    print("Finished successfully.")
    print(f"Processed files are located at {output_dir}")


if __name__ == "__main__":
    main()
