#!/bin/bash
conda activate emotts
cd repo/data

export OUTPUT_DIR=data

# Unzip dataset and reorganize folders
unzip -q zip/vctk.zip txt/* wav48_silence_trimmed/*

mkdir -p raw/text
mv txt/* raw/text
mkdir -p raw/audio
mv wav48_silence_trimmed/* raw/audio
rm -rf txt wav48_silence_trimmed

cd ..

echo -e "\n1. Selecting only one mic per speaker"
python src/preprocessing/preprocessing.py --input-dir $OUTPUT_DIR/raw/audio --output-dir $OUTPUT_DIR/processed/vctk/audio_single_mic --audio-ext flac

conda deactivate
conda activate pausation

echo -e "\n2. Pausation cutting with VAD"
python src/preprocessing/pausation_cutting.py --input-dir $OUTPUT_DIR/processed/vctk/audio_single_mic --output-dir $OUTPUT_DIR/processed/vctk/no_pause --target-sr 48000

conda deactivate
conda activate emotts

echo -e "\n3. Resampling"
python src/preprocessing/resampling.py --input-dir $OUTPUT_DIR/processed/vctk/no_pause --output-dir $OUTPUT_DIR/processed/vctk/resampled --resample-rate 22050

echo -e "\n4. Audio to Mel"
python src/preprocessing/wav_to_mel.py --input-dir $OUTPUT_DIR/processed/vctk/resampled --output-dir $OUTPUT_DIR/processed/vctk/mels

echo -e "\n5. Text normalization"
python src/preprocessing/text_normalization.py --input-dir $OUTPUT_DIR/raw/text --output-dir $OUTPUT_DIR/processed/vctk/mfa_inputs

echo -e "\n6. MFA Alignment setup"

# download a pretrained english acoustic model, and english lexicon
mkdir -p models
wget -q --show-progress https://github.com/MontrealCorpusTools/mfa-models/raw/main/acoustic/english.zip -P models
wget -q --show-progress http://www.openslr.org/resources/11/librispeech-lexicon.txt -P models

conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib  # link to libopenblas
conda deactivate
conda activate emotts

echo -e "\n7. MFA Preprocessing"
python src/preprocessing/mfa_preprocessing.py --input-dir $OUTPUT_DIR/processed/vctk/resampled --output-dir $OUTPUT_DIR/processed/vctk/mfa_inputs

# FINALLY, align phonemes and speech
echo -e "\n8. MFA Alignment"
echo $OUTPUT_DIR

mfa align -t ./temp --clean -j 4 $OUTPUT_DIR/processed/vctk/mfa_inputs models/librispeech-lexicon.txt models/english.zip $OUTPUT_DIR/processed/vctk/mfa_outputs
rm -rf temp

echo -e "\n9. MFA Postprocessing"
# Aggregate mels by speakers
python src/preprocessing/mfa_postprocessing.py --input-dir $OUTPUT_DIR/processed/vctk/mels
