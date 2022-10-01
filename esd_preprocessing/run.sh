#!/bin/bash
conda activate emotts
cd repo

export OUTPUT_DIR=data

[ -d "$OUTPUT_DIR/processed/esd/" ] && rm -rf $OUTPUT_DIR/processed/esd/

echo -e "\n1. Extract from zip"
python src/preprocessing/extract_esd.py --input-esd-zip $OUTPUT_DIR/zip/"Emotional Speech Dataset (ESD).zip" --output-dir $OUTPUT_DIR/processed/esd

conda deactivate
conda activate pausation

echo -e "\n2. Pausation cutting with VAD"
python src/preprocessing/pausation_cutting.py --input-dir $OUTPUT_DIR/processed/esd/english/audio --output-dir $OUTPUT_DIR/processed/esd/english/no_pause --target-sr 48000 --audio-ext wav

conda deactivate
conda activate emotts

echo -e "\n3. Resampling"
python src/preprocessing/resampling.py --input-dir $OUTPUT_DIR/processed/esd/english/no_pause --output-dir $OUTPUT_DIR/processed/esd/english/resampled --resample-rate 22050 --audio-ext wav

echo -e "\n4. Audio to Mel"
python src/preprocessing/wav_to_mel.py --input-dir $OUTPUT_DIR/processed/esd/english/resampled --output-dir $OUTPUT_DIR/processed/esd/english/mels  --audio-ext wav

echo -e "\n5. Text normalization"
python src/preprocessing/text_normalization.py --input-dir $OUTPUT_DIR/processed/esd/english/text --output-dir $OUTPUT_DIR/processed/esd/english/mfa_inputs

echo -e "\n6. MFA Alignment setup"

# download a pretrained english acoustic model, and english lexicon
mkdir -p models
wget -q --show-progress https://github.com/MontrealCorpusTools/mfa-models/raw/main/acoustic/english.zip -P models
wget -q --show-progress http://www.openslr.org/resources/11/librispeech-lexicon.txt -P models

conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib  # link to libopenblas
conda deactivate
conda activate emotts

echo -e "\n7. MFA Preprocessing"
python src/preprocessing/mfa_preprocessing.py --input-dir $OUTPUT_DIR/processed/esd/english/resampled --output-dir $OUTPUT_DIR/processed/esd/english/mfa_inputs

# FINALLY, align phonemes and speech
echo -e "\n8. MFA Alignment"

mfa align -t ./temp --clean -j 4 $OUTPUT_DIR/processed/esd/english/mfa_inputs models/librispeech-lexicon.txt models/english.zip $OUTPUT_DIR/processed/esd/english/mfa_outputs
rm -rf temp

echo -e "\n9. MFA Postprocessing"
# Aggregate mels by speakers
python src/preprocessing/mfa_postprocessing.py --input-dir $OUTPUT_DIR/processed/esd/english/mels
