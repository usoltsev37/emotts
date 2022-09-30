# VCTK Preprocessing

## Data

put [vckt](https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip) dataset to data/zip/vctk.zip 

## Prerequisites

* Docker: 20.10.7

## Usage

```bash
docker build --rm --tag emotts ./vctk_preprocessing
docker run --rm -it -v $(pwd):/emotts/repo emotts
```

Processed data will be located at `data/processed/mfa_outputs` (as `.TextGrid`, grouped by speaker IDs) and `data/processed/mels` (as `.pkl`, grouped by speaker IDs).