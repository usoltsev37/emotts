# ESD Preprocessing

## Data

put [esd](https://drive.google.com/file/d/1scuFwqh8s7KIYAfZW1Eu6088ZAK2SI-v/view) dataset to data/zip/Emotional Speech Dataset (ESD).zip 

## Prerequisites

* Docker: 20.10.7

## Usage

```bash
docker build --rm --tag esd ./esd_preprocessing
docker run --rm -it -v $(pwd):/emotts/repo esd
```

Processed data will be located at `data/esd/processed/mfa_outputs` (as `.TextGrid`, grouped by speaker IDs) and `data/esd/processed/mels` (as `.pkl`, grouped by speaker IDs).