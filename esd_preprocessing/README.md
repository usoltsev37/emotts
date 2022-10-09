# ESD Preprocessing

## Data

put [esd](https://drive.google.com/file/d/1scuFwqh8s7KIYAfZW1Eu6088ZAK2SI-v/view) dataset to data/zip/Emotional Speech Dataset (ESD).zip 

## Prerequisites

* Docker: 20.10.7

## Usage


```bash
docker build --rm --tag esd ./esd_preprocessing 
docker run --env language=english --rm -it -v $(pwd):/emotts/repo esd #for english 
docker run --env language=chinese --rm -it -v $(pwd):/emotts/repo esd  #for chinese 
```
## default language - english, for change on chinese, change in run.sh env variable - LANG

Processed data will be located at `data/processed/esd/[lang]/mfa_outputs` (as `.TextGrid`, grouped by speaker IDs) and `data/processed/esd/[lang]/mels` (as `.pkl`, grouped by speaker IDs).