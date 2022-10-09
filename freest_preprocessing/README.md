# FreeST Preprocessing

## Data

put [FreeST](https://openslr.elda.org/resources/38/ST-CMDS-20170001_1-OS.tar.gz) dataset to data/zip/ST-CMDS-20170001_1-OS.tar.gz

## Prerequisites

* Docker: 20.10.7

## Usage

```bash
docker build --rm --tag freest ./freest_preprocessing
docker run --rm -it -v $(pwd):/emotts/repo freest
```

Processed data will be located at `data/processed/freest/mfa_outputs` (as `.TextGrid`, grouped by speaker IDs) and `data/processed/freest/mels` (as `.pkl`, grouped by speaker IDs).