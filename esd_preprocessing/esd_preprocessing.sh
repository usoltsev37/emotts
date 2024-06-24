#!bin/bash

# cp /media/public/nikita_u/vctk.zip vctk.zip
# mv ./vctk_preprocessing/data/zip/vctk.zip vctk.zip

docker build --rm --tag esd ./esd_preprocessing
# -v /media/public/nikita_u/data/:/emotts/repo/data does not work
docker run --env language=english --rm -it -v $(pwd):/emotts/repo esd
