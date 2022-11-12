#!/usr/bin/bash

while getopts u:p:h: flag
do
    case "${flag}" in
        u) username=${OPTARG};;
        p) password=${OPTARG};;
        *) echo "Usage ./download.sh -u username -p password"
    esac
done

wget --user $username --password $password --show-progress https://data.cstr.ed.ac.uk/blizzard2013/lessac/BC2013_segmented_README.txt
wget --user $username --password $password --show-progress https://data.cstr.ed.ac.uk/blizzard2013/lessac/BC2013_segmented_v0_txt1.zip
wget --user $username --password $password --show-progress https://data.cstr.ed.ac.uk/blizzard2013/lessac/BC2013_segmented_v0_txt2.zip   
wget --user $username --password $password --show-progress https://data.cstr.ed.ac.uk/blizzard2013/lessac/BC2013_segmented_v0_wav1.zip 
wget --user $username --password $password --show-progress https://data.cstr.ed.ac.uk/blizzard2013/lessac/BC2013_segmented_v0_wav2.zip
wget --user $username --password $password --show-progress https://data.cstr.ed.ac.uk/blizzard2013/lessac/BC2013_segmented_v1_labels_EM.zip
wget --user $username --password $password --show-progress https://data.cstr.ed.ac.uk/blizzard2013/lessac/BC2013_segmented_v1_transcripts_selection.zip 
wget --user $username --password $password --show-progress https://data.cstr.ed.ac.uk/blizzard2013/lessac/BC2013_segmented_v1_txt_selection.zip  
wget --user $username --password $password --show-progress https://data.cstr.ed.ac.uk/blizzard2013/lessac/BC2013_segmented_v1_wav_selection.zip  
wget --user $username --password $password --show-progress https://data.cstr.ed.ac.uk/blizzard2013/lessac/BlackBeauty.zip
wget --user $username --password $password --show-progress https://data.cstr.ed.ac.uk/blizzard2013/lessac/Lessac_Blizzard2013_CatherineByers_train.tar.bz2
wget --user $username --password $password --show-progress https://data.cstr.ed.ac.uk/blizzard2013/lessac/README_for_Lessac_Blizzard2013_CatherineByers_train
wget --user $username --password $password --show-progress https://data.cstr.ed.ac.uk/blizzard2013/lessac/mansfield1.zip
wget --user $username --password $password --show-progress https://data.cstr.ed.ac.uk/blizzard2013/lessac/mansfield2.zip  
wget --user $username --password $password --show-progress https://data.cstr.ed.ac.uk/blizzard2013/lessac/mansfield3.zip      
wget --user $username --password $password --show-progress https://data.cstr.ed.ac.uk/blizzard2013/lessac/pride_and_prejudice1.zip 
wget --user $username --password $password --show-progress https://data.cstr.ed.ac.uk/blizzard2013/lessac/pride_and_prejudice2.zip  
wget --user $username --password $password --show-progress https://data.cstr.ed.ac.uk/blizzard2013/lessac/pride_and_prejudice3.zip
wget --user $username --password $password --show-progress https://data.cstr.ed.ac.uk/blizzard2013/lessac/training_inventory.xls
