#!/usr/bin/bash

speakerArray=("0001" "0002" "0003" "0004" "0005" "0006" "0007" "0008" "0009" "0010")

for speaker in ${speakerArray[@]}; do
  mkdir $speaker
done
