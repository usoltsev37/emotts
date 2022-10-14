#!/usr/bin/bash

speakerArray=("0001" "0002" "0003" "0004" "0005" "0006" "0007" "0008" "0009" "0010")

audioArray=("000351" "000701" "000002" "001051" "001401")
emmoArray=("angry.pkl" "happy.pkl" "neutral.pkl" "sad.pkl" "surprise.pkl")
for speaker in ${speakerArray[@]}; do
  cd $speaker
  for i in "${!audioArray[@]}"; do 
      cp ../../../data/freest/mels/$speaker/"$speaker"_"${audioArray[$i]}.pkl" .
      mv "$speaker"_"${audioArray[$i]}.pkl" ${emmoArray[$i]}
  done
  cd ..

done

