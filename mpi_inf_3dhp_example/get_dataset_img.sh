#!/bin/bash
# Script to fetch and unzip the available data.  
# It uses config as the configuration file (duh!)
echo "Reading configuration from ./config....." >&2
source ./conf.ig
if [[ $ready_to_download -eq 0 ]]; then
  echo "Please read the documentation and edit the config file accordingly." >&2
  exit 1
fi
seq_sets=('imageSequence')
img_filepath='imageFrames'
source_path="http://gvv.mpi-inf.mpg.de/3dhp-dataset"
echo "operate destination set to $destination " >&2

for subject in ${subjects[@]}; do 
  for seq in 1 2; do 
      for im in "${seq_sets[@]}"; do 
          if [ ! -d "$destination/S$subject/Seq$seq/$img_filepath" ]; then
              mkdir "$destination/S$subject/Seq$seq/$img_filepath"
          fi
          for video_file in ` ls $destination/S$subject/Seq$seq/$im `; do
              if [ ! -d "$destination/S$subject/Seq$seq/$img_filepath/${video_file%.*}" ]; then
                  mkdir "$destination/S$subject/Seq$seq/$img_filepath/${video_file%.*}"
              fi
              echo "... S$subject/Seq$seq/$img_filepath/${video_file%.*} ... " >&2
              ffmpeg -i "$destination/S$subject/Seq$seq/$im/$video_file" -qscale:v 1 "$destination/S$subject/Seq$seq/$img_filepath/${video_file%.*}/frame_%06d.jpg"
          done
      done
  done #Seq
done #Subject