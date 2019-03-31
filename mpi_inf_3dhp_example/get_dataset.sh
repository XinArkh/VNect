#!/bin/bash
# Script to fetch and unzip the available data.  
# It uses config as the configuration file (duh!)
echo "Reading configuration from ./config....." >&2
source ./conf.ig
if [[ $ready_to_download -eq 0 ]]; then
  echo "Please read the documentation and edit the config file accordingly." >&2
  exit 1
fi
if [ ! -d "$destination" ]; then
    mkdir "$destination"
fi
seq_sets=('imageSequence')
if [[ $download_masks -eq 1 ]]; then
   seq_sets=('imageSequence' 'FGmasks' 'ChairMasks')
fi
source_path="http://gvv.mpi-inf.mpg.de/3dhp-dataset"
echo "Download destination set to $destination " >&2

for subject in ${subjects[@]}; do 
  if [ ! -d "$destination/S$subject" ]; then
      mkdir "$destination/S$subject"
  fi
  for seq in 1 2; do 
      if [ ! -d "$destination/S$subject/Seq$seq" ]; then
          mkdir "$destination/S$subject/Seq$seq"
      fi
      echo "Downloading Subject $subject, Sequence $seq ... " >&2
      wget "$source_path/S$subject/Seq$seq/annot.mat"  
      mv "./annot.mat" "$destination/S$subject/Seq$seq/annot.mat"
      wget "$source_path/S$subject/Seq$seq/camera.calibration"  
      mv "./camera.calibration" "$destination/S$subject/Seq$seq/camera.calibration"

    #Download the videos first, and then unzip them
    for im in "${seq_sets[@]}"; do 
      echo "... $im ... " >&2
      if [ ! -d "$destination/S$subject/Seq$seq/$im" ]; then
          mkdir "$destination/S$subject/Seq$seq/$im"
      fi
      #One could check here if the downloaded videos are available unzipped, but whatever, download if
      #zip is missing
      if [ ! -f "$destination/S$subject/Seq$seq/$im/vnect_cameras.zip" ]; then
          wget "$source_path/S$subject/Seq$seq/$im/vnect_cameras.zip"  
          mv "./vnect_cameras.zip" "$destination/S$subject/Seq$seq/$im/vnect_cameras.zip"
      fi
      if [ $download_extra_wall_cameras -ne 0 ]; then
          if [ ! -f "$destination/S$subject/Seq$seq/$im/other_angled_cameras.zip" ]; then
              wget "$source_path/S$subject/Seq$seq/$im/other_angled_cameras.zip"  
              mv "./other_angled_cameras.zip" "$destination/S$subject/Seq$seq/$im/other_angled_cameras.zip"
          fi
      fi
      if [ $download_extra_ceiling_cameras -ne 0 ]; then
          if [ ! -f "$destination/S$subject/Seq$seq/$im/ceiling_cameras.zip" ]; then
              wget "$source_path/S$subject/Seq$seq/$im/ceiling_cameras.zip"  
              mv "./ceiling_cameras.zip" "$destination/S$subject/Seq$seq/$im/ceiling_cameras.zip"
          fi
      fi
    done
    #Unzip the videos now
    for im in "${seq_sets[@]}"; do 
      echo "... $im ... " >&2
      if [ ! -d "$destination/S$subject/Seq$seq/$im" ]; then
          mkdir "$destination/S$subject/Seq$seq/$im"
      fi
      if [ -f "$destination/S$subject/Seq$seq/$im/vnect_cameras.zip" ]; then
          unzip -j "$destination/S$subject/Seq$seq/$im/vnect_cameras.zip" -d "$destination/S$subject/Seq$seq/$im/"
          rm "$destination/S$subject/Seq$seq/$im/vnect_cameras.zip"
      fi
      if [ -f "$destination/S$subject/Seq$seq/$im/other_angled_cameras.zip" ]; then
          unzip -j "$destination/S$subject/Seq$seq/$im/other_angled_cameras.zip" -d "$destination/S$subject/Seq$seq/$im/"
          rm "$destination/S$subject/Seq$seq/$im/other_angled_cameras.zip"
      fi
      if [ -f "$destination/S$subject/Seq$seq/$im/ceiling_cameras.zip" ]; then
          unzip -j "$destination/S$subject/Seq$seq/$im/ceiling_cameras.zip" -d "$destination/S$subject/Seq$seq/$im/"
          rm "$destination/S$subject/Seq$seq/$im/ceiling_cameras.zip"
      fi
    done

  done #Seq
done #Subject
