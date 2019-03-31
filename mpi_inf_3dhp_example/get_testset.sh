#!/bin/bash
# Script to fetch and unzip the test set  
echo "Reading configuration from ./config....." >&2
source ./conf.ig
if [[ $ready_to_download -eq 0 ]]; then
  echo "Please read the documentation and edit the config file accordingly." >&2
  exit 1
fi
source_path="http://gvv.mpi-inf.mpg.de/3dhp-dataset"
if [ ! -f "./mpi_inf_3dhp_test_set.zip" ]; then
  wget "$source_path/mpi_inf_3dhp_test_set.zip"  
fi
if [ -f "./mpi_inf_3dhp_test_set.zip" ]; then
  if [ ! -d "$destination" ]; then
      mkdir "$destination"
  fi
  unzip "./mpi_inf_3dhp_test_set.zip" -d "$destination/mpi_inf_3dhp_test_set"
  rm "./mpi_inf_3dhp_test_set.zip"
fi
