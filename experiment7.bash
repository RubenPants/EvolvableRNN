#!/bin/bash

# Default parameters
pop_name=gru_nr;  # Topology-ID
cpu=1;  # Number of unused CPUs
push=10;  # After how many version git should push

# Run the program
for v in {1..50}
do
  python3 experiment7.py --pop_name=$pop_name --version=$v --unused_cpu=$cpu;

  if [ $(($v%$push)) == 0 ];  # Push every 'push' populations
  then
    git add .;
    git commit -m "Evaluated experiment7 (version $v)";
    git push;
  fi
done
