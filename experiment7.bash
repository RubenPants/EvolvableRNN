#!/bin/bash

# Default parameters
pop_name=gru_nr_connection;  # Topology-ID
iter=100;  # Number of training-iterations each loop
cpu=1;  # Number of unused CPUs
push=10;  # After how many version git should push

# Run the program
for v in {1..10}
do
  python3 experiment7.py --pop_name=$pop_name --version=$v --iterations=$iter --unused_cpu=$cpu;

  if [ $(($v%$push)) == 0 ];  # Push every 'push' populations
  then
    git add .;
    git commit -m "Evaluated experiment7 (version $v)";
    git push;
  fi
done
