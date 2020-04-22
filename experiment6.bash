#!/bin/bash

# Default parameters
topology=1;  # ID of the used topology
batch=100000;  # Number of training-iterations each loop
cpu=0;  # Number of unused CPUs

# Run the program
while true
do
#  python3 experiment6.py --topology_id=$topology --batch=$batch --unused_cpu=$cpu;
  python3 experiment6_2.py --topology_id=$topology --batch=$batch --unused_cpu=$cpu;

#  git add .;
#  git commit -m "Evaluated $batch genomes in experiment6";
#  git push;
done
