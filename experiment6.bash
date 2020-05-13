#!/bin/bash

# Default parameters
batch=300000;  # Number of training-iterations each loop
cpu=1;  # Number of unused CPUs

# Run the program
for t in {22,}
do
  python3 experiment6_2.py --topology_id=$t --batch=$batch --unused_cpu=$cpu;

  git add .;
  git commit -m "Evaluated $batch genomes in experiment6 for topology $t";
  git push;
done
