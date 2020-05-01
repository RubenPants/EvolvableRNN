#!/bin/bash

# Default parameters
iter=100;  # Number of training-iterations each loop
cpu=1;  # Number of unused CPUs
push=1;  # After how many version git should push

# Run the program
for v in {11..15}
do
  for i in {1..5}
  do
    python3 experiment3.py --prob_gru=0 --prob_sru=0 --prob_lstm=0 --iterations=$iter --version=$v --unused_cpu=$cpu;
    python3 experiment3.py --prob_gru=0.6 --prob_sru=0 --prob_lstm=0 --iterations=$iter --version=$v --unused_cpu=$cpu;
    python3 experiment3.py --prob_gru=0 --prob_sru=0.6 --prob_lstm=0 --iterations=$iter --version=$v --unused_cpu=$cpu;
#    python3 experiment3.py --prob_gru=0 --prob_sru=0 --prob_lstm=0.6 --iterations=$iter --version=$v --unused_cpu=$cpu;
  done

  if [ $(($v%$push)) == 0 ];  # Push every 'push' generations
  then
    git add .;
    git commit -m "Evaluated experiment3 (version $v)";
    git push;
  fi
done
