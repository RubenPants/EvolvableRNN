#!/bin/bash

# Default parameters
iter=100;  # Number of training-iterations each loop
cpu=1;  # Number of unused CPUs

# Run the program
for v in {11..15}
  do
  for i in {1..5}
  do
    python3 experiment3.py --prob_gru=0 --prob_gru_nr=0 --prob_gru_nu=0 --prob_simple_rnn=0 --iterations=$iter --version=$v --unused_cpu=$cpu;
    python3 experiment3.py --prob_gru=0.6 --prob_gru_nr=0 --prob_gru_nu=0 --prob_simple_rnn=0 --iterations=$iter --version=$v --unused_cpu=$cpu;
#    python3 experiment3.py --prob_gru=0 --prob_gru_nr=0.6 --prob_gru_nu=0 --prob_simple_rnn=0 --iterations=$iter --version=$v --unused_cpu=$cpu;
#    python3 experiment3.py --prob_gru=0 --prob_gru_nr=0 --prob_gru_nu=0.6 --prob_simple_rnn=0 --iterations=$iter --version=$v --unused_cpu=$cpu;
    python3 experiment3.py --prob_gru=0 --prob_gru_nr=0 --prob_gru_nu=0 --prob_simple_rnn=0.6 --iterations=$iter --version=$v --unused_cpu=$cpu;
  done

  git add .;
  git commit -m "Ran experiment3 for version $v";
  git push;
done
