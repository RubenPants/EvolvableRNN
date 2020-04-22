#!/bin/bash

# Default parameters
iter=100;  # Number of training-iterations each loop
cpu=1;  # Number of unused CPUs

# Run the program
for v in {5..10}
  do
  for i in {1..3}
  do
#    python3 experiment4.py --prob_gru=0 --prob_gru_nr=0 --prob_gru_nu=0 --prob_simple_rnn=0 --iterations=$iter --version=$v --unused_cpu=$cpu;
#    python3 experiment4.py --prob_gru=1 --prob_gru_nr=0 --prob_gru_nu=0 --prob_simple_rnn=0 --iterations=$iter --version=$v --unused_cpu=$cpu;
#    python3 experiment4.py --prob_gru=0 --prob_gru_nr=1 --prob_gru_nu=0 --prob_simple_rnn=0 --iterations=$iter --version=$v --unused_cpu=$cpu;
#    python3 experiment4.py --prob_gru=0 --prob_gru_nr=0 --prob_gru_nu=1 --prob_simple_rnn=0 --iterations=$iter --version=$v --unused_cpu=$cpu;
    python3 experiment4.py --prob_gru=0 --prob_gru_nr=0 --prob_gru_nu=0 --prob_simple_rnn=1 --iterations=$iter --version=$v --unused_cpu=$cpu;
  done

  git add .;
  git commit -m "Ran experiment4 for 100gen (iteration $i - version $v)";
  git push;
done
