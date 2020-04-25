#!/bin/bash

# Default parameters
iter=100;  # Number of training-iterations each loop
cpu=2;  # Number of unused CPUs

# Run the program  TODO: Check if 35 is done!
for v in {35,}
do
  for i in {1..5}
  do
#    python3 experiment1.py --prob_gru=0 --prob_gru_nr=0 --prob_gru_nu=0 --prob_simple_rnn=0 --iterations=$iter --version=$v --unused_cpu=$cpu;
    python3 experiment1.py --prob_gru=0.6 --prob_gru_nr=0 --prob_gru_nu=0 --prob_simple_rnn=0 --iterations=$iter --version=$v --unused_cpu=$cpu;
#    python3 experiment1.py --prob_gru=0 --prob_gru_nr=0.6 --prob_gru_nu=0 --prob_simple_rnn=0 --iterations=$iter --version=$v --unused_cpu=$cpu;
#    python3 experiment1.py --prob_gru=0 --prob_gru_nr=0 --prob_gru_nu=0.6 --prob_simple_rnn=0 --iterations=$iter --version=$v --unused_cpu=$cpu;
#    python3 experiment1.py --prob_gru=0 --prob_gru_nr=0 --prob_gru_nu=0 --prob_simple_rnn=0.6 --iterations=$iter --version=$v --unused_cpu=$cpu;
  done
#  python3 experiment2.py --prob_gru=0 --prob_gru_nr=0 --prob_gru_nu=0 --prob_simple_rnn=0 --version=$v --unused_cpu=$cpu;
  python3 experiment2.py --prob_gru=0.6 --prob_gru_nr=0 --prob_gru_nu=0 --prob_simple_rnn=0 --version=$v --unused_cpu=$cpu;
#  python3 experiment2.py --prob_gru=0 --prob_gru_nr=0.6 --prob_gru_nu=0 --prob_simple_rnn=0 --version=$v --unused_cpu=$cpu;
#  python3 experiment2.py --prob_gru=0 --prob_gru_nr=0 --prob_gru_nu=0.6 --prob_simple_rnn=0 --version=$v --unused_cpu=$cpu;
#  python3 experiment2.py --prob_gru=0 --prob_gru_nr=0 --prob_gru_nu=0 --prob_simple_rnn=0.6 --version=$v --unused_cpu=$cpu;

#  git add .;
#  git commit -m "Evaluated experiment1&2 (version $v)";
#  git push;
done
