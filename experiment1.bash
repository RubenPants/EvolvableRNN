#!/bin/bash

# Default parameters
iter=100;  # Number of training-iterations each loop
cpu=1;  # Number of unused CPUs
push=10;  # After how many version git should push

# Run the program
for v in {1..50}
do
  for i in {1..5}
  do
#    python3 experiment1.py --prob_gru=0 --prob_simple_rnn=0 --prob_lstm=0 --iterations=$iter --version=$v --unused_cpu=$cpu;
#    python3 experiment1.py --prob_gru=0.6 --prob_simple_rnn=0 --prob_lstm=0 --iterations=$iter --version=$v --unused_cpu=$cpu;
#    python3 experiment1.py --prob_gru=0 --prob_simple_rnn=0.6 --prob_lstm=0 --iterations=$iter --version=$v --unused_cpu=$cpu;
    python3 experiment1.py --prob_gru=0 --prob_simple_rnn=0 --prob_lstm=0.6 --iterations=$iter --version=$v --unused_cpu=$cpu;
  done
#  python3 experiment2.py --prob_gru=0 --prob_simple_rnn=0 --prob_lstm=0 --iterations=$iter --version=$v --unused_cpu=$cpu;
#  python3 experiment2.py --prob_gru=0.6 --prob_simple_rnn=0 --prob_lstm=0 --iterations=$iter --version=$v --unused_cpu=$cpu;
#  python3 experiment2.py --prob_gru=0 --prob_simple_rnn=0.6 --prob_lstm=0 --iterations=$iter --version=$v --unused_cpu=$cpu;
  python3 experiment2.py --prob_gru=0 --prob_simple_rnn=0 --prob_lstm=0.6 --iterations=$iter --version=$v --unused_cpu=$cpu;

  if [ $(($v%$push)) == 0 ];  # Push every 10 generations
  then
    git add .;
    git commit -m "Evaluated experiment1&2 (version $v)";
    git push;
  fi
done
