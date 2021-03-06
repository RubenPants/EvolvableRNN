#!/bin/bash

# Default parameters
unused_cpu=2;  # Number of CPU threads not used  [int]
evaluate_gen=1;  # Evaluate the generations  [bint]
evaluate_pop=0;  # Combine population evaluations  [bint]
evaluate_train=0;  # Compare the training fitness over the populations [bint]
hops=10;  # Jumps between evaluations [int]
max_gen=100;  # Maximum generation for evaluation  [int]
max_v=50;  # Maximum version for evaluation  [int]

# Run the program
for experiment in {8,}
do
#  python3 evaluate_populations.py --folder_pop=NEAT --evaluate_gen=$evaluate_gen --evaluate_pop=$evaluate_pop --evaluate_training=$evaluate_train --experiment=$experiment --max_gen=$max_gen --max_v=$max_v --unused_cpu=$unused_cpu;
#  python3 evaluate_populations.py --folder_pop=NEAT-GRU --evaluate_gen=$evaluate_gen --evaluate_pop=$evaluate_pop --evaluate_training=$evaluate_train --experiment=$experiment --max_gen=$max_gen --max_v=$max_v --unused_cpu=$unused_cpu;
#  python3 evaluate_populations.py --folder_pop=NEAT-LSTM --evaluate_gen=$evaluate_gen --evaluate_pop=$evaluate_pop --evaluate_training=$evaluate_train --experiment=$experiment --max_gen=$max_gen --max_v=$max_v --unused_cpu=$unused_cpu;
#  python3 evaluate_populations.py --folder_pop=NEAT-SRU --evaluate_gen=$evaluate_gen --evaluate_pop=$evaluate_pop --evaluate_training=$evaluate_train --experiment=$experiment --max_gen=$max_gen --max_v=$max_v --unused_cpu=$unused_cpu;

  # Never run NEAT-SRU-S in main since it uses NEAT-SRU's class! Go to modified branch instead
#  python3 evaluate_populations.py --folder_pop=NEAT-SRU-S --evaluate_gen=$evaluate_gen --evaluate_pop=$evaluate_pop --evaluate_training=$evaluate_train --experiment=$experiment --max_gen=$max_gen --max_v=$max_v --unused_cpu=$unused_cpu;


  # Experiment 7 specific populations
  python3 evaluate_populations.py --folder_pop=default --evaluate_gen=$evaluate_gen --experiment=$experiment --hops=$hops --max_v=$max_v --unused_cpu=$unused_cpu;
  python3 evaluate_populations.py --folder_pop=biased --evaluate_gen=$evaluate_gen --experiment=$experiment --hops=$hops --max_v=$max_v --unused_cpu=$unused_cpu;
  python3 evaluate_populations.py --folder_pop=gru_nr --evaluate_gen=$evaluate_gen --experiment=$experiment --hops=$hops --max_v=$max_v --unused_cpu=$unused_cpu;
done
