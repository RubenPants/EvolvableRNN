#!/bin/bash

# Default parameters
unused_cpu=0;  # Number of CPU threads not used  [int]
evaluate_gen=0;  # Evaluate the generations  [bint]
evaluate_pop=1;  # Combine population evaluations  [bint]
evaluate_train=0;  # Compare the training fitness over the populations [bint]
max_v=50;  # Maximum version for evaluation  [int]

# Run the program
for experiment in {1,2,}
do
  python3 evaluate_populations.py --evaluate_gen=$evaluate_gen --evaluate_pop=$evaluate_pop --evaluate_training=$evaluate_train --experiment=$experiment --folder_pop=NEAT --max_v=$max_v --unused_cpu=$unused_cpu;
  python3 evaluate_populations.py --evaluate_gen=$evaluate_gen --evaluate_pop=$evaluate_pop --evaluate_training=$evaluate_train --experiment=$experiment --folder_pop=NEAT-GRU --max_v=$max_v --unused_cpu=$unused_cpu;
#  python3 evaluate_populations.py --evaluate_gen=$evaluate_gen --evaluate_pop=$evaluate_pop --evaluate_training=$evaluate_train --experiment=$experiment --folder_pop=NEAT-LSTM --max_v=$max_v --unused_cpu=$unused_cpu;
  python3 evaluate_populations.py --evaluate_gen=$evaluate_gen --evaluate_pop=$evaluate_pop --evaluate_training=$evaluate_train --experiment=$experiment --folder_pop=NEAT-SRU --max_v=$max_v --unused_cpu=$unused_cpu;
#  python3 evaluate_populations.py --evaluate_gen=$evaluate_gen --evaluate_pop=$evaluate_pop --evaluate_training=$evaluate_train --experiment=$experiment --folder_pop=NEAT-SRU-S --max_v=$max_v --unused_cpu=$unused_cpu;
done
