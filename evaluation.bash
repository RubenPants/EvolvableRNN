#!/bin/bash

# Default parameters
evaluate_gen=1;  # Evaluate the generations
evaluate_pop=1;  # Combine population evaluations
max_v=50;  # Maximum version for evaluation

# Run the program
for experiment in {1,2}
do
  python3 evaluate_populations.py --evaluate_gen=$evaluate_gen --evaluate_pop=$evaluate_pop --experiment=$experiment --folder_pop=NEAT --max_v=$max_v;
  python3 evaluate_populations.py --evaluate_gen=$evaluate_gen --evaluate_pop=$evaluate_pop --experiment=$experiment --folder_pop=NEAT-GRU --max_v=$max_v;
  python3 evaluate_populations.py --evaluate_gen=$evaluate_gen --evaluate_pop=$evaluate_pop --experiment=$experiment --folder_pop=NEAT-SRU --max_v=$max_v;

#  git add .;
#  git commit -m "Evaluated populations";
#  git push;
done
