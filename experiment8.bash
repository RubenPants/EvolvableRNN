#!/bin/bash

# Default parameters
cpu=1;  # Number of unused CPUs
push=10;  # After how many version git should push

# Run the program
for v in {21..30}
do
  for pop_name in default gru_nr biased
  do
    python3 experiment8.py --pop_name=$pop_name --version=$v --unused_cpu=$cpu;
  done

  if [ $(($v%$push)) == 0 ];  # Push every 'push' populations
  then
    git add .;
    git commit -m "Evaluated experiment7 (version $v)";
    git push;
  fi
done
