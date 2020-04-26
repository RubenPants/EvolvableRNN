# EvolvableRNN
Introducing GRU components in NEAT to enhance interpretation of its a robot's surroundings.


## Main Idea

Update a GRU his weights based on its current value, the value fed into the GRU and the result it obtained (i.e. difference in distance over the step)

* Focus on the no-bearing task!
    * I have the feeling that the other task is somewhat solved, where the no-bearing task definitely is not
* Possible to update the GRU-weights (in NEAT) with the help of local (plastic) learning rules? (adaptively learn during its lifetime to improve weight updates)
* Compare :
    * NEAT
    * NEAT-GRU
    * NEAT-RNN
    * Mutations on NEAT-GRU that lack some of the gates



## TODO

* All games are trained with the wrong axis.. (14 instead of 20, this has effects on the distance normalisation..)
    Make game-config's layout fixed for each maze (i.e. axis, but not duration)

* Fixed training-set spawn-order for experiment3

* Enlarge spawning range of targets in experiment 3, being targets within the maze spaced at least 1meter from the boundaries and the player's position

* Do only 20 versions for experiment3

* GRU is still not able to find target in all of the cases (for example when target further than training distance..)

* Analyze model complexity across the populations (proof that GRU doesn't need a complex genome to be able to solve the problem)

* Evaluate on OpenAI's gym environment (e.g. MountainCarContinuous-v0) to compare on another task?

* Research: How long does the GRU remember?

* Monitor GRU; small variations in weights, how does path of genome change?

* Experiment: Distance via ping, update every 5 frames (0.5 seconds)

* Do more in-depth research on monitored behaviour of genomes

* Check: "Topology completely dictates the search space"



## Potential titles

* *Distance Aware Gated Recurrent Units for Target Finding*
    * Does not say a thing about NEAT
    * "foot target findings" sounds so cheap
* *Neuro-evolved Gated Recurrent Units for distance-based target finding*
    * "target finding" sounds so cheap
* *An empirical study on the effectiveness of recurrent units as memory components in evolved neural networks*



## Thesis layout

1) Comparison between NEAT, NEAT-GRU, and NEAT-SRU on experiment 1-3
2) Focus on NEAT-GRU (experiment4+5)
    * Single node networks suffice, follows from NEAT's minimality assurance (minimal viable solution shall be found)
3) Compare NEAT-GRU configurations
    * What is the importance of the gates?
    * Does the GRU-unit (single-unit networks) always converge (this reasoning follows from the linearity in the heatmaps)
    * What if parent-selection=0? Always convergence when only following the elite?
