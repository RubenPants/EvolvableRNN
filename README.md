# EvolvableRNN
Introducing GRU components in NEAT to enhance interpretation of its a robot's surroundings.


## Feedback meeting 1st of May

* Discuss:
    * SRU is overfitting in experiment 1 --> Always wants to go to 6m distances (visualize!)
    * Observe that SRU does not have the capability of expressing the delta-distance difference
    * Main question: Why does SRU perform better than GRU? (is the distance delay not possible in SRU?)
* Research:
    * Delay of distance as actuation, would this lead to a controller finding the target? (hand-made controller), 
        If so, why isn't SRU able but GRU is to find this delay?
    * Do all the GRU solutions follow the delay pattern?
    * Symmetry of GRU --> Connections after GRU always positive? (half search space, which is a lot since exponential behavior)
        What if multiple connections?
    * Experiment 3, how does the graph change for different domains? (same simulations, just different distances) --> interesting?
* Other remarks:
    * 50 pops is enough (increase experiment3 tot 30pop though --> currently training!)
    * Create statistical analysis of experiment1, var/mean conclusions, is there a significance between populations? (ANOVA?)
        If done, do them also for experiment2
    * Mention complexity of solutions (topology ~NEAT's claim of minimal viable solution)
    * Work further on experiment 3 if enough time left, first focus on experiment 1 and 2!
    * Topology 3 (experiment6) is incapable of finding a solution
    * Perhaps interesting to show progress of probability distribution over time (GIF for experiment 2, see how distributions shift over the generations)
    * Karl (Wannes?) are up to reading through the thesis for feedback 
    * Should I cite Python, Numpy, SciPy, Cython, ...?


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

* GRU monitor needs extension; also the candidate hidden state!

* Remove delta-distance graph from monitors

* Analyze topology complexity across the populations (proof that GRU doesn't need a complex genome to be able to solve the problem)

* Evaluate on OpenAI's gym environment (e.g. MountainCarContinuous-v0) to compare on another task? --> Would deviate too much from thesis I think

* Research: How long does the GRU remember?

* Experiment with distance information in pings?

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
