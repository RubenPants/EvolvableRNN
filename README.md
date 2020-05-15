# EvolvableRNN
Introducing GRU components in NEAT to enhance interpretation of its a robot's surroundings.


## Feedback meeting 1st of May

* Discuss:
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
    * Mention complexity of solutions (topology ~NEAT's claim of minimal viable solution)
    * Perhaps interesting to show progress of probability distribution over time (GIF for experiment 2, see how distributions shift over the generations)



## Questions
* SS divided by four --> no improvement! Why? --> solution space is also divided by four!



## TODO

* Heatmap of the solution space for topology3 in experiment6?

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
