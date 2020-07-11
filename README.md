# EvolvableRNN
This project performs the experiments I performed for my Master's thesis: *An Empirical Investigation of Using Gated 
Recurrent Units in Evolved Robotic Controllers*. My supervisors were Prof. dr. K. Tuyls and Dr. ir. W. Meert, with daily
supervision of J. Butterworth and ir. W. Yang.

This project investigates the usage of a various catalogue of recurrent units in evolvable recurrent neural networks to
navigate though a sparse environment. More even, the capabilities of the evolved networks is analysed to interpret the
environment via a single stream of distance data. This distance data gives the actual distance between the robot and 
target at each frame, which implies that the network should be able to extract the relative direction based on this
distance information alone. The considered recurrent units are a single recurrent unit (hidden node with single 
recurrent connection), a Gated Recurrent Unit (GRU), and a Long Short-Term Memory (LSTM) cell.

The project is implemented in Python, extended with Cython to increase performance. Cython translates the simulations
into C++ to speed up the simulation process. The networks themselves are implemented using Numpy, this because only
trivial matrix operations were applied.



## Thesis Abstract
This work presents a study on the effectiveness of using gated recurrent units in 
evolved neural networks for the navigation of a mobile robot towards a static target.
It considers an environment in which only the relative distance between the robot
and its target is known. The proposed problem validates the evolved networks in
their capability to navigate the robot towards its target based on a single stream
of distance data alone. To bring this task to a successful conclusion, the network
should be able to derive the relative direction of the target based on this distance
data. This study empirically shows that the presence of feedback in the network - in
the form of recurrent connections - is necessary in the path towards a solution.

This thesis investigates the benefits a network receives from the presence of
various recurrent units. The obtained results show that such units - or the presence
of recurrency in general - form a minimal requirement for a network to lead towards
a solution. It further shows that there is a performance difference between the
types of recurrent units used, with units relying on gating mechanisms significantly
outperforming those that do not.

This study continues with an investigation in the differences between a Gated
Recurrent Unit and a trivial recurrent unit that carries only a single non-gated
recurrent connection. It shows the importance of the update gate - found in the Gated
Recurrent Unit - to maintain a long-term memory of its observations, which help
the network to derive the relative orientation of the target. A further investigation
shows that this update gate is the only gate present in the Gated Recurrent Unit
that is necessary to lead to a solution. From this observation, it is derived that the
Gated Recurrent Unit's reset gate is superfluous for this domain, which is confirmed
by a statistical analysis.



## Project Overview
To have all the desired dependencies (with the exclusion of PyTorch, only useful for running tests, see `run_tests.py`),
run the `experiments.txt` file found in root directory.

This project consists of a hand full of experiments, each containing:
* A dedicated subfolder `./population/storage/experiment[X]` in which all the trained populations are stored
* A Python script to run a single experiment `experiment[X].py` (root directory)
* A Bash file to run a batch of populations on the same experiment `experiment[X].bash` (root directory)
* Potentially also a Jupyter Notebook to analyse the results `experiment[X].ipynb` (root directory)

All supported functionality is implemented in the `main.py` file (root directory). Via this file, it is possible to 
train each of the supported populations. This file supports the following:
* Population **creation**
* **Training** of the requested population
* Creating an **overview of the training** session (visualisations)
* Create a **blueprint** of a single run, which shows a single environment with the final position of all of the 
population's candidates
* Create a **trace** of a single candidate in a given environment, which shows the trajectory of this candidate 
throughout the complete simulation
* Create a **trace of the most fit genome**, similar to the *trace* discussed in the previous bullet
* **Monitor** a single genome in a given environment, which not only shows the detailed trajectory of this genome in the
simulation, but also its internal state
* **Evaluate** the population
* Create a **visualisation of a single genome**, which shows its internal state, as well as its configuration (latter is
optional)
* **Analyse the GRU** (Gated Recurrent Unit) if present in the genome, similar to *monitor*
* Create a **live visualisation/simulation** of a single genome in a given environment

In order to create a new population with custom characteristics, manipulate the configuration files found under the 
`./configs` subdirectory. In this folder, the following configurations are found:
* `bot_config` contains all the configuration hyperparameters related to the robot; driving speed, size, ...
* `evaluation_config` contains all the configuration hyperparameters related to the evaluation process of the evolved
networks; which fitness function to use, how to combine fitness scores, ...
* `game_config` contains all the configuration hyperparameters related to the game/environment itself; how long a 
simulation lasts, frames per second, ...
* `genome_config` is the most elaborate configuration file since it contains the hyperparameters dictating the evolution
process of the genomes (representing the evolved networks), which contains; type of recurrent units used, 
hyperparameters of the evolutionary process, ...
* `population_config` contains all the hyperparameters managing the genomes within a population; parent selection, 
number of elitist genomes, ...



## Visualisations
To help with the analysis of the populations' performance, a wide variety of visualisations are implemented in this
project.


### Population Type Comparison
Visualisations comparing the performance of the different population types. The example below shows the ratio of 
successfully finished simulations for each of the population types over the full evolutionary cycle for the first 
experiment.
<p align="center">
  <img src="https://github.com/RubenPants/TransferLearningKerasDemo/blob/master/population/storage/experiment1/images/comb_finished.png"/>
</p>


### Population Analysis
Visualisations that help to analyse a population's performance. The examples shown in this section are fom the NEAT-GRU
population that was evolved for the third experiment.

#### Population Performance
Performance of the population, as represented by its elite genome, throughout the full evolutionary cycle.
<p align="center">
  <img src="https://github.com/RubenPants/TransferLearningKerasDemo/blob/master/population/storage/experiment3/NEAT-GRU/example/images/elites/gen_1000.png"/>
</p>

#### Architectural Timeline
Representation of all the architectural changes throughout the complete population timeline.
<p align="center">
  <img src="https://github.com/RubenPants/TransferLearningKerasDemo/blob/master/population/storage/experiment3/NEAT-GRU/example/images/elites/architecture_timeline.png"/>
</p>

#### Species Representatives
The genome architecture of each species' representative.
<p align="center">
  <img src="https://github.com/RubenPants/TransferLearningKerasDemo/blob/master/population/storage/experiment3/NEAT-GRU/example/images/species/representatives_gen1000.png"/>
</p>

#### Species Distance
The genome distance between each two species' representatives.
<p align="center">
  <img src="https://github.com/RubenPants/TransferLearningKerasDemo/blob/master/population/storage/experiment3/NEAT-GRU/example/images/species/distances_gen_1000.png"/>
</p>

#### Game Blueprint
A blueprint of all the final positions of the population's genome. Note that this is only useful when the simulation
ends when the target is found, hence the example given is that of the NEAT-GRU population on the first experiment.
<p align="center">
  <img src="https://github.com/RubenPants/TransferLearningKerasDemo/blob/master/population/storage/experiment1/NEAT-GRU/example/images/games/game10001/blueprint_gen00500.png"/>
</p>

#### Trace
An extension on the blueprint plot shown above, where also the traces of each respective genome are shown. This plot 
also shows the NEAT-GRU population of the first experiment. Note that it is possible to show only the best X genomes, 
which results in a cleaner graph.
<p align="center">
  <img src="https://github.com/RubenPants/TransferLearningKerasDemo/blob/master/population/storage/experiment1/NEAT-GRU/example/images/games/game10001/trace_gen00500.png"/>
</p>


### Genome Analysis
Visualisations that help to analyse a single genome. The examples shown in this section are fom the NEAT-GRU population
that was evolved for the third experiment.

#### Architecture
Detailed graph of the genome's architecture, which shows both the configurations of all of its connections as well as 
its internal nodes.
<p align="center">
  <img src="https://github.com/RubenPants/TransferLearningKerasDemo/blob/master/population/storage/experiment3/NEAT-GRU/example/images/architecture_debug/genome_482171.png"/>
</p>

#### Monitor
A plot that shows how a genome's internal changes during a single simulation. Note that this plot assumes that the 
genome consists of only a single hidden node.
<p align="center">
  <img src="https://github.com/RubenPants/TransferLearningKerasDemo/blob/master/population/storage/experiment3/NEAT-GRU/example/images/monitor/482171/30001.png"/>
</p>

#### Trace
Creates the trace of the genome on a specific environment. The trace shown is that of the NEAT-GRU population its elite
genome of the first experiment.
<p align="center">
  <img src="https://github.com/RubenPants/TransferLearningKerasDemo/blob/master/population/storage/experiment1/NEAT-GRU/example/images/games/game10001/trace_239823_gen00500.png"/>
</p>

#### Live
A live visualisation which shows how the genome behaves in a certain environment. During this live visualisation, it is
possible to manually set targets to the environment, this to better analyse the network's capabilities.
<p align="center">
  <img src="https://github.com/RubenPants/TransferLearningKerasDemo/blob/master/population/storage/experiment3/NEAT-GRU/example/images/live_example.gif"/>
</p>


### Game
Visualisations showing the configuration of the game, which are the positions of the potential targets, as well as the
initial position as well as the starting direction of the robot.
<p align="center">
  <img src="https://github.com/RubenPants/TransferLearningKerasDemo/blob/master/environmentvisualisations/30001.png"/>
</p>
