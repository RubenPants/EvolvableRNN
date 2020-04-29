"""
deviate_hidden.py

Deviate a genome's candidate hidden state weight and bias values and monitor the result.
"""
import argparse
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from config import Config
from environment.game import get_game
from genomes_gru.deviate_main import get_save_path, main as plot_positions, merge
from genomes_gru.persist import load_genome
from population.utils.gene_util.gru import GruNodeGene
from population.utils.genome import Genome
from population.utils.network_util.activations import sigmoid
from population.utils.network_util.feed_forward_net import make_net
from population.utils.rnn_cell_util.berkeley_gru import GRUCell
from utils.dictionary import *

COLORS = ['r', 'b', 'c', 'm', 'y']


def main(genome: Genome,
         gid: int,
         delta: float = 1e-1,
         duration: int = 60,
         mut_bias: bool = False,
         mut_hh: bool = False,
         mut_xh: bool = False,
         ):
    """Load the genome, deviate candidate hidden state's weights and bias and plot the results."""
    # Check if valid genome (contains at least one hidden GRU, first GRU is monitored)
    assert len([n for n in genome.get_used_nodes().values() if type(n) == GruNodeGene]) >= 1
    
    # Get the GRU node's ID
    gru_id = None
    for nid, n in genome.get_used_nodes().items():
        if type(n) == GruNodeGene:
            gru_id = nid
            break
    
    # Get the deviated genomes
    genome_plus = deepcopy(genome)
    genome_minus = deepcopy(genome)
    
    # Perform the requested mutations
    if mut_bias:
        genome_plus.nodes[gru_id].bias_h[2] += delta
        genome_minus.nodes[gru_id].bias_h[2] -= delta
    if mut_hh:
        genome_plus.nodes[gru_id].weight_hh[2, 0] += delta
        genome_minus.nodes[gru_id].weight_hh[2, 0] -= delta
    if mut_xh:
        genome_plus.nodes[gru_id].weight_xh_full[2, 0] += delta
        genome_plus.nodes[gru_id].update_weight_xh()
        genome_minus.nodes[gru_id].weight_xh_full[2, 0] -= delta
        genome_minus.nodes[gru_id].update_weight_xh()
    
    # Plot the positions
    name = ''
    if mut_bias: name += f"bias{delta}"
    if mut_hh: name += f"{'_' if name else ''}hh{delta}"
    if mut_xh: name += f"{'_' if name else ''}xh{delta}"
    plot_positions(
            genome=genome,
            genome_plus=genome_plus,
            genome_minus=genome_minus,
            save_name=f'hidden/{name}_trajectory',
            gid=gid,
            duration=duration,
    )
    
    # Plot the activations
    default = monitor_activation(genome=genome, gid=gid, duration=duration)
    plus = monitor_activation(genome=genome_plus, gid=gid, duration=duration)
    minus = monitor_activation(genome=genome_minus, gid=gid, duration=duration)
    states = dict()
    states['default'] = default
    states['plus'] = plus
    states['minus'] = minus
    plot_states(
            gid=genome.key,
            states=states,
            save_name=f"hidden/{name}_state",
    )
    
    # Merge the two graphs together
    merge(
            gid=genome.key,
            save_name=f"hidden/{name}",
    )


def plot_states(gid: int, states: dict, save_name: str):
    """Plot the hidden-gate's state for each time moment."""
    # Validate the input
    size = len(list(states.values())[0][0])
    for state, _ in states.values():
        assert size == len(state)
    
    # Setup
    cfg = Config()
    time = [i / cfg.game.fps for i in range(size)]
    
    # Create the graph
    ax = plt.figure(figsize=(6, 2)).gca()
    for idx, (name, state) in enumerate(states.items()):
        plt.plot(time, state[0], color=COLORS[idx], label=name)
        for t in state[1]: plt.axvline(x=t / cfg.game.fps, color=COLORS[idx], linestyle=':', linewidth=2)
    plt.grid()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Forces to use only integers
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.xlabel("Simulation time (s)")
    plt.tight_layout()
    path = get_save_path(gid=gid, save_name=save_name)
    plt.savefig(f"{path}.png")
    plt.close()


def monitor_activation(genome: Genome, gid: int, debug: bool = False, duration: int = 60):
    """Monitor the activation of the hidden state. Note: game is started again, no worries since deterministic."""
    cfg = Config()
    cfg.game.duration = duration
    cfg.update()
    
    # Check if valid genome (contains at least one hidden GRU, first GRU is monitored)
    assert len([n for n in genome.get_used_nodes().values() if type(n) == GruNodeGene]) >= 1
    
    # Get the game
    game = get_game(i=gid, cfg=cfg, noise=False)
    state = game.reset()[D_SENSOR_LIST]
    step_num = 0
    
    # Create the network
    net = make_net(genome=genome,
                   genome_config=cfg.genome,
                   batch_size=1,
                   initial_read=state,
                   )
    
    # Containers to monitor
    Ht = []
    target_found = []
    score = 0
    
    # Initialize the containers
    Ht.append(get_state(gru=net.rnn_array[0], x=np.asarray([state])))
    if debug:
        print(f"Step: {step_num}")
        print(f"\t> Hidden state: {round(Ht[-1], 5)}")
    
    # Start monitoring
    while True:
        # Check if maximum iterations is reached
        if step_num == duration * cfg.game.fps: break
        
        # Determine the actions made by the agent for each of the states
        action = net(np.asarray([state]))
        
        # Check if each game received an action
        assert len(action) == 1
        
        # Proceed the game with one step, based on the predicted action
        obs = game.step(l=action[0][0], r=action[0][1])
        finished = obs[D_DONE]
        
        # Update the score-count
        if game.score > score:
            target_found.append(step_num)
            score = game.score
        
        # Update the candidate's current state
        state = obs[D_SENSOR_LIST]
        
        # Stop if agent reached target in all the games
        if finished: break
        step_num += 1
        
        # Update the containers
        Ht.append(get_state(gru=net.rnn_array[0], x=np.asarray([state])))
        if debug:
            print(f"Step: {step_num}")
            print(f"\t> Hidden gate state: {round(Ht[-1], 5)}")
    return Ht, target_found


def get_state(gru: GRUCell, x):
    W_xh = np.matmul(x, gru.weight_xh.transpose())
    W_hh = np.matmul(gru.hx, gru.weight_hh.transpose())
    R_t = sigmoid(W_xh[:, 0:1] + W_hh[:, 0:1] + gru.bias[0:1])
    Z_t = sigmoid(W_xh[:, 1:2] + W_hh[:, 1:2] + gru.bias[1:2])
    return ((1 - Z_t) * np.tanh(W_xh[:, 2:3] + R_t * W_hh[:, 2:3] + gru.bias[2:3]) + Z_t * gru.hx)[0, 0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--delta', type=float, default=.1)  # Deviation
    parser.add_argument('--duration', type=int, default=30)  # Simulation duration
    parser.add_argument('--gid', type=int, default=30001)  # First evaluation game of experiment3
    parser.add_argument('--mut_bias', type=int, default=1)  # Mutate the bias component of the reset gate
    parser.add_argument('--mut_hh', type=int, default=1)  # Mutate the hidden-hidden weight of the reset gate
    parser.add_argument('--mut_xh', type=int, default=1)  # Mutate the input-hidden weight of the reset gate
    parser.add_argument('--name', type=str, default='genome1')
    parser.add_argument('--show', type=int, default=1)
    args = parser.parse_args()
    
    # Go back to root
    os.chdir("..")
    
    # Load in the genome
    g = load_genome(
            g_name=args.name,
    )
    
    # Execute the process
    main(
            genome=g,
            gid=args.gid,
            delta=args.delta,
            duration=args.duration,
            mut_bias=bool(args.mut_bias),
            mut_hh=bool(args.mut_hh),
            mut_xh=bool(args.mut_xh),
    )
