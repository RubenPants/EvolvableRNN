"""
trace.py

Trace the path the genome walks under some slight deviations in its configuration.
"""
import argparse
import os
from math import cos, sin

import matplotlib.pyplot as plt
import numpy as np

from config import Config
from environment.game import Game, get_game
from genomes_gru.persist import load_genome
from population.utils.gene_util.gru import GruNodeGene
from population.utils.genome import Genome
from population.utils.network_util.feed_forward_net import make_net
from utils.dictionary import *

COLORS = ['r', 'b', 'c', 'm', 'y']


def get_positions(genome: Genome, gid: int, debug: bool = False, duration: int = 60):
    """Get the position of the genome at every 0.5 seconds during the given simulation."""
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
    position = []
    target_found = []
    score = 0
    
    # Initialize the containers
    position.append(game.player.pos.get_tuple())
    if debug:
        print(f"Step: {step_num}")
        print(f"\t> Position: {(round(position[-1][0], 2), round(position[-1][1], 2))!r}")
        print(f"\t> Score:    {score!r}")
    
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
        position.append(game.player.pos.get_tuple())
        if debug:
            print(f"Step: {step_num}")
            print(f"\t> Position: {(round(position[-1][0], 2), round(position[-1][1], 2))!r}")
            print(f"\t> Score:    {score!r}")
    return position, game


def visualize_positions(positions: dict,
                        game: Game,
                        annotate_time: bool = True,
                        save_path: str = None,
                        show: bool = False,
                        ):
    """Visualize the list of positions."""
    # Initialize the figure dimensions
    plt.figure(figsize=(6, 6)).gca()
    x_min, x_max = game.x_axis / 2, game.x_axis / 2
    y_min, y_max = game.y_axis / 2, game.y_axis / 2
    
    # Create the blueprint - Only visualize targets found by the agent
    for i in range(min(game.score + 1, len(game.spawn_function.locations))):
        t = game.spawn_function.locations[i]
        
        # Plot the targets
        plt.plot(t[0], t[1], 'go')
        plt.annotate(str(i + 1), xy=(t[0] + 0.1, t[1] + 0.1))
        
        # Add green dotted circle around targets
        c = plt.Circle((t[0], t[1]), 0.5, color='g', linestyle=':', linewidth=2, fill=False)
        plt.gca().add_artist(c)
        
        # Update the boundaries
        x_min, x_max = min(x_min, t[0]), max(x_max, t[0])
        y_min, y_max = min(y_min, t[1]), max(y_max, t[1])
    
    # Show the player's initial direction
    x = game.player.init_pos[0]
    y = game.player.init_pos[1]
    dx = cos(game.player.noisy_init_angle)
    dy = sin(game.player.noisy_init_angle)
    plt.arrow(x, y, dx, dy, color='r', head_width=0.1, length_includes_head=True)
    
    # Plot the player's path
    for pos_idx, (pos_name, pos) in enumerate(positions.items()):
        c = COLORS[pos_idx]
        x_pos, y_pos = zip(*pos)
        x_min, x_max = min(x_min, min(x_pos)), max(x_max, max(x_pos))
        y_min, y_max = min(y_min, min(y_pos)), max(y_max, max(y_pos))
        for p in range(0, len(x_pos), 5):
            if p == 0:
                plt.plot(x_pos[p], y_pos[p], f'{c}o', markersize=2, label=pos_name)
            else:
                plt.plot(x_pos[p], y_pos[p], f'{c}o', markersize=2)
            
            # Annotate every 5 seconds
            if annotate_time:
                if p % (5 * game.game_config.fps) == 0 and p > 0:
                    # offset = (x_pos[p + 1] - x_pos[p], y_pos[p + 1] - y_pos[p])
                    offset = (y_pos[p - 1] - y_pos[p], x_pos[p] - x_pos[p - 1])
                    plt.annotate(
                            str(int(p / game.game_config.fps)),
                            xy=(x_pos[p], y_pos[p]),
                            xytext=(x_pos[p] + offset[0] * 15, y_pos[p] + offset[1] * 15),
                            ha="center", va="center",
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc,rad=0."),
                    )
    
    # Constraint the plot's boundaries
    x_center = (x_max - x_min) / 2 + x_min
    y_center = (y_max - y_min) / 2 + y_min
    r = max((x_max - x_min) / 2 + 1, (y_max - y_min) / 2 + 1)
    plt.xlim(x_center - r, x_center + r)
    plt.ylim(y_center - r, y_center + r)
    
    # Add a legend to the figure and store/show
    plt.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    if show: plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--name', type=str, default='genome1')
    parser.add_argument('--gid', type=int, default=30001)  # First evaluation game of experiment3
    parser.add_argument('--show', type=bool, default=True)
    args = parser.parse_args()
    
    # Go back to root
    os.chdir("..")
    
    # Load in the genome
    chosen_genome = load_genome(
            g_name=args.name,
    )
    
    # Run the script
    positions_evaluated, game_evaluated = get_positions(
            genome=chosen_genome,
            gid=args.gid,
            duration=25,
    )
    positions_dict = dict()
    positions_dict['positions'] = positions_evaluated
    visualize_positions(
            positions=positions_dict,
            game=game_evaluated,
            annotate_time=False,
            show=args.show,
    )
