"""
monitor_genome_single_sru_original.py  TODO: Make this file back the default! (remove "_original")

Monitor a single genome during its run on a single game. This monitoring focuses on the SRU-cell that must be present
in the genome.
"""
import argparse
import os
from math import cos, sin

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from config import Config
from configs.game_config import GameConfig
from environment.game import Game, get_game
from main import get_game_ids
from population.population import Population
from population.utils.gene_util.fixed_rnn import FixedRnnNodeGene
from population.utils.gene_util.simple_rnn import SimpleRnnNodeGene
from population.utils.genome import Genome
from population.utils.network_util.feed_forward_net import make_net
from population.utils.visualizing.averaging_functions import SMA
from utils.dictionary import D_DONE, D_SENSOR_LIST
from utils.myutils import get_subfolder

# Parameters
TIME_SERIES_WIDTH = 8
TIME_SERIES_HEIGHT = 1.5
CORRECTION = 0.5


def main(population: Population,
         game_id: int,
         genome: Genome = None,
         game_cfg: Config = None,
         average: int = 1,
         debug: bool = False):
    """
    Monitor the genome on the following elements:
        * Position
        * Hidden state of SRU (Ht)
        * Actuation of both wheels
        * Distance
        * Delta distance
    """
    # Make sure all parameters are set
    if not genome: genome = population.best_genome
    if not game_cfg: game_cfg = pop.config
    
    # Check if valid genome (contains at least one hidden SRU, first SRU is monitored) - also possible for fixed RNNs
    a = len([n for n in genome.get_used_nodes().values() if type(n) == SimpleRnnNodeGene]) >= 1
    b = len([n for n in genome.get_used_nodes().values() if type(n) == FixedRnnNodeGene]) >= 1
    assert a or b
    
    # Get the game
    game = get_game(game_id, cfg=game_cfg, noise=False)
    state = game.reset()[D_SENSOR_LIST]
    step_num = 0
    
    # Create the network
    net = make_net(genome=genome,
                   genome_config=population.config.genome,
                   batch_size=1,
                   initial_read=state,
                   )
    
    # Containers to monitor
    actuation = []
    distance = []
    in2out = []
    delta_distance = []
    position = []
    Ht = []
    target_found = []
    score = 0
    in2out_conn = genome.connections[(-1, 0)].weight
    
    # Initialize the containers
    actuation.append([0, 0])
    distance.append(state[0])
    in2out.append(state[0] * in2out_conn)
    delta_distance.append(0)
    position.append(game.player.pos.get_tuple())
    Ht.append(net.rnn_state[0, 0, 0])
    if debug:
        print(f"Step: {step_num}")
        print(f"\t> Actuation: {(round(actuation[-1][0], 5), round(actuation[-1][1], 5))!r}")
        print(f"\t> Distance: {round(distance[-1], 5)} - Delta distance: {round(delta_distance[-1], 5)}")
        print(f"\t> Position: {(round(position[-1][0], 2), round(position[-1][1], 2))!r}")
        print(f"\t> SRU state: Ht={round(Ht[-1], 5)}")
    
    # Start monitoring
    while True:
        # Check if maximum iterations is reached
        if step_num == game_cfg.game.duration * game_cfg.game.fps: break
        
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
        actuation.append(action[0])
        distance.append(state[0])
        in2out.append(state[0] * in2out_conn)
        delta_distance.append(distance[-2] - distance[-1])
        position.append(game.player.pos.get_tuple())
        Ht.append(net.rnn_state[0, 0, 0])
        if debug:
            print(f"Step: {step_num}")
            print(f"\t> Actuation: {(round(actuation[-1][0], 5), round(actuation[-1][1], 5))!r}")
            print(f"\t> Distance: {round(distance[-1], 5)} - Delta distance: {round(delta_distance[-1], 5)}")
            print(f"\t> Position: {(round(position[-1][0], 2), round(position[-1][1], 2))!r}")
            print(f"\t> SRU state: Ht={round(Ht[-1], 5)}")
    
    if average > 1:
        # Average out the noise
        x, y = zip(*actuation)
        x = SMA(x, window=average)
        y = SMA(y, window=average)
        actuation = list(zip(x, y))
        distance = SMA(distance, window=average)
        delta_distance = SMA(delta_distance, window=average)
        Ht = SMA(Ht, window=average)
        
        # Resolve weird artifacts at the beginning
        for i in range(average, 0, -1):
            actuation[i - 1] = actuation[i]
            distance[i - 1] = distance[i]
            delta_distance[i - 1] = delta_distance[i]
            Ht[i - 1] = Ht[i]
    
    # Visualize the monitored values
    path = get_subfolder(f"population{'_backup' if population.use_backup else ''}/"
                         f"storage/"
                         f"{population.folder_name}/"
                         f"{population}/", "images")
    path = get_subfolder(path, f"monitor")
    path = get_subfolder(path, f"{genome.key}")
    path = get_subfolder(path, f"{game_id}")
    visualize_actuation(actuation,
                        target_found=target_found,
                        game_cfg=game_cfg.game,
                        save_path=f"{path}actuation.png")
    visualize_distance(distance,
                       target_found=target_found,
                       game_cfg=game_cfg.game,
                       save_path=f"{path}distance.png")
    visualize_in2out(in2out,
                     target_found=target_found,
                     game_cfg=game_cfg.game,
                     save_path=f"{path}in2out.png")
    visualize_hidden_state(Ht,
                           target_found=target_found,
                           game_cfg=game_cfg.game,
                           save_path=f"{path}hidden_state.png")
    visualize_position(position,
                       game=game,
                       save_path=f"{path}trace.png")
    merge(f"Monitored genome={genome.key} on game={game.id}", path=path)


def visualize_actuation(actuation_list: list, target_found: list, game_cfg: GameConfig, save_path: str):
    """Create a graph of the actuation over time"""
    actuation_list[0] = actuation_list[1]  # Ignore zero-actuation at start
    lw_a, rw_a = zip(*actuation_list)
    assert len(lw_a) == len(rw_a)
    time = [i / game_cfg.fps for i in range(len(actuation_list))]
    
    # Create the graph
    ax = plt.figure(figsize=(TIME_SERIES_WIDTH, TIME_SERIES_HEIGHT)).gca()
    plt.plot(time, lw_a, label="left")
    plt.plot(time, rw_a, label="right")
    for t in target_found: plt.axvline(x=t / game_cfg.fps, color='g', linestyle=':', linewidth=2)
    plt.legend()
    plt.grid()
    plt.xticks([i * 5 for i in range(9)])
    plt.yticks([0.8, 0.85, 0.9, 0.95])
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Forces to use only integers
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.title("Actuation force - Normalized")
    plt.xlim(0)
    # plt.ylabel("Normalized force")
    # plt.xlabel("Simulation time (s)")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02, dpi=500)
    plt.close()


def visualize_distance(distance_list: list, target_found: list, game_cfg: GameConfig, save_path: str):
    """Create a graph of the distance over time"""
    time = [i / game_cfg.fps for i in range(len(distance_list))]
    
    # Create the graph
    ax = plt.figure(figsize=(TIME_SERIES_WIDTH, TIME_SERIES_HEIGHT)).gca()
    plt.plot(time, distance_list)
    for t in target_found: plt.axvline(x=t / game_cfg.fps, color='g', linestyle=':', linewidth=2)
    plt.grid()
    plt.xticks([i * 5 for i in range(9)])
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Forces to use only integers
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.title("Distance to target - Normalized")
    plt.xlim(0)
    # plt.ylabel("Normalized distance")
    # plt.xlabel("Simulation time (s)")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02, dpi=500)
    plt.close()


def visualize_in2out(in2out_list: list, target_found: list, game_cfg: GameConfig, save_path: str):
    """Create a graph of the value received at the output directly from the network's input"""
    time = [i / game_cfg.fps for i in range(len(in2out_list))]
    
    # Create the graph
    ax = plt.figure(figsize=(TIME_SERIES_WIDTH, TIME_SERIES_HEIGHT)).gca()
    plt.plot(time, in2out_list)
    for t in target_found: plt.axvline(x=t / game_cfg.fps, color='g', linestyle=':', linewidth=2)
    plt.grid()
    plt.xticks([i * 5 for i in range(9)])
    plt.yticks([-1, -0.5, 0])
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Forces to use only integers
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.title("Direct influence of distance on output")
    plt.xlim(0)
    # plt.ylabel("Normalized distance")
    # plt.xlabel("Simulation time (s)")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02, dpi=500)
    plt.close()


def visualize_hidden_state(hidden_state: list, target_found: list, game_cfg: GameConfig, save_path: str):
    """Create a graph of the hidden stat's value over time"""
    time = [i / game_cfg.fps for i in range(len(hidden_state))]
    
    # Create the graph
    ax = plt.figure(figsize=(TIME_SERIES_WIDTH, TIME_SERIES_HEIGHT * 1.1)).gca()
    plt.plot(time, hidden_state)
    for t in target_found: plt.axvline(x=t / game_cfg.fps, color='g', linestyle=':', linewidth=2)
    plt.grid()
    plt.xticks([i * 5 for i in range(9)])
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Forces to use only integers
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.title("Hidden state")
    plt.xlim(0)
    # plt.ylabel("SRU output value")
    plt.xlabel("Simulation time (s)")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02, dpi=500)
    plt.close()


def visualize_position(position_list: list, game: Game, save_path: str):
    """Create a trace-graph."""
    plt.figure(figsize=(TIME_SERIES_HEIGHT * 3 - CORRECTION, TIME_SERIES_HEIGHT * 3 - CORRECTION)).gca()
    x_min, x_max = game.x_axis / 2, game.x_axis / 2
    y_min, y_max = game.y_axis / 2, game.y_axis / 2
    
    # Create the blueprint - Only visualize targets found by the agent
    for i in range(min(game.score + 1, len(game.spawn_function.locations))):
        t = game.spawn_function.locations[i]
        
        # Plot the targets
        plt.plot(t[0], t[1], 'go')
        # plt.annotate(str(i + 1), xy=(t[0] + 0.3, t[1] + 0.3))  # TODO: Uncomment if multiple targets
        
        # Add green dotted circle around targets
        c = plt.Circle((t[0], t[1]), 0.5, color='g', linestyle=':', linewidth=2, fill=False)
        plt.gca().add_artist(c)
        
        # Update the boundaries
        x_min, x_max = min(x_min, t[0]), max(x_max, t[0])
        y_min, y_max = min(y_min, t[1]), max(y_max, t[1])
    
    # Plot the player's path
    x_pos, y_pos = zip(*position_list)
    x_min, x_max = min(x_min, min(x_pos)), max(x_max, max(x_pos))
    y_min, y_max = min(y_min, min(y_pos)), max(y_max, max(y_pos))
    for p in range(0, len(x_pos), 5):
        plt.plot(x_pos[p], y_pos[p], 'ro', markersize=2)
        
        # Annotate every 5 seconds
        if p % (5 * game.game_config.fps) == 0 and p > 0:
            # offset = (x_pos[p + 1] - x_pos[p], y_pos[p + 1] - y_pos[p])
            offset = (y_pos[p] - y_pos[p - 1], x_pos[p - 1] - x_pos[p])
            plt.annotate(
                    str(int(p / game.game_config.fps)),
                    xy=(x_pos[p], y_pos[p]),
                    xytext=(x_pos[p] + offset[0] * 20, y_pos[p] + offset[1] * 20),
                    ha="center", va="center",
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc,rad=0."),
            )
    
    # Show the player's initial direction
    x = game.player.init_pos[0]
    y = game.player.init_pos[1]
    dx = 0.5 * cos(game.player.noisy_init_angle)
    dy = 0.5 * sin(game.player.noisy_init_angle)
    plt.arrow(x, y, dx, dy, color='k', head_width=0.15, width=0.02, zorder=10, length_includes_head=True)
    
    # Constraint the plot's boundaries
    x_center = (x_max - x_min) / 2 + x_min
    y_center = (y_max - y_min) / 2 + y_min
    r = max((x_max - x_min) / 2 + 1, (y_max - y_min) / 2 + 1)
    plt.xlim(x_center - r, x_center + r)
    plt.ylim(y_center - r, y_center + r)
    
    plt.grid()
    plt.title("Driving trace")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02, dpi=500)
    plt.close()


def merge(title: str, path: str):  # TODO
    """Merge each of the previously created images together"""
    # Load in all the images to merge
    # images = [plt.imread('population/utils/visualizing/images/white.png')]
    image_names = ['actuation', 'distance', 'in2out', 'hidden_state']
    images = [plt.imread(f'{path}{n}.png') for n in image_names]
    trace = plt.imread(f'{path}trace.png')
    
    # Make sure width of all images is the same
    for i in range(len(images)):
        while images[i].shape[1] > 3868:
            images[i] = images[i][:, :-1, :]
        while images[i].shape[1] < 3868:
            images[i] = np.concatenate((images[i], np.ones((images[i].shape[0], 1, images[i].shape[2]))), axis=1)
    
    # Concatenate the images, time_series vertical, and trace on the right
    # images.append(plt.imread('population/utils/visualizing/images/time774.png'))
    # images.append(plt.imread('population/utils/visualizing/images/white.png'))
    time_series = np.concatenate(images, axis=0)
    height = time_series.shape[0]
    while trace.shape[0] > height:
        if trace.shape[0] == height + 1:
            trace = trace[:-1, :, :]
        else:  # Symmetric prune
            trace = trace[1:-1, :, :]
    while trace.shape[0] < height:
        if trace.shape[0] == height - 1:
            trace = np.concatenate((trace,
                                    np.ones((1, trace.shape[1], trace.shape[2]))), axis=0)
        else:  # Symmetric addition
            if trace.shape[0] < height - 200:  # The big guns
                trace = np.concatenate((np.ones((100, trace.shape[1], trace.shape[2])),
                                        trace,
                                        np.ones((100, trace.shape[1], trace.shape[2]))), axis=0)
            elif trace.shape[0] < height - 20:  # The normal guns
                trace = np.concatenate((np.ones((10, trace.shape[1], trace.shape[2])),
                                        trace,
                                        np.ones((10, trace.shape[1], trace.shape[2]))), axis=0)
            else:  # Baby-steps
                trace = np.concatenate((np.ones((1, trace.shape[1], trace.shape[2])),
                                        trace,
                                        np.ones((1, trace.shape[1], trace.shape[2]))), axis=0)
    result = np.concatenate([time_series, trace], axis=1)
    
    # Create the figure
    plt.figure(figsize=(TIME_SERIES_WIDTH + 6 * TIME_SERIES_HEIGHT, 6 * TIME_SERIES_HEIGHT + 0.5))
    plt.axis('off')
    # plt.title(title, fontsize=24, fontweight='bold')
    plt.imshow(result)
    plt.tight_layout()
    plt.savefig(f"{path[:-1]}.png", bbox_inches='tight', pad_inches=0, dpi=500)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--game_id', type=int, default=10002)
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    
    # Go back to root
    os.chdir("../../../")
    
    _, eval_games = get_game_ids(experiment_id=4)
    
    # Create the population
    config = Config()
    pop = Population(
            name='test',
            # name='NEAT-SRU/v1',
            folder_name='test',
            # folder_name=get_folder(args.experiment),
            config=config,
    )
    chosen_genome = pop.best_genome
    # chosen_genome = pop.population[47280]
    
    # Run the monitor-tool
    main(
            population=pop,
            genome=chosen_genome,
            game_cfg=config,
            game_id=args.game_id,
            debug=args.debug
    )
