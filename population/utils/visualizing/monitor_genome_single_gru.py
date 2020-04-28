"""
monitor_genome_simple_gru.py

Monitor a single genome during its run on a single game. This monitoring focuses on the GRU-cell that must be present
in the genome.
"""
import argparse
import os
import warnings
from math import cos, sin

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from config import Config
from configs.game_config import GameConfig
from environment.game import Game, get_game
from main import get_game_ids
from population.population import Population
from population.utils.gene_util.gru import GruNodeGene
from population.utils.genome import Genome
from population.utils.network_util.activations import sigmoid
from population.utils.network_util.feed_forward_net import make_net
from population.utils.rnn_cell_util.berkeley_gru import GRUCell
from utils.dictionary import D_DONE, D_SENSOR_LIST
from utils.myutils import get_subfolder

# Parameters
TIME_SERIES_WIDTH = 8
TIME_SERIES_HEIGHT = 2
CORRECTION = 1.06


def main(population: Population, game_id: int, genome: Genome = None, game_cfg: Config = None, debug: bool = False):
    """
    Monitor the genome on the following elements:
        * Position
        * Reset gate (Rt) and Update gate (Zt)
        * Hidden state of GRU (Ht)
        * Actuation of both wheels
        * Distance
        * Delta distance
    """
    # Make sure all parameters are set
    if not genome: genome = population.best_genome
    if not game_cfg: game_cfg = pop.config
    
    # Check if valid genome (contains at least one hidden GRU, first GRU is monitored)
    assert len([n for n in genome.get_used_nodes().values() if type(n) == GruNodeGene]) >= 1
    
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
    delta_distance = []
    position = []
    Ht = []
    Rt = []
    Zt = []
    target_found = []
    score = 0
    
    # Initialize the containers
    actuation.append([0, 0])
    distance.append(state[0])
    delta_distance.append(0)
    position.append(game.player.pos.get_tuple())
    ht, rt, zt = get_gru_states(gru=net.rnn_array[0], x=np.asarray([state]))
    Ht.append(ht)
    Rt.append(rt)
    Zt.append(zt)
    if debug:
        print(f"Step: {step_num}")
        print(f"\t> Actuation: {(round(actuation[-1][0], 5), round(actuation[-1][1], 5))!r}")
        print(f"\t> Distance: {round(distance[-1], 5)} - Delta distance: {round(delta_distance[-1], 5)}")
        print(f"\t> Position: {(round(position[-1][0], 2), round(position[-1][1], 2))!r}")
        print(f"\t> GRU states: Ht={round(Ht[-1], 5)} - Rt={round(Rt[-1], 5)} - Zt={round(Zt[-1], 5)}")
    
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
        delta_distance.append(distance[-2] - distance[-1])
        position.append(game.player.pos.get_tuple())
        ht, rt, zt = get_gru_states(gru=net.rnn_array[0], x=np.asarray([state]))
        Ht.append(ht)
        Rt.append(rt)
        Zt.append(zt)
        if debug:
            print(f"Step: {step_num}")
            print(f"\t> Actuation: {(round(actuation[-1][0], 5), round(actuation[-1][1], 5))!r}")
            print(f"\t> Distance: {round(distance[-1], 5)} - Delta distance: {round(delta_distance[-1], 5)}")
            print(f"\t> Position: {(round(position[-1][0], 2), round(position[-1][1], 2))!r}")
            print(f"\t> GRU states: Ht={round(Ht[-1], 5)} - Rt={round(Rt[-1], 5)} - Zt={round(Zt[-1], 5)}")
    
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
    visualize_delta_distance(delta_distance,
                             target_found=target_found,
                             game_cfg=game_cfg.game,
                             save_path=f"{path}delta_distance.png")
    visualize_hidden_state(Ht,
                           target_found=target_found,
                           game_cfg=game_cfg.game,
                           save_path=f"{path}hidden_state.png")
    visualize_reset_gate(Rt,
                         target_found=target_found,
                         game_cfg=game_cfg.game,
                         save_path=f"{path}reset_gate.png")
    visualize_update_gate(Zt,
                          target_found=target_found,
                          game_cfg=game_cfg.game,
                          save_path=f"{path}update_gate.png")
    visualize_position(position,
                       game=game,
                       save_path=f"{path}trace.png")
    merge(f"Monitored genome={genome.key} on game={game.id}", path=path)


def get_gru_states(gru: GRUCell, x):
    W_xh = np.matmul(x, gru.weight_xh.transpose())
    W_hh = np.matmul(gru.hx, gru.weight_hh.transpose())
    R_t = sigmoid(W_xh[:, 0:1] + W_hh[:, 0:1] + gru.bias[0:1])
    Z_t = sigmoid(W_xh[:, 1:2] + W_hh[:, 1:2] + gru.bias[1:2])
    H_t = (1 - Z_t) * np.tanh(W_xh[:, 2:3] + R_t * W_hh[:, 2:3] + gru.bias[2:3]) + Z_t * gru.hx
    return H_t[0, 0], R_t[0, 0], Z_t[0, 0]


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
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Forces to use only integers
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.title("Actuation force - Normalized")
    # plt.ylabel("Normalized force")
    # plt.xlabel("Simulation time (s)")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def visualize_distance(distance_list: list, target_found: list, game_cfg: GameConfig, save_path: str):
    """Create a graph of the distance over time"""
    time = [i / game_cfg.fps for i in range(len(distance_list))]
    
    # Create the graph
    ax = plt.figure(figsize=(TIME_SERIES_WIDTH, TIME_SERIES_HEIGHT)).gca()
    plt.plot(time, distance_list)
    for t in target_found: plt.axvline(x=t / game_cfg.fps, color='g', linestyle=':', linewidth=2)
    plt.grid()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Forces to use only integers
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.title("Distance to target - Normalized")
    # plt.ylabel("Normalized distance")
    # plt.xlabel("Simulation time (s)")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def visualize_delta_distance(delta_distance_list: list, target_found: list, game_cfg: GameConfig, save_path: str):
    """Create a graph of the delta distance over time"""
    delta_distance_list[0] = delta_distance_list[1]  # Ignore zero-delta at start
    time = [i / game_cfg.fps for i in range(len(delta_distance_list))]
    max_abs = sorted([abs(d) for d in delta_distance_list])[-10]
    
    # Create the graph
    ax = plt.figure(figsize=(TIME_SERIES_WIDTH, TIME_SERIES_HEIGHT)).gca()
    plt.plot(time, delta_distance_list)
    for t in target_found: plt.axvline(x=t / game_cfg.fps, color='g', linestyle=':', linewidth=2)
    plt.grid()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Forces to use only integers
    plt.ylim([-max_abs * 1.1, max_abs * 1.1])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.title("Delta distance to target - Normalized")
    # plt.ylabel("Normalized delta distance")
    # plt.xlabel("Simulation time (s)")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def visualize_hidden_state(hidden_state: list, target_found: list, game_cfg: GameConfig, save_path: str):
    """Create a graph of the hidden stat's value over time"""
    time = [i / game_cfg.fps for i in range(len(hidden_state))]
    
    # Create the graph
    ax = plt.figure(figsize=(TIME_SERIES_WIDTH, TIME_SERIES_HEIGHT)).gca()
    plt.plot(time, hidden_state)
    for t in target_found: plt.axvline(x=t / game_cfg.fps, color='g', linestyle=':', linewidth=2)
    plt.grid()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Forces to use only integers
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.title("Hidden state")
    # plt.ylabel("GRU output value")
    # plt.xlabel("Simulation time (s)")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def visualize_reset_gate(reset_gate: list, target_found: list, game_cfg: GameConfig, save_path: str):
    """Create a graph of the reset gate's value over time"""
    time = [i / game_cfg.fps for i in range(len(reset_gate))]
    
    # Create the graph
    ax = plt.figure(figsize=(TIME_SERIES_WIDTH, TIME_SERIES_HEIGHT)).gca()
    plt.plot(time, reset_gate)
    for t in target_found: plt.axvline(x=t / game_cfg.fps, color='g', linestyle=':', linewidth=2)
    plt.grid()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Forces to use only integers
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.title("Reset gate")
    # plt.ylabel("Gate value")
    # plt.xlabel("Simulation time (s)")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def visualize_update_gate(update_gate: list, target_found: list, game_cfg: GameConfig, save_path: str):
    """Create a graph of the update gate's value over time"""
    time = [i / game_cfg.fps for i in range(len(update_gate))]
    
    # Create the graph
    ax = plt.figure(figsize=(TIME_SERIES_WIDTH, TIME_SERIES_HEIGHT)).gca()
    plt.plot(time, update_gate)
    for t in target_found: plt.axvline(x=t / game_cfg.fps, color='g', linestyle=':', linewidth=2)
    plt.grid()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Forces to use only integers
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.title("Update gate")
    # plt.ylabel("Gate value")
    # plt.xlabel("Simulation time (s)")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def visualize_position(position_list: list, game: Game, save_path: str):
    """Create a trace-graph."""
    plt.figure(figsize=(TIME_SERIES_HEIGHT * 6 - CORRECTION, TIME_SERIES_HEIGHT * 6 - CORRECTION)).gca()
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
    x_pos, y_pos = zip(*position_list)
    x_min, x_max = min(x_min, min(x_pos)), max(x_max, max(x_pos))
    y_min, y_max = min(y_min, min(y_pos)), max(y_max, max(y_pos))
    for p in range(0, len(x_pos), 5):
        plt.plot(x_pos[p], y_pos[p], 'ro', markersize=2)
        
        # Annotate every 5 seconds
        if p % (5 * game.game_config.fps) == 0 and p > 0:
            # offset = (x_pos[p + 1] - x_pos[p], y_pos[p + 1] - y_pos[p])
            offset = (y_pos[p - 1] - y_pos[p], x_pos[p] - x_pos[p - 1])
            plt.annotate(
                    str(int(p / game.game_config.fps)),
                    xy=(x_pos[p], y_pos[p]),
                    xytext=(x_pos[p] + offset[0] * 10, y_pos[p] + offset[1] * 10),
                    ha="center", va="center",
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc,rad=0."),
            )
    
    # Constraint the plot's boundaries
    x_center = (x_max - x_min) / 2 + x_min
    y_center = (y_max - y_min) / 2 + y_min
    r = max((x_max - x_min) / 2 + 1, (y_max - y_min) / 2 + 1)
    plt.xlim(x_center - r, x_center + r)
    plt.ylim(y_center - r, y_center + r)
    
    plt.grid()
    plt.title("Driving trace")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def merge(title: str, path: str):
    """Merge each of the previously created images together"""
    # Load in all the images to merge
    image_names = ['actuation', 'distance', 'delta_distance', 'hidden_state', 'reset_gate', 'update_gate']
    images = [plt.imread(f'{path}{n}.png') for n in image_names]
    trace = plt.imread(f'{path}trace.png')
    
    # Make sure width of all images is the same
    min_width = min([im.shape[1] for im in images])
    for i in range(len(images)):
        while images[i].shape[1] > min_width:
            images[i] = images[i][:, :-1, :]
    
    # Concatenate the images, time_series vertical, and trace on the right
    try:
        images.append(plt.imread('population/utils/visualizing/images/time774.png'))
        time_series = np.concatenate(images, axis=0)
        result = np.concatenate([time_series, trace], axis=1)
    except ValueError:
        try:
            images.pop(-1)
            images.append(plt.imread('population/utils/visualizing/images/time773.png'))
            time_series = np.concatenate(images, axis=0)
            result = np.concatenate([time_series, trace], axis=1)
        except ValueError as e:
            warnings.warn(f"Failed to monitor genome ({title}), silently terminating\n"
                          f"Received error: {e}")
            return
    
    # Create the figure
    plt.figure(figsize=(TIME_SERIES_WIDTH + 6 * TIME_SERIES_HEIGHT, 6 * TIME_SERIES_HEIGHT + 0.5))
    plt.axis('off')
    plt.title(title, fontsize=24, fontweight='bold')
    plt.imshow(result)
    plt.tight_layout()
    plt.savefig(f"{path[:-1]}.png", bbox_inches='tight', pad_inches=0)
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
            # name='NEAT-GRU/v1',
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
