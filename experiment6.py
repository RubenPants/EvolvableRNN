"""
experiment6.py

Perform an infinite loop of genome-creations and -evaluations.
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from random import random

import matplotlib.pyplot as plt
from numpy.random import random as rand_arr
from six import iteritems

from config import Config
from main import get_game_ids
from population.utils.gene_util.connection import ConnectionGene
from population.utils.gene_util.gru import GruNodeGene
from population.utils.gene_util.output_node import OutputNodeGene
from population.utils.gene_util.simple_rnn import SimpleRnnNodeGene
from population.utils.genome import Genome
from population.utils.population_util.fitness_functions import calc_pop_fitness
from utils.dictionary import D_DISTANCE_SCORE
from utils.myutils import get_subfolder


# --------------------------------------------------> MAIN METHODS <-------------------------------------------------- #

def train(topology_id: int, batch_size: int = 1000, unused_cpu: int = 2, use_backup: bool = False):
    """Create an infinite amount of genomes for the requested topology."""
    # Get the config
    cfg = get_config()
    
    # Get initial genome key
    g_key, csv_path = get_initial_keys(topology_id, use_backup=use_backup)
    
    # Setup the environment
    _, games = get_game_ids(experiment_id=6)
    multi_env = get_multi_env(config=cfg)
    multi_env.set_games(games, noise=False)
    
    # Create genomes
    t = time.localtime()
    print(f"\nCurrent time: {t.tm_hour:02d}h-{t.tm_min:02d}m-{t.tm_sec:02d}s")
    print(f"> Evaluating {batch_size} genomes in csv '{csv_path.split('/')[-1]}'")
    genomes = list(iteritems({g_key + i: get_genome(topology_id, g_id=g_key + i, cfg=cfg) for i in range(batch_size)}))
    g_key += batch_size
    
    # Evaluate the genome-dictionary in parallel
    try:
        pool = mp.Pool(mp.cpu_count() - unused_cpu)
        manager = mp.Manager()
        return_dict = manager.dict()
        for genome in genomes:
            pool.apply_async(func=multi_env.eval_genome, args=(genome, return_dict))
        pool.close()  # Close the pool
        pool.join()  # Postpone continuation until everything is finished
        
        # Calculate the fitness from the given return_dict
        fitness = calc_pop_fitness(
                fitness_cfg=cfg.evaluation,
                game_cfg=cfg.game,
                game_obs=return_dict,
        )
        for i, genome in genomes:
            genome.fitness = fitness[i]
        
        # Write the result to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for _, g in genomes:
                writer.writerow(get_genome_parameters(g, topology_id=topology_id))
    except KeyboardInterrupt:
        # Remove the temporary CSV first
        os.remove(csv_path)
        raise KeyboardInterrupt


def visualize_bar(topology_id: int, rounding: int = 2, use_backup: bool = False):
    """Visualize a bar-plot of how many genomes obtained which fitness score"""
    fitness = []
    path_shared = get_subfolder(f"population{'_backup' if use_backup else ''}/storage/", "experiment6")
    path_data = get_subfolder(path_shared, "data")
    path_images = get_subfolder(path_shared, 'images')
    name = f"topology_{topology_id}"
    csv_name = f"{path_data}{name}.csv"
    
    # Read in the scores
    total_size = 0
    with open(csv_name, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip the headers
        for row in reader:
            fitness.append(round(float(row[-1]), rounding))
            total_size += 1
    
    # Count the scores
    c = Counter()
    for f in fitness:
        c[f] += 1
    
    # Plot the result
    plt.figure(figsize=(10, 5))
    x, y = zip(*sorted(c.items()))
    i = 1
    while i <= max(y):
        plt.axhline(i, color="grey", linewidth=0.5)
        i *= 10
    plt.bar(x, y, width=1 / (10 ** rounding))
    plt.yscale('log')
    plt.title("Fitness-distribution of uniformly sampled genome-space")
    plt.ylabel("Number of genomes")
    plt.xlabel("Fitness score")
    plt.savefig(f"{path_images}{name}.png")
    # plt.show()
    plt.close()


# -------------------------------------------------> HELPER METHODS <------------------------------------------------- #

def get_config():
    """Get the game-config."""
    cfg = Config()
    cfg.bot.angular_dir = []
    cfg.bot.delta_dist_enabled = False
    cfg.bot.dist_enabled = True
    cfg.evaluation.fitness = D_DISTANCE_SCORE
    cfg.game.duration = 60  # 60 seconds should be enough to reach the target from each starting orientation
    cfg.update()
    return cfg


def get_genome_parameters(g, topology_id: int):
    """Unravel the genome's parameters as an ordered list."""
    result = [v for v in g.nodes[2].bias_h]  # GRU biases
    result += [v[0] for v in g.nodes[2].weight_xh_full]  # GRU input->output
    result += [v[0] for v in g.nodes[2].weight_hh]  # GRU hidden->hidden
    if topology_id in [1]:
        result += [g.connections[(2, 1)].weight, g.connections[(-1, 1)].weight]
    elif topology_id in [2, 22]:
        result += [g.nodes[1].bias]
        result += [g.connections[(-1, 1)].weight]
    elif topology_id in [3, 33]:
        result += [g.nodes[1].bias]
        result += [g.connections[(-1, 2)].weight, g.connections[(2, 1)].weight, g.connections[(-1, 1)].weight]
    else:
        raise Exception(f"Topology of ID {topology_id} not supported!")
    result += [g.fitness]
    return result


def get_multi_env(config: Config):
    """Create a multi-environment used to evaluate the genomes."""
    if 'linux' in sys.platform:
        from environment.cy.env_multi_cy import MultiEnvironmentCy
        return MultiEnvironmentCy(
                game_config=config,
                pop_config=config,
        )
    elif 'win32' in sys.platform:
        from environment.env_multi import MultiEnvironment
        return MultiEnvironment(
                game_config=config,
                pop_config=config,
        )
    else:
        raise Exception(f"Platform {sys.platform} not yet supported!")


def get_initial_keys(topology_id: int, use_backup: bool):
    """Get the genome-key based on CSV-file's length."""
    path = get_subfolder(f"population{'_backup' if use_backup else ''}/storage/", "experiment6")
    path = get_subfolder(path, "data")
    csv_name = f"topology_{topology_id}"
    path = f"{path}{csv_name}.csv"
    
    # CSV exists, count number of rows
    if os.path.exists(path):
        with open(path, 'r') as f:
            return sum(1 for _ in f), path
    
    # CSV does not exist, create new
    else:
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Construct the CSV's head, all genomes have the full GRU-parameter suite
            head = ['bias_r', 'bias_z', 'bias_h',
                    'weight_xr', 'weight_xz', 'weight_xh',
                    'weight_hr', 'weight_hz', 'weight_hh']
            if topology_id in [1]:
                head += ['conn1', 'conn2']
            elif topology_id in [2, 22]:
                head += ['bias_rw', 'conn2']
            elif topology_id in [3, 33]:
                head += ['bias_rw', 'conn0', 'conn1', 'conn2']
            else:
                raise Exception(f"Topology ID {topology_id} not supported!")
            head += ['fitness']
            writer.writerow(head)
            return 1, path


# ----------------------------------------------> POSSIBLE TOPOLOGIES <----------------------------------------------- #

def get_genome(topology_id: int, g_id: int, cfg: Config):
    """Get the genome corresponding the given topology_id."""
    if topology_id == 1:
        topology = get_topology1
    elif topology_id == 2:
        topology = get_topology2
    elif topology_id == 22:
        topology = get_topology22
    elif topology_id == 3:
        topology = get_topology3
    elif topology_id == 33:
        topology = get_topology33
    else:
        raise Exception(f"Topology ID '{topology_id}' not supported")
    return topology(g_id, cfg)


def enforce_topology(g: Genome, topology_id: int):
    """Enforce the genome to the requested topology. It is assumed that topology hasn't changed."""
    if topology_id == 1:
        enforce_topology1(g)
    elif topology_id == 2:
        enforce_topology2(g)
    elif topology_id == 22:
        enforce_topology22(g)
    elif topology_id == 3:
        enforce_topology3(g)
    elif topology_id == 33:
        enforce_topology33(g)
    else:
        raise Exception(f"Topology ID '{topology_id}' not supported")


def get_topology1(gid: int, cfg: Config):
    """
    Create a uniformly and randomly sampled genome of fixed topology:
      (key=0, bias=1.5)  (key=1, bias=0)
                     ____ /   /
                   /         /
                GRU         /
                |     _____/
                |   /
              (key=-1)
    """
    # Create an initial dummy genome with fixed configuration
    genome = Genome(
            key=gid,
            num_outputs=cfg.genome.num_outputs,
            bot_config=cfg.bot,
    )
    
    # Create the nodes
    genome.nodes[0] = OutputNodeGene(key=0, cfg=cfg.genome)  # OutputNode 0
    genome.nodes[0].bias = 1.5  # Drive with full actuation by default
    genome.nodes[1] = OutputNodeGene(key=1, cfg=cfg.genome)  # OutputNode 1
    genome.nodes[1].bias = 0  # Drive with 0.5 actuation by default
    genome.nodes[2] = GruNodeGene(key=2, cfg=cfg.genome, input_keys=[-1], input_keys_full=[-1])  # Hidden node
    genome.nodes[2].bias = 0  # Bias is irrelevant for GRU-node
    
    # Setup the parameter-ranges
    conn_range = cfg.genome.weight_max_value - cfg.genome.weight_min_value
    bias_range = cfg.genome.bias_max_value - cfg.genome.bias_min_value
    rnn_range = cfg.genome.rnn_max_value - cfg.genome.rnn_min_value
    
    # Uniformly sample the genome's GRU-component
    genome.nodes[2].bias_h = rand_arr((3,)) * bias_range + cfg.genome.bias_min_value
    genome.nodes[2].weight_xh_full = rand_arr((3, 1)) * rnn_range + cfg.genome.weight_min_value
    genome.nodes[2].weight_hh = rand_arr((3, 1)) * rnn_range + cfg.genome.weight_min_value
    
    # Create the connections
    genome.connections = dict()
    
    # input2gru
    key = (-1, 2)
    genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
    genome.connections[key].weight = 1  # Simply forward distance
    genome.connections[key].enabled = True
    
    # gru2output - Uniformly sampled
    key = (2, 1)
    genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
    genome.connections[key].weight = random() * conn_range + cfg.genome.weight_min_value
    genome.connections[key].enabled = True
    
    # input2output - Uniformly sampled
    key = (-1, 1)
    genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
    genome.connections[key].weight = random() * conn_range + cfg.genome.weight_min_value
    genome.connections[key].enabled = True
    
    genome.update_rnn_nodes(config=cfg.genome)
    return genome


def get_topology2(gid: int, cfg: Config):
    """
    Create a uniformly and randomly sampled genome of fixed topology:
    Sigmoid with bias 1.5 --> Actuation default of 95,3%
      (key=0, bias=1.5)      (key=1, bias=?)
                     ____ /   /
                   /         /
                GRU         /
                |     _____/
                |   /
              (key=-1)
    """
    # Create an initial dummy genome with fixed configuration
    genome = Genome(
            key=gid,
            num_outputs=cfg.genome.num_outputs,
            bot_config=cfg.bot,
    )
    
    # Setup the parameter-ranges
    conn_range = cfg.genome.weight_max_value - cfg.genome.weight_min_value
    bias_range = cfg.genome.bias_max_value - cfg.genome.bias_min_value
    rnn_range = cfg.genome.rnn_max_value - cfg.genome.rnn_min_value
    
    # Create the nodes
    genome.nodes[0] = OutputNodeGene(key=0, cfg=cfg.genome)  # OutputNode 0
    genome.nodes[0].bias = 1.5  # Drive with 0.953 actuation by default
    genome.nodes[1] = OutputNodeGene(key=1, cfg=cfg.genome)  # OutputNode 1
    genome.nodes[1].bias = random() * bias_range + cfg.genome.bias_min_value  # Uniformly sampled bias
    genome.nodes[2] = GruNodeGene(key=2, cfg=cfg.genome, input_keys=[-1], input_keys_full=[-1])  # Hidden node
    genome.nodes[2].bias = 0  # Bias is irrelevant for GRU-node
    
    # Uniformly sample the genome's GRU-component
    genome.nodes[2].bias_h = rand_arr((3,)) * bias_range + cfg.genome.bias_min_value
    genome.nodes[2].weight_xh_full = rand_arr((3, 1)) * rnn_range + cfg.genome.weight_min_value
    genome.nodes[2].weight_hh = rand_arr((3, 1)) * rnn_range + cfg.genome.weight_min_value
    
    # Create the connections
    genome.connections = dict()
    
    # input2gru
    key = (-1, 2)
    genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
    genome.connections[key].weight = 1  # Simply forward distance
    genome.connections[key].enabled = True
    
    # gru2output - Uniformly sampled
    key = (2, 1)
    genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
    genome.connections[key].weight = 3  # Enforce capabilities of full spectrum
    genome.connections[key].enabled = True
    
    # input2output - Uniformly sampled
    key = (-1, 1)
    genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
    genome.connections[key].weight = random() * conn_range + cfg.genome.weight_min_value
    genome.connections[key].enabled = True
    
    genome.update_rnn_nodes(config=cfg.genome)
    return genome


def get_topology22(gid: int, cfg: Config):
    """
    Create a uniformly and randomly sampled genome of fixed topology:
    Sigmoid with bias 1.5 --> Actuation default of 95,3%
      (key=0, bias=1.5)      (key=1, bias=?)
                     ____ /   /
                   /         /
                SRU         /
                 |    _____/
                |   /
              (key=-1)
    """
    # Create an initial dummy genome with fixed configuration
    genome = Genome(
            key=gid,
            num_outputs=cfg.genome.num_outputs,
            bot_config=cfg.bot,
    )
    
    # Setup the parameter-ranges
    conn_range = cfg.genome.weight_max_value - cfg.genome.weight_min_value
    bias_range = cfg.genome.bias_max_value - cfg.genome.bias_min_value
    rnn_range = cfg.genome.rnn_max_value - cfg.genome.rnn_min_value
    
    # Create the nodes
    genome.nodes[0] = OutputNodeGene(key=0, cfg=cfg.genome)  # OutputNode 0
    genome.nodes[0].bias = 1.5  # Drive with 0.953 actuation by default
    genome.nodes[1] = OutputNodeGene(key=1, cfg=cfg.genome)  # OutputNode 1
    genome.nodes[1].bias = random() * bias_range + cfg.genome.bias_min_value  # Uniformly sampled bias
    genome.nodes[2] = SimpleRnnNodeGene(key=2, cfg=cfg.genome, input_keys=[-1], input_keys_full=[-1])  # Hidden node
    genome.nodes[2].bias = 0  # Bias is irrelevant for GRU-node
    
    # Uniformly sample the genome's GRU-component
    genome.nodes[2].bias_h = rand_arr((1,)) * bias_range + cfg.genome.bias_min_value
    genome.nodes[2].weight_xh_full = rand_arr((1, 1)) * rnn_range + cfg.genome.weight_min_value
    genome.nodes[2].weight_hh = rand_arr((1, 1)) * rnn_range + cfg.genome.weight_min_value
    
    # Create the connections
    genome.connections = dict()
    
    # input2gru
    key = (-1, 2)
    genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
    genome.connections[key].weight = 1  # Simply forward distance
    genome.connections[key].enabled = True
    
    # gru2output - Uniformly sampled
    key = (2, 1)
    genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
    genome.connections[key].weight = 3  # Enforce capabilities of full spectrum
    genome.connections[key].enabled = True
    
    # input2output - Uniformly sampled
    key = (-1, 1)
    genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
    genome.connections[key].weight = random() * conn_range + cfg.genome.weight_min_value
    genome.connections[key].enabled = True
    
    genome.update_rnn_nodes(config=cfg.genome)
    return genome


def get_topology3(gid: int, cfg: Config):
    """
    Create a uniformly and randomly sampled genome of fixed topology:
    Sigmoid with bias 1.5 --> Actuation default of 95,3%
      (key=0, bias=1.5)   (key=1, bias=?)
                      ____ /  /
                    /        /
                 GRU        /
                 |    _____/
                |   /
              (key=-1)
    """
    # Create an initial dummy genome with fixed configuration
    genome = Genome(
            key=gid,
            num_outputs=cfg.genome.num_outputs,
            bot_config=cfg.bot,
    )
    
    # Setup the parameter-ranges
    conn_range = cfg.genome.weight_max_value - cfg.genome.weight_min_value
    bias_range = cfg.genome.bias_max_value - cfg.genome.bias_min_value
    rnn_range = cfg.genome.rnn_max_value - cfg.genome.rnn_min_value
    
    # Create the nodes
    genome.nodes[0] = OutputNodeGene(key=0, cfg=cfg.genome)  # OutputNode 0
    genome.nodes[0].bias = 1.5  # Drive with 0.953 actuation by default
    genome.nodes[1] = OutputNodeGene(key=1, cfg=cfg.genome)  # OutputNode 1
    genome.nodes[1].bias = random() * bias_range + cfg.genome.bias_min_value  # Uniformly sampled bias
    genome.nodes[2] = GruNodeGene(key=2, cfg=cfg.genome, input_keys=[-1], input_keys_full=[-1])  # Hidden node
    genome.nodes[2].bias = 0  # Bias is irrelevant for GRU-node
    
    # Uniformly sample the genome's GRU-component
    genome.nodes[2].bias_h = rand_arr((3,)) * bias_range + cfg.genome.bias_min_value
    genome.nodes[2].weight_xh_full = rand_arr((3, 1)) * rnn_range + cfg.genome.weight_min_value
    genome.nodes[2].weight_hh = rand_arr((3, 1)) * rnn_range + cfg.genome.weight_min_value
    
    # Create the connections
    genome.connections = dict()
    
    # input2gru
    key = (-1, 2)
    genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
    genome.connections[key].weight = random() * conn_range + cfg.genome.weight_min_value
    genome.connections[key].enabled = True
    
    # gru2output - Uniformly sampled
    key = (2, 1)
    genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
    genome.connections[key].weight = random() * conn_range + cfg.genome.weight_min_value
    genome.connections[key].enabled = True
    
    # input2output - Uniformly sampled
    key = (-1, 1)
    genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
    genome.connections[key].weight = random() * conn_range + cfg.genome.weight_min_value
    genome.connections[key].enabled = True
    
    genome.update_rnn_nodes(config=cfg.genome)
    return genome


def get_topology33(gid: int, cfg: Config):
    """
    Create a uniformly and randomly sampled genome of fixed topology:
    Sigmoid with bias 1.5 --> Actuation default of 95,3%
      (key=0, bias=1.5) (key=1, bias=?)
                      __ /   /
                    /       /
                 SRU       /
                 |   _____/
                |   /
              (key=-1)
    """
    # Create an initial dummy genome with fixed configuration
    genome = Genome(
            key=gid,
            num_outputs=cfg.genome.num_outputs,
            bot_config=cfg.bot,
    )
    
    # Setup the parameter-ranges
    conn_range = cfg.genome.weight_max_value - cfg.genome.weight_min_value
    bias_range = cfg.genome.bias_max_value - cfg.genome.bias_min_value
    rnn_range = cfg.genome.rnn_max_value - cfg.genome.rnn_min_value
    
    # Create the nodes
    genome.nodes[0] = OutputNodeGene(key=0, cfg=cfg.genome)  # OutputNode 0
    genome.nodes[0].bias = 1.5  # Drive with 0.953 actuation by default
    genome.nodes[1] = OutputNodeGene(key=1, cfg=cfg.genome)  # OutputNode 1
    genome.nodes[1].bias = random() * bias_range + cfg.genome.bias_min_value  # Uniformly sampled bias
    genome.nodes[2] = SimpleRnnNodeGene(key=2, cfg=cfg.genome, input_keys=[-1], input_keys_full=[-1])  # Hidden node
    genome.nodes[2].bias = 0  # Bias is irrelevant for GRU-node
    
    # Uniformly sample the genome's GRU-component
    genome.nodes[2].bias_h = rand_arr((1,)) * bias_range + cfg.genome.bias_min_value
    genome.nodes[2].weight_xh_full = rand_arr((1, 1)) * rnn_range + cfg.genome.weight_min_value
    genome.nodes[2].weight_hh = rand_arr((1, 1)) * rnn_range + cfg.genome.weight_min_value
    
    # Create the connections
    genome.connections = dict()
    
    # input2gru
    key = (-1, 2)
    genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
    genome.connections[key].weight = random() * conn_range + cfg.genome.weight_min_value
    genome.connections[key].enabled = True
    
    # gru2gru - Uniformly sampled
    key = (2, 1)
    genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
    genome.connections[key].weight = random() * conn_range + cfg.genome.weight_min_value
    genome.connections[key].enabled = True
    
    # input2output - Uniformly sampled
    key = (-1, 1)
    genome.connections[key] = ConnectionGene(key=key, cfg=cfg.genome)
    genome.connections[key].weight = random() * conn_range + cfg.genome.weight_min_value
    genome.connections[key].enabled = True
    
    genome.update_rnn_nodes(config=cfg.genome)
    return genome


def enforce_topology1(g: Genome):
    """Enforce the fixed parameters of topology1. It is assumed that topology hasn't changed."""
    g.nodes[0].bias = 1.5  # Drive with full speed by default
    g.nodes[1].bias = 0  # Drive with 0.5 actuation by default
    g.connections[(-1, 2)].weight = 1  # Simply forward distance


def enforce_topology2(g: Genome):
    """Enforce the fixed parameters of topology2. It is assumed that topology hasn't changed."""
    g.nodes[0].bias = 1.5  # Drive with 0.953 actuation by default
    g.connections[(-1, 2)].weight = 1  # Simply forward distance
    g.connections[(2, 1)].weight = 3  # Enforce capabilities of full spectrum


def enforce_topology22(g: Genome):
    enforce_topology2(g)


def enforce_topology3(g: Genome):
    """Enforce the fixed parameters of topology3. It is assumed that topology hasn't changed."""
    g.nodes[0].bias = 1.5  # Drive with 0.953 actuation by default


def enforce_topology33(g: Genome):
    enforce_topology3(g)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--evaluate', type=bool, default=True)  # Evaluate new genomes
    parser.add_argument('--topology_id', type=int, default=1)  # ID of the used topology
    parser.add_argument('--batch', type=int, default=10000)  # Number of genomes evaluated per batch
    parser.add_argument('--unused_cpu', type=int, default=2)  # Number of CPU cores not used during evaluation
    parser.add_argument('--visualize', type=bool, default=True)  # Visualize the current results
    parser.add_argument('--use_backup', type=bool, default=False)  # Use the backup-data
    args = parser.parse_args()
    
    # Run the program
    if args.evaluate:
        train(
                topology_id=args.topology_id,
                batch_size=args.batch,
                unused_cpu=args.unused_cpu,
                use_backup=args.use_backup,
        )
    if args.visualize:
        visualize_bar(
                topology_id=args.topology_id,
                use_backup=args.use_backup,
        )
