"""
deviate_main.py

Show an upper (plus) and lower (minus) deviation from a default genome.
"""
import argparse
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from genomes_gru.persist import load_genome
from genomes_gru.trace import get_positions, visualize_positions
from population.utils.gene_util.gru import GruNodeGene
from population.utils.genome import Genome
from utils.myutils import get_subfolder

COLORS = ['r', 'b', 'c', 'm', 'y']


def main(genome: Genome,
         genome_plus: Genome,
         genome_minus: Genome,
         save_name: str,
         gid: int,
         duration: int = 60,
         ):
    """Visualize the deviated genomes."""
    # Check if valid genome (contains at least one hidden GRU, first GRU is monitored)
    assert len([n for n in genome.get_used_nodes().values() if type(n) == GruNodeGene]) >= 1
    assert len([n for n in genome_plus.get_used_nodes().values() if type(n) == GruNodeGene]) >= 1
    assert len([n for n in genome_minus.get_used_nodes().values() if type(n) == GruNodeGene]) >= 1
    
    # Trace the genomes
    positions, game = get_positions(genome=genome, gid=gid, duration=duration)
    positions_plus, _ = get_positions(genome=genome_plus, gid=gid, duration=duration)
    positions_minus, _ = get_positions(genome=genome_minus, gid=gid, duration=duration)
    
    # Visualize the genome paths
    pos_dict = dict()
    pos_dict['default'] = positions
    pos_dict[f'plus'] = positions_plus
    pos_dict[f'minus'] = positions_minus
    path = get_save_path(gid=genome.key, save_name=save_name)
    visualize_positions(
            positions=pos_dict,
            game=game,
            annotate_time=False,
            save_path=f"{path}.png",
            show=False,
    )


def merge(gid: int, save_name: str):
    """Merge the trajectory together with the state time-series."""
    # Load in the files
    path = get_save_path(gid=gid, save_name=f"{save_name}")
    trajectory = plt.imread(f"{path}_trajectory.png")
    state = plt.imread(f"{path}_state.png")
    result = np.concatenate([trajectory, state], axis=0)  # Concatenate vertically

    # Create the figure
    plt.figure(figsize=(6, 8))
    plt.axis('off')
    plt.imshow(result)
    plt.tight_layout()
    plt.savefig(f"{path}.png", bbox_inches='tight', pad_inches=0)
    plt.close()


def get_save_path(gid: int, save_name):
    """Get the correct path to save a certain file in."""
    save_path = get_subfolder(f"genomes_gru/", f"images")
    save_path = get_subfolder(save_path, f"genome{gid}")
    if len(save_name.split('/')) == 1:
        path = f"{save_path}{save_name}"
    elif len(save_name.split('/')) == 2:
        path = get_subfolder(f"{save_path}", save_name.split('/')[0])
        path = f"{path}{save_name.split('/')[1]}"
    else:
        raise Exception(f"Too long save_name: '{save_name}'")
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--duration', type=int, default=60)  # Simulation duration
    parser.add_argument('--gid', type=int, default=30001)  # First evaluation game of experiment3
    parser.add_argument('--name', type=str, default='genome1')
    parser.add_argument('--show', type=int, default=1)
    args = parser.parse_args()
    
    # Go back to root
    os.chdir("..")
    
    # Load in the genome
    g = load_genome(
            g_name=args.name,
    )
    g_plus = deepcopy(g)
    g_minus = deepcopy(g)
    # Execute the process
    main(
            genome=g,
            genome_plus=g_plus,
            genome_minus=g_minus,
            gid=args.gid,
            duration=args.duration,
            save_name='test/dummy_no_mutations'
    )
