"""
deviate.py

Apply delta deviations on a given genome's weights and biases.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from genomes_gru.deviate_candidate_hidden import main as candidate_hidden_main
from genomes_gru.deviate_reset import main as reset_main
from genomes_gru.deviate_update import main as update_main
from genomes_gru.monitor_genome import main as monitor_genome
from genomes_gru.persist import load_genome
from genomes_gru.show_genome import main as show_gru
from population.utils.genome import Genome


def main(genome: Genome,
         gid: int,
         duration: int,
         delta: float,
         architecture: bool = False,
         monitor: bool = False,
         mut_bias: bool = False,
         mut_hh: bool = False,
         mut_xh: bool = False,
         ):
    """Run each of the 'deviate' scripts on the requested delta."""
    # Show genome's architecture if requested
    if architecture:
        show_gru(
                genome=genome,
                show=False
        )
    
    # Full fledged monitor on the default genome
    if monitor:
        monitor_genome(
                genome=genome,
                gid=gid,
                duration=duration,
                debug=False,
        )
    
    # Create each separate deviation plot
    reset_main(
            genome=genome,
            gid=gid,
            delta=delta,
            duration=duration,
            mut_bias=mut_bias,
            mut_hh=mut_hh,
            mut_xh=mut_xh,
    )
    update_main(
            genome=genome,
            gid=gid,
            delta=delta,
            duration=duration,
            mut_bias=mut_bias,
            mut_hh=mut_hh,
            mut_xh=mut_xh,
    )
    candidate_hidden_main(
            genome=genome,
            gid=gid,
            delta=delta,
            duration=duration,
            mut_bias=mut_bias,
            mut_hh=mut_hh,
            mut_xh=mut_xh,
    )
    
    # Merge the graphs
    name = ''
    if mut_bias: name += f"bias{delta}"
    if mut_hh: name += f"{'_' if name else ''}hh{delta}"
    if mut_xh: name += f"{'_' if name else ''}xh{delta}"
    path = f"genomes_gru/images/genome{genome.key}/"
    reset = plt.imread(f"{path}reset/{name}.png")
    update = plt.imread(f"{path}update/{name}.png")
    hidden = plt.imread(f"{path}candidate_hidden/{name}.png")
    result = np.concatenate([reset, update, hidden], axis=1)  # Concatenate horizontally
    
    # Create the figure
    plt.figure(figsize=(3 * 6, 8))
    plt.axis('off')
    plt.imshow(result)
    plt.tight_layout()
    plt.savefig(f"{path}{name}.png", bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--delta', type=float, default=.01)  # Deviation
    parser.add_argument('--duration', type=int, default=25)  # Simulation duration
    parser.add_argument('--gid', type=int, default=60001)  # First evaluation game of experiment3
    parser.add_argument('--name', type=str, default='genome2')
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
            duration=args.duration,
            delta=args.delta,
            architecture=True,
            monitor=True,
            mut_bias=True,
            mut_hh=True,
            mut_xh=True,
    )
