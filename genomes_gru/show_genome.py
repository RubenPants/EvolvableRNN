"""
show_genome.py

Draw the genome's network.
"""
import argparse
import os

from config import Config
from genomes_gru.persist import load_genome
from population.utils.genome import Genome
from population.utils.visualizing.genome_visualizer import draw_net
from utils.myutils import get_subfolder


def main(genome: Genome, show: bool = False):
    """Visualize the genome's network."""
    cfg = Config()
    path = get_subfolder(f'genomes_gru/images/', f"genome{genome.key}")
    draw_net(config=cfg.genome,
             genome=genome,
             debug=True,
             filename=f'{path}architecture',
             view=show)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--name', type=str, default='genome2')
    parser.add_argument('--show', type=int, default=1)
    args = parser.parse_args()
    
    # Go back to root
    os.chdir("..")
    
    # Load in the genome
    g = load_genome(g_name=args.name)
    
    # Run the script
    main(
            genome=g,
            show=bool(args.show),
    )
