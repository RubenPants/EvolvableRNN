"""
show_genome.py

Draw the genome's network.
"""
import argparse
import os

from config import Config
from genomes_gru.persist import load_genome
from population.utils.visualizing.genome_visualizer import draw_net
from utils.myutils import get_subfolder


def main(g_name: str, show: bool = False):
    """Visualize the genome's network."""
    g = load_genome(g_name=g_name)
    cfg = Config()
    path = get_subfolder('genomes_gru/images/', g_name)
    draw_net(config=cfg.genome,
             genome=g,
             debug=True,
             filename=f'{path}architecture',
             view=show)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--name', type=str, default='genome1')
    parser.add_argument('--show', type=int, default=1)
    args = parser.parse_args()
    
    # Go back to root
    os.chdir("..")
    
    # Run the script
    main(
            g_name=args.name,
            show=bool(args.show),
    )
