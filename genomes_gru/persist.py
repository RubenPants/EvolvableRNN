"""
persist.py

Load and save the requested genome.
"""
import os
import pickle
from glob import glob

from population.population import Population
from population.utils.genome import Genome


def load_genome(g_name: str):
    """Load in one of the persisted genomes."""
    with open(f"genomes_gru/genomes/{g_name}.pickle", 'rb') as f:
        return pickle.load(f)


def store_genome(genome: Genome, g_name: str = None):
    """Persist a single genome."""
    if not g_name:
        genomes = glob(f"genomes_gru/genomes/genome*.pickle")
        g_name = f"genome{len(genomes) + 1}"
        genome.key = len(genomes) + 1
    with open(f"genomes_gru/genomes/{g_name}.pickle", 'wb') as f:
        return pickle.dump(genome, f)


def pull(pop_name: str, pop_folder: str, gid: int = None, backup_pop: bool = True):
    """Pull a genome from a specified population"""
    pop = Population(
            name=pop_name,
            folder_name=pop_folder,
            use_backup=backup_pop,
    )
    genome = pop.population[gid] if gid else pop.best_genome
    store_genome(genome=genome)


if __name__ == '__main__':
    # Go back to root
    os.chdir("..")
    
    pull(
            pop_name='NEAT-GRU/v8',
            pop_folder='experiment3',
    )
