"""
population_neat_rnn_test.py

Test the basic functionality of a NEAT-RNN population (i.e. all genomes contain RNN hidden nodes).

:note: This test-suite will not disable prints of the population, hence it can be quite a mess...
"""
import os
import shutil
import unittest

from config import Config
from main import blueprint, evaluate, trace, trace_most_fit, train, visualize_genome
from population.population import Population


def get_population():
    """Get a dummy population with minimal configuration."""
    cfg = Config()
    cfg.game.duration = 1  # Small duration
    cfg.game.fps = 10  # Games of low accuracy but faster
    cfg.genome.rnn_prob_gru_nr = 0
    cfg.genome.rnn_prob_gru_nu = 0
    cfg.genome.rnn_prob_gru = 0
    cfg.genome.rnn_prob_lstm = 0
    cfg.genome.rnn_prob_simple_rnn = 1  # Always mutate a simple RNN-node
    cfg.population.pop_size = 2  # Keep a small population
    cfg.population.compatibility_thr = float('inf')  # Make sure that population does not expand
    cfg.update()
    
    # Create the population
    pop = Population(
            name='delete_me',
            folder_name='test_scenario_neat_rnn',
            config=cfg,
            log_print=False,
    )
    
    # Mutate population such that at least one genome has a simple RNN-node
    pop.population[1].mutate_add_node(cfg.genome)  # Population starts count at 1
    
    return pop, cfg


class PopulationNeatRnnTest(unittest.TestCase):
    """Test the basic population operations."""
    
    def setUp(self):
        """> Create the population used during testing."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        # Create the population
        get_population()
    
    def test_creation(self):
        """> Test if population can be successfully created."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        self.assertTrue(os.path.exists('population_backup/storage/test_scenario_neat_rnn/delete_me/'))
    
    def test_train(self):
        """> Test if population can be successfully trained."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        pop, cfg = get_population()
        train(
                population=pop,
                game_config=cfg,
                unused_cpu=0,
                iterations=1,
                debug=True,
                games=[-4],
        )
    
    def test_blueprint(self):
        """> Test if population can blueprint its results."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        pop, cfg = get_population()
        blueprint(
                population=pop,
                games=[-4],  # Random game
                game_config=cfg,
                debug=True,
        )
    
    def test_trace(self):
        """> Test if population can trace its current population."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        pop, cfg = get_population()
        trace(
                population=pop,
                games=[-4],  # Random game
                game_config=cfg,
                debug=True,
        )
    
    def test_trace_fit(self):
        """> Test if population can trace its best genome."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        pop, cfg = get_population()
        genome = pop.best_genome if pop.best_genome else list(pop.population.values())[-1]
        trace_most_fit(
                population=pop,
                genome=genome,
                games=[-4],  # Random game
                game_config=cfg,
                debug=True,
        )
    
    def test_evaluate(self):
        """> Test if the population can be evaluated."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        pop, cfg = get_population()
        evaluate(
                population=pop,
                game_config=cfg,
                games=[-4],
                debug=True,
        )
    
    def test_genome_visualization(self):
        """> Test if a genome from the population can be visualized."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        
        pop, cfg = get_population()
        genome = pop.best_genome if pop.best_genome else list(pop.population.values())[-1]
        visualize_genome(
                population=pop,
                genome=genome,
                debug=True,
                show=False,
        )
    
    def tearDown(self):
        """> Remove the population that was used for testing."""
        # Folder must be root to load in make_net properly
        if os.getcwd().split('\\')[-1] == 'tests': os.chdir('..')
        shutil.rmtree('population_backup/storage/test_scenario_neat_rnn')


def main():
    # Test wall collision
    prnn = PopulationNeatRnnTest()
    prnn.test_creation()
    prnn.test_train()
    prnn.test_blueprint()
    prnn.test_trace()
    prnn.test_trace_fit()
    prnn.test_evaluate()
    prnn.test_genome_visualization()


if __name__ == '__main__':
    unittest.main()
