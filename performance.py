import argparse
import multiprocessing as mp
import os
import shutil
from timeit import timeit

from six import iteritems

from config import Config
from environment.cy.env_multi_cy import MultiEnvironmentCy
from environment.env_multi import MultiEnvironment
from main import get_game_ids
from population.population import Population


def simulation(cython: bool = False, parallel: bool = False):
    """Execute a simulation of one training iteration."""
    # Create the population
    cfg = Config()
    cfg.game.duration = 200
    cfg.update()
    pop = Population(
            name="delete_me",
            config=cfg,
            folder_name="test_performance",
            use_backup=True,
            overwrite=True,  # Every iteration, create a new population from scratch
    )
    
    # Perform the simulations of experiment3
    train_games, _ = get_game_ids(experiment_id=3)
    if cython:
        multi_env = MultiEnvironmentCy(game_config=cfg, pop_config=pop.config)
    else:
        multi_env = MultiEnvironment(game_config=cfg, pop_config=pop.config)
    multi_env.set_games(train_games, noise=True)
    
    # Prepare the generation's reporters for the generation
    pop.reporters.start_generation(gen=pop.generation, logger=pop.log)
    
    # Fetch the dictionary of genomes
    genomes = list(iteritems(pop.population))
    
    if parallel:
        # Initialize the evaluation-pool
        pool = mp.Pool(mp.cpu_count())
        manager = mp.Manager()
        return_dict = manager.dict()
        
        for genome in genomes:
            pool.apply_async(func=multi_env.eval_genome, args=(genome, return_dict))
        pool.close()  # Close the pool
        pool.join()  # Postpone continuation until everything is finished
    else:
        return_dict = dict()
        for genome in genomes:
            multi_env.eval_genome(genome, return_dict)
    
    path = f"population_backup/storage/test_performance/"
    if os.path.exists(path):
        shutil.rmtree(path)


def test_drive():
    print("--> Running the drive test <--")
    
    # Do measures
    N = 100
    time_cy = timeit('drive_test_cy.main()',
                     setup='from tests.cy import drive_test_cy',
                     number=N) / N
    time_py = timeit('drive_test.main()',
                     setup='from tests import drive_test',
                     number=N) / N
    
    # Print results
    print("Time Python:", time_py)
    print("Time Cython:", time_cy)
    print("Cython is {} times faster on the drive test".format(round(time_py / time_cy, 2)))


def test_simulation():
    print("--> Running the simulation test (may take a while..) <--")
    
    # Do measures
    N = 20
    time_py = timeit('simulation(False, False)',
                     setup='from performance import simulation',
                     number=N) / N
    time_cy = timeit('simulation(True, False)',
                     setup='from performance import simulation',
                     number=N) / N
    time_py_p = timeit('simulation(False, True)',
                       setup='from performance import simulation',
                       number=N) / N
    time_cy_p = timeit('simulation(True, True)',
                       setup='from performance import simulation',
                       number=N) / N
    
    # Print results
    print("Time Python:", time_py)
    print("Time Cython:", time_cy)
    print("Time Python - parallel:", time_py_p)
    print("Time Cython - parallel:", time_cy_p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--eval', type=str, default='all')
    args = parser.parse_args()
    
    x = args.eval
    # if x == 'all' or x == 'drive':
    #     test_drive()
    if x == 'all' or x == 'simulation':
        test_simulation()
