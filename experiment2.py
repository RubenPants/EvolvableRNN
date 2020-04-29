"""
experiment2.py

Run the second experiment.
"""
import argparse
import os
import traceback
from distutils.dir_util import copy_tree

from config import Config
from main import evaluate, get_folder, get_game_ids, get_name
from population.population import Population
from process_killer import main as process_killer
from utils.dictionary import *


def main(fitness,
         prob_gru: float,
         prob_sru: float,
         prob_lstm: float,
         version=0,
         unused_cpu=1,
         ):
    """
    Run a population's configuration.

    :param fitness: Fitness function used to evaluate the population
    :param prob_gru: Probability of mutating towards a GRU-node
    :param prob_sru: Probability of mutating towards a SRU-node
    :param prob_lstm: Probability of mutating towards a LSTM-node
    :param version: Version of the model
    :param unused_cpu: Number of CPUs not used during training
    """
    # Re-configure the config-file
    cfg = Config()
    cfg.bot.angular_dir = []
    cfg.bot.delta_dist_enabled = False
    cfg.bot.dist_enabled = True
    cfg.game.duration = 60  # 60 seconds should be enough to reach the target from each starting orientation
    cfg.population.pop_size = 512
    
    # Let inputs apply to configuration
    cfg.genome.rnn_prob_gru = prob_gru
    cfg.genome.rnn_prob_simple_rnn = prob_sru
    cfg.genome.rnn_prob_lstm = prob_lstm
    cfg.evaluation.fitness = fitness
    cfg.update()
    
    # Copy population over from experiment1
    name = get_name(cfg=cfg, version=version)
    path_exp1 = f'population/storage/experiment1/{name}/'
    if not os.path.exists(path_exp1):
        raise Exception(f"Experiment 1 must be executed first for population {name}, terminating experiment 2...")
    
    # Population exists in experiment1, copy over to experiment2 (change experiment1 population's folder and save)
    pop = Population(
            name=name,
            config=cfg,
            folder_name=get_folder(experiment_id=1),
            use_backup=False
    )
    assert pop.generation > 0  # Population is not new (redundant check)
    folder = get_folder(experiment_id=2)
    pop.folder_name = folder
    pop.save()  # Overrides pre-existing populations!
    
    # Copy over all generations as well, since these are used during population evaluation
    path = f"population{'_backup' if pop.use_backup else ''}/storage/{pop.folder_name}/{pop}/"
    copy_tree(f"{path_exp1}generations", f"{path}generations")
    
    # Give overview of population
    gru = pop.config.genome.rnn_prob_gru
    gru_nr = pop.config.genome.rnn_prob_gru_nr
    gru_nu = pop.config.genome.rnn_prob_gru_nu
    rnn = pop.config.genome.rnn_prob_simple_rnn
    msg = f"\n\n\n\n\n===> RUNNING EXPERIMENT 2 FOR THE FOLLOWING CONFIGURATION: <===" \
          f"\n\t> fitness:             {cfg.evaluation.fitness}" \
          f"\n\t> GRU enabled:         {gru > 0}  (probability={round(gru, 2)})" \
          f"\n\t> GRU-NR enabled:      {gru_nr > 0}  (probability={round(gru_nr, 2)})" \
          f"\n\t> GRU-NU enabled:      {gru_nu > 0}  (probability={round(gru_nu, 2)})" \
          f"\n\t> SRU enabled:         {rnn > 0}  (probability={round(rnn, 2)})" \
          f"\n\t> Saving under folder: {folder}\n"
    pop.log(msg)
    
    # Set games used for evaluation
    _, games_eval = get_game_ids(experiment_id=2)
    
    # Execute the requested segments
    try:
        # Evaluate the trained population
        evaluate(
                games=games_eval,
                population=pop,
                unused_cpu=unused_cpu,
        )
    except Exception as e:
        pop.log(traceback.format_exc(), print_result=False)
        raise e
    finally:
        process_killer('run_population.py')  # Close all the terminated files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--prob_gru', type=float, default=0)
    parser.add_argument('--prob_sru', type=float, default=0)
    parser.add_argument('--prob_lstm', type=float, default=0)
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--unused_cpu', type=int, default=2)
    args = parser.parse_args()
    
    main(
            fitness=D_DISTANCE,
            prob_gru=args.prob_gru,
            prob_sru=args.prob_sru,
            prob_lstm=args.prob_lstm,
            version=args.version,
            unused_cpu=args.unused_cpu,
    )
