"""
experiment3.py

Run the third experiment.
"""
import argparse
import traceback

from config import Config
from main import evaluate, get_folder, get_game_ids, get_name, train, training_overview, visualize_genome
from population.population import Population
from process_killer import main as process_killer
from utils.dictionary import *


def main(fitness,
         prob_gru: float,
         prob_sru: float,
         prob_lstm: float,
         train_iterations=0,
         version=0,
         unused_cpu=1,
         ):
    """
    Run a population's configuration.

    :param fitness: Fitness function used to evaluate the population
    :param prob_gru: Probability of mutating towards a GRU-node
    :param prob_sru: Probability of mutating towards a SRU-node
    :param prob_lstm: Probability of mutating towards a LSTM-node
    :param train_iterations: Number of training generations
    :param version: Version of the model
    :param unused_cpu: Number of CPUs not used during training
    """
    # Re-configure the config-file
    cfg = Config()
    cfg.bot.angular_dir = []
    cfg.bot.delta_dist_enabled = False
    cfg.bot.dist_enabled = True
    cfg.game.duration = 200  # 200 seconds should be enough to reach several targets
    cfg.population.pop_size = 512
    
    # Let inputs apply to configuration
    cfg.genome.rnn_prob_gru = prob_gru
    cfg.genome.rnn_prob_simple_rnn = prob_sru
    cfg.genome.rnn_prob_lstm = prob_lstm
    cfg.evaluation.fitness = fitness
    cfg.update()
    
    # Create the population
    folder = get_folder(experiment_id=3)
    name = get_name(cfg=cfg, version=version)
    pop = Population(
            name=name,
            config=cfg,
            folder_name=folder,
            use_backup=False,
    )
    
    # Give overview of population
    gru = cfg.genome.rnn_prob_gru
    sru = cfg.genome.rnn_prob_simple_rnn
    lstm = cfg.genome.rnn_prob_lstm
    msg = f"\n\n\n\n\n===> RUNNING EXPERIMENT 3 FOR THE FOLLOWING CONFIGURATION: <===" \
          f"\n\t> fitness:             {cfg.evaluation.fitness}" \
          f"\n\t> GRU enabled:         {gru > 0}  (probability={round(gru, 2)})" \
          f"\n\t> SRU enabled:         {sru > 0}  (probability={round(sru, 2)})" \
          f"\n\t> LSTM enabled:        {lstm > 0}  (probability={round(lstm, 2)})" \
          f"\n\t> Saving under folder: {folder}" \
          f"\n\t> Training iterations: {train_iterations}\n"
    pop.log(msg)
    
    # Set games used for evaluation
    games_train, games_eval = get_game_ids(experiment_id=3)
    
    # Execute the requested segments
    try:
        train(
                debug=False,
                games=games_train,
                iterations=train_iterations,
                population=pop,
                unused_cpu=unused_cpu,
        )
        
        # Evaluate the trained population
        evaluate(
                games=games_eval,
                population=pop,
                unused_cpu=unused_cpu,
        )
        training_overview(
                population=pop,
        )
        visualize_genome(
                genome=pop.best_genome,
                population=pop,
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
    parser.add_argument('--iterations', type=int, default=0)
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--unused_cpu', type=int, default=2)
    args = parser.parse_args()
    
    main(
            fitness=D_DISTANCE_SCORE,
            prob_gru=args.prob_gru,
            prob_sru=args.prob_sru,
            prob_lstm=args.prob_lstm,
            train_iterations=args.iterations,
            version=args.version,
            unused_cpu=args.unused_cpu,
    )
