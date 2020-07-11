"""
main.py

All the possible method that can be queried on the population.
"""
import argparse
import traceback
from copy import deepcopy

from config import Config
from population.population import Population
from population.utils.gene_util.fixed_rnn import FixedRnnNodeGene
from population.utils.gene_util.gru import GruNodeGene
from population.utils.gene_util.gru_no_reset import GruNoResetNodeGene
from population.utils.gene_util.lstm import LstmNodeGene
from population.utils.gene_util.output_node import OutputNodeGene
from population.utils.gene_util.simple_node import SimpleNodeGene
from population.utils.gene_util.simple_rnn import SimpleRnnNodeGene
from population.utils.genome import Genome
from process_killer import main as process_killer


def blueprint(population: Population,
              games: list,
              debug: bool = False,
              duration: int = 0,
              unused_cpu: int = 0,
              ):
    """Create a blueprint evaluation for the given population on the first 5 games."""
    from environment.env_visualizing import VisualizingEnv
    
    population.log("\n===> CREATING BLUEPRINTS <===\n")
    population.log(f"Creating blueprints for games: {games}")
    
    game_config = deepcopy(population.config)
    if duration > 0: game_config.game.duration = duration
    
    visualizer = VisualizingEnv(
            game_config=game_config,
            games=games,
            unused_cpu=unused_cpu,
    )
    visualizer.blueprint_genomes(
            pop=population,
            parallel=not debug,
    )


def evaluate(population: Population,
             games: list,
             genomes: list = None,
             debug: bool = False,
             duration: int = 0,
             overwrite: bool = False,
             unused_cpu: int = 0,
             ):
    """Evaluate the given population on the evaluation game-set."""
    from environment.env_evaluation import EvaluationEnv
    
    population.log("\n===> EVALUATING <===\n")
    
    game_config = deepcopy(population.config)
    if duration > 0: game_config.game.duration = duration
    
    evaluator = EvaluationEnv(
            game_config=game_config,
            games=games,
            unused_cpu=unused_cpu,
    )
    if genomes is None:
        genomes = sorted(
                [g for g in population.population.values()],
                key=lambda x: x.fitness if x.fitness else 0,
                reverse=True,
        )[:max(int(population.config.population.parent_selection * len(population.population)), 1)]
    evaluator.evaluate_genome_list(
            genome_list=genomes,  # Evaluate the ten best performing genomes
            pop=population,
            parallel=not debug,
            overwrite=overwrite,
    )


def monitor(game_id: int,
            population: Population,
            debug: bool = False,
            duration: int = 0,
            genome: Genome = None,
            ):
    """Monitor a single run of the given genome that contains a single GRU-node."""
    print("\n===> MONITORING GENOME <===\n")
    if genome is None: genome = population.best_genome
    
    game_config = deepcopy(population.config)
    if duration > 0: game_config.game.duration = duration
    
    # Take first GRU or SRU node
    node_type = None
    for n in genome.get_used_nodes().values():
        t = type(n)
        if t != OutputNodeGene and t != SimpleNodeGene:
            node_type = t
            break
    if node_type is None:
        raise Exception(f"No hidden node to monitor in genome {genome}")
    
    if node_type == GruNodeGene:
        from population.utils.visualizing.monitor_genome_single_gru import main as gru_monitor
        gru_monitor(
                population=population,
                game_id=game_id,
                genome=genome,
                game_cfg=game_config,
                debug=debug,
        )
    elif node_type == GruNoResetNodeGene:
        from population.utils.visualizing.monitor_genome_single_gru_nr import main as gru_nr_monitor
        gru_nr_monitor(
                population=population,
                game_id=game_id,
                genome=genome,
                game_cfg=game_config,
                debug=debug,
        )
    elif node_type == SimpleRnnNodeGene or node_type == FixedRnnNodeGene:
        from population.utils.visualizing.monitor_genome_single_sru import main as sru_monitor
        sru_monitor(
                average=2,
                population=population,
                game_id=game_id,
                genome=genome,
                game_cfg=game_config,
                debug=debug,
        )
    elif node_type == LstmNodeGene:
        from population.utils.visualizing.monitor_genome_single_lstm import main as lstm_monitor
        lstm_monitor(
                population=population,
                game_id=game_id,
                genome=genome,
                game_cfg=game_config,
                debug=debug,
        )
    else:
        raise Exception(f"Not able to monitor the genome of config:\n{genome}")


def gru_analysis(population: Population,
                 d: int = 10,
                 range_width: int = 20,
                 genome: Genome = None,
                 unused_cpu: int = 2,
                 experiment_id: int = 1,
                 overwrite: bool = False,
                 ):
    """Perform an in-depth analysis on a single-hidden single-GRU genome."""
    from population.utils.visualizing.gru_analysis import main as gru_analysis_main
    
    print("\n===> CREATING GRU ANALYSIS <===\n")
    
    gru_analysis_main(
            pop=population,
            d=d,
            range_width=range_width,
            genome=genome,
            cpu=unused_cpu,
            experiment_id=experiment_id,
            overwrite=overwrite
    )


def live(game_id: int,
         population: Population,
         genome: Genome,
         debug: bool = False,
         duration: int = 0,
         speedup: float = 3,
         ):
    """Create a live visualization for the performance of the given genome."""
    from environment.env_visualizing_live import LiveVisualizer
    
    print("\n===> STARTING LIVE DEMO <===\n")
    print(f"Genome {genome.key} with size: {genome.size()}")
    
    game_config = deepcopy(population.config)
    if duration > 0: game_config.game.duration = duration
    
    visualizer = LiveVisualizer(
            pop=population,
            game_config=game_config,
            debug=debug,
            speedup=speedup,
    )
    visualizer.visualize(
            genome=genome,
            game_id=game_id,
    )


def trace(population: Population,
          games: list,
          debug: bool = False,
          duration: int = 0,
          unused_cpu: int = 0,
          ):
    """Create a trace evaluation for the given population on the provided games."""
    from environment.env_visualizing import VisualizingEnv
    
    population.log("\n===> CREATING TRACES <===\n")
    population.log(f"Creating traces for games: {games}")
    
    game_config = deepcopy(population.config)
    if duration > 0: game_config.game.duration = duration
    
    visualizer = VisualizingEnv(
            game_config=game_config,
            games=games,
            unused_cpu=unused_cpu,
    )
    visualizer.trace_genomes(
            pop=population,
            parallel=not debug
    )


def trace_most_fit(population: Population,
                   genome: Genome,
                   games: list,
                   debug: bool = False,
                   duration: int = 0,
                   unused_cpu: int = 0,
                   ):
    """Create a trace evaluation for the given genome on the provided games."""
    from environment.env_visualizing import VisualizingEnv
    
    game_config = deepcopy(population.config)
    if duration > 0: game_config.game.duration = duration
    
    population.log("\n===> CREATING GENOME TRACE <===\n")
    population.log(f"Creating traces for games: {games}")
    
    visualizer = VisualizingEnv(
            game_config=game_config,
            games=games,
            unused_cpu=unused_cpu,
    )
    for g in games:  # TODO: Bug in warm-up of network if multiple games evaluated
        visualizer.set_games([g])
        visualizer.trace_genomes(
                pop=population,
                given_genome=genome,
                parallel=not debug,
        )


def train(population: Population,
          games: list,
          iterations: int,
          debug: bool = False,
          duration: int = 0,
          unused_cpu: int = 0,
          ):
    """Train the population on the requested number of iterations."""
    from environment.env_training import TrainingEnv
    
    population.log("\n===> TRAINING <===\n")
    
    game_config = deepcopy(population.config)
    if duration > 0: game_config.game.duration = duration
    
    trainer = TrainingEnv(
            unused_cpu=unused_cpu,  # Use two cores less to keep laptop usable
            game_config=game_config,
            games=games,
    )
    trainer.evaluate_and_evolve(
            pop=population,
            n=iterations,
            parallel=not debug,
            save_interval=1 if debug else 10,
    )


def training_overview(population: Population,
                      ):
    """Give an overview of the population's training process."""
    from population.utils.visualizing.population_visualizer import create_training_overview
    
    print("\n===> CREATING TRAINING OVERVIEW <===\n")
    
    create_training_overview(
            pop=population,
    )


def visualize_genome(population: Population,
                     genome: Genome,
                     debug: bool = True,
                     show: bool = False,
                     ):
    """Visualize the requested genome."""
    print("\n===> VISUALIZING GENOME <===\n")
    print(f"Genome {genome.key} with size: {genome.size()}")
    
    population.visualize_genome(
            debug=debug,
            genome=genome,
            show=show,
    )


def get_game_ids(experiment_id):
    """
    Get the games (train, eval) belonging to the given experiment ID. Training games will be randomized, where the
    evaluation games will not.
    """
    if experiment_id in [1]:
        return [10000, ] * 10, \
               [10000 + i for i in range(1, 21)]
    elif experiment_id in [2]:
        return [10000, ] * 10, \
               [20000 + i for i in range(1, 19)] + [20100 + i for i in range(1, 19)]
    elif experiment_id in [3, 8]:
        return [30000, ] * 10, \
               [30000 + i for i in range(1, 21)]
    elif experiment_id in [4]:  # Combines experiment1&2
        return [10000, ] * 10, \
               [10000 + i for i in range(1, 21)] + \
               [20000 + i for i in range(1, 19)] + \
               [20100 + i for i in range(1, 19)]
    elif experiment_id in [6, 7]:
        return [60000, ] * 10, \
               [60000 + i for i in range(1, 19)]
    else:
        raise Exception(f"Experiment of ID {experiment_id} is not supported")


def get_name(cfg: Config, version: int = 0):
    """
    Get the folder named based on the use of the following:
     * GRU cells
     * Recurrent connections
     * Sexual reproduction
     
    :param cfg: Config object representing how the population is configured
    :param version: Version of the population
    """
    return f"{'angular ' if len(cfg.bot.angular_dir) > 0 else ''}" \
           f"{'delta ' if cfg.bot.delta_dist_enabled else ''}" \
           f"NEAT" \
           f"{'-LSTM' if cfg.genome.rnn_prob_lstm > 0 else ''}" \
           f"{'-GRU' if cfg.genome.rnn_prob_gru > 0 else ''}" \
           f"{'-GRU-NR' if cfg.genome.rnn_prob_gru_nr > 0 else ''}" \
           f"{'-GRU-NU' if cfg.genome.rnn_prob_gru_nu > 0 else ''}" \
           f"{'-SRU' if cfg.genome.rnn_prob_simple_rnn > 0 else ''}" \
           f"/v{version}"  # Make subfolder for each version!


def get_folder(experiment_id: int):
    """Get the folder-name based on the experiment's ID."""
    assert type(experiment_id) == int
    return f"experiment{experiment_id}"


# TODO: Start of main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    
    # Main methods
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--train_overview', type=bool, default=False)
    parser.add_argument('--blueprint', type=bool, default=False)
    parser.add_argument('--trace', type=bool, default=True)  # Keep it False
    parser.add_argument('--trace_fit', type=bool, default=True)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--genome', type=bool, default=False)
    parser.add_argument('--monitor', type=bool, default=False)
    parser.add_argument('--gru_analysis', type=bool, default=False)
    parser.add_argument('--live', type=bool, default=False)
    
    # Extra arguments
    parser.add_argument('--iterations', type=int, default=0)
    parser.add_argument('--experiment', type=int, default=1)
    parser.add_argument('--unused_cpu', type=int, default=2)
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--duration', type=int, default=60)
    parser.add_argument('--use_backup', type=bool, default=False)
    args = parser.parse_args()
    
    # Setup the population
    pop = Population(
            name='NEAT-GRU/example',
            folder_name=get_folder(args.experiment),
            use_backup=args.use_backup,
    )
    
    game_ids_train, game_ids_eval = get_game_ids(experiment_id=args.experiment)
    
    # Potentially modify the population
    if not pop.best_genome: pop.best_genome = list(pop.population.values())[0]
    
    # Chosen genome used for genome-evaluation
    chosen_genome = None
    # chosen_genome = pop.population[47280]
    
    try:
        if args.train:
            train(
                    debug=args.debug,
                    duration=args.duration,
                    games=game_ids_train,
                    iterations=args.iterations,
                    population=pop,
                    unused_cpu=args.unused_cpu,
            )
        
        if args.train_overview:
            training_overview(
                    population=pop
            )
        
        if args.blueprint:
            blueprint(
                    debug=args.debug,
                    duration=args.duration,
                    games=game_ids_eval,
                    population=pop,
                    unused_cpu=args.unused_cpu,
            )
        
        if args.trace:
            trace(
                    debug=args.debug,
                    duration=args.duration,
                    games=[game_ids_eval[0]],
                    population=pop,
                    unused_cpu=args.unused_cpu,
            )
        
        if args.trace_fit:
            trace_most_fit(
                    debug=args.debug,
                    duration=args.duration,
                    games=[game_ids_eval[0]],
                    # games=[-11],
                    genome=chosen_genome if chosen_genome else pop.best_genome,
                    population=pop,
                    unused_cpu=args.unused_cpu,
            )
        
        if args.monitor:
            monitor(
                    debug=args.debug,
                    duration=args.duration,
                    game_id=game_ids_eval[0],
                    genome=chosen_genome,
                    population=pop,
            )
        
        if args.evaluate:
            evaluate(
                    debug=args.debug,
                    duration=args.duration,
                    games=game_ids_eval,
                    population=pop,
                    unused_cpu=args.unused_cpu,
            )
        
        if args.genome:
            visualize_genome(
                    debug=True,
                    genome=chosen_genome if chosen_genome else pop.best_genome,
                    population=pop,
            )
        
        if args.gru_analysis:
            gru_analysis(
                    experiment_id=args.experiment,
                    genome=chosen_genome,
                    population=pop,
                    unused_cpu=args.unused_cpu,
            )
        
        if args.live:
            live(
                    debug=args.debug,
                    game_id=game_ids_eval[0],
                    # game_id=30001,
                    duration=args.duration,
                    genome=chosen_genome if chosen_genome else pop.best_genome,
                    population=pop,
                    speedup=4,
            )
    except Exception as e:
        pop.log(traceback.format_exc(), print_result=False)
        raise e
    finally:
        process_killer('main.py')  # Close all the terminated files
