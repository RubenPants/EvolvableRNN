"""
population_visualizer.py

Visualize the behaviour of a complete population.
"""
from math import cos, sin

import matplotlib.pyplot as plt

from population.utils.visualizing.elite_visualizer import main as elite_visualizer
from population.utils.visualizing.specie_distance import main as specie_distance
from population.utils.visualizing.specie_visualizer import main as specie_visualizer
from utils.dictionary import D_GAME_ID, D_POS
from utils.myutils import get_subfolder


def create_blueprints(final_observations: dict, games: list, gen: int, save_path: str):
    """
    Save images in the relative 'images/' subfolder of the population.

    :param final_observations: Dictionary of all the final game observations made
    :param games: List Game-objects used during evaluation
    :param gen: Population's current generation
    :param save_path: Path of 'images'-folder under which image must be saved
    """
    genome_keys = list(final_observations.keys())
    for g in games:
        # Get the game's blueprint
        g.get_blueprint()
        
        # Add arrow to indicate initial direction of robot
        x = g.player.init_pos[0]
        y = g.player.init_pos[1]
        dx = cos(g.player.init_angle)
        dy = sin(g.player.init_angle)
        plt.arrow(x, y, dx, dy, head_width=0.1, length_includes_head=True)
        
        # Get all the final positions of the agents
        positions = []
        for gk in genome_keys:
            positions += [fo[D_POS] for fo in final_observations[gk] if fo[D_GAME_ID] == g.id]
        
        # Plot the positions
        dot_x = [p[0] for p in positions]
        dot_y = [p[1] for p in positions]
        plt.plot(dot_x, dot_y, 'ro')
        
        # Add title
        plt.title(f"Blueprint - Game {g.id:05d} - Generation {gen:05d}")
        
        # Save figure
        game_path = get_subfolder(save_path, 'game{id:05d}'.format(id=g.id))
        plt.savefig(f'{game_path}blueprint_gen{gen:05d}')
        plt.close()


def create_traces(traces: dict, games: list, gen: int, save_path: str, save_name: str = 'trace'):
    """
    Save images in the relative 'images/' subfolder of the population.

    :param traces: Dictionary of all the traces
    :param games: List Game-objects used during evaluation
    :param gen: Population's current generation
    :param save_path: Path of 'images'-folder under which image must be saved
    :param save_name: Name of saved file
    """
    genome_keys = list(traces.keys())
    
    for i, g in enumerate(games):
        # Get the game's blueprint
        plt.figure(figsize=(2, 2))
        # g.get_blueprint(ax=plt.gca(), annotate=False)
        x_min, x_max = min(g.x_axis / 2, g.target.x), max(g.x_axis / 2, g.target.x)
        y_min, y_max = min(g.y_axis / 2, g.target.y), max(g.y_axis / 2, g.target.y)
        
        # Add green dotted circle around targets
        c = plt.Circle((g.target.x, g.target.y), 0.5, color='g', linestyle=':', linewidth=1.5, fill=False)
        plt.gca().add_artist(c)
        
        # Append the traces agent by agent
        for gk in genome_keys:
            # Get the trace of the genome for the requested game
            x_pos, y_pos = zip(*traces[gk][i])
            x_min, x_max = min(x_min, min(x_pos)), max(x_max, max(x_pos))
            y_min, y_max = min(y_min, min(y_pos)), max(y_max, max(y_pos))
            
            # Plot the trace (gradient)
            plt.plot([10], [10], marker='o', markersize=5, color=(1, 0, 0))
            for p in range(0, len(x_pos) - 1):
                plt.plot((x_pos[p], x_pos[p + 1]), (y_pos[p], y_pos[p + 1]), color=(1, p / len(x_pos), 0))

        # Replace ticks by stripes
        plt.xticks([i for i in range(g.x_axis + 1)])
        plt.yticks([i for i in range(g.y_axis + 1)])
        plt.setp(plt.gca().get_xticklabels(), visible=False)
        plt.setp(plt.gca().get_yticklabels(), visible=False)
        
        # Constraint the plot's boundaries
        x_center = (x_max - x_min) / 2 + x_min
        y_center = (y_max - y_min) / 2 + y_min
        r = max((x_max - x_min) / 2 + .5, (y_max - y_min) / 2 + .5)
        plt.xlim(x_center - r, x_center + r)
        plt.ylim(y_center - r, y_center + r)
        
        # Add title
        # plt.title(f"Game {g.id:05d}")
        
        # Save figure
        plt.tight_layout()
        game_path = get_subfolder(save_path, 'game{id:05d}'.format(id=g.id))
        # plt.savefig(f'{game_path}{save_name}_gen{gen:05d}', bbox_inches='tight', pad_inches=0.02)
        plt.savefig(f'game{i + 1}', bbox_inches='tight', pad_inches=0.02)  # Places in main folder
        plt.close()


def create_training_overview(pop):
    """Create overview-plots of a population's training history."""
    # Visualize the population elites and for each specie its elites
    elite_visualizer(
            pop=pop,
            show=False,
    )
    
    specie_visualizer(
            pop=pop,
            show=False,
    )
    
    # Visualize the specie-distance of the current species
    specie_distance(
            pop=pop,
            show=False,
    )
