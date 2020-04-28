"""
game_visualizer.py

This method is purely used to visualize the game-maps (used in the thesis). This file must be called in the environment
folder to make sure the visualizations are saved properly.
"""
import os

import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config
from environment.game import get_game
from main import get_game_ids


def game_blueprint(game, show: bool = False):
    """
    Create the blueprint of the game.
    
    :param game: Game object
    :param show: Show the result
    """
    # Get game's blueprint
    game.get_blueprint(show_player=True)
    
    plt.title("Blueprint - Game {id:05d}".format(id=game.id))
    plt.savefig(f'environment/visualizations/blueprint_game{game.id:05d}')
    if show: plt.show()
    plt.close()


if __name__ == '__main__':
    os.chdir("../..")
    cfg = Config()
    games = [0]  # Game Dummy
    
    # Experiment 1
    exp1_train, exp1_eval = get_game_ids(1)
    games += exp1_train[:1]  # Repeats itself
    games += exp1_eval
    
    # Experiment 2
    exp2_train, exp2_eval = get_game_ids(2)
    games += exp2_train[:1]  # Repeats itself
    games += exp2_eval
    
    # Experiment 3
    exp3_train, exp3_eval = get_game_ids(3)
    games += exp3_train[:1]  # Repeats itself
    games += exp3_eval
    
    # Experiment 6
    exp6_train, exp6_eval = get_game_ids(6)
    games += exp6_train[:1]  # Repeats itself
    games += exp6_eval
    
    # Create the visualizations
    for g_id in tqdm(games):
        g = get_game(g_id, cfg=cfg)
        game_blueprint(g, show=False)
