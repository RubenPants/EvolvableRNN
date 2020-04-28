"""
game_creator.py

Create games used during the experiments.
"""
import argparse
import os
from math import cos, pi, sin

import matplotlib.pyplot as plt

from config import Config
from environment.game import Game
from environment.robot import Robot
from environment.spawn_functions.random import SpawnRandom
from environment.spawn_functions.range import SpawnRange
from environment.spawn_functions.simple import SpawnSimple
from utils.vec2d import Vec2d


def get_shared_config():
    cfg = Config()
    cfg.game.x_axis = 20
    cfg.game.y_axis = 20
    cfg.update()
    return cfg


def get_game_id(i: int, experiment_id: int):
    return int(experiment_id * 1e4 + i)


def check_and_save_game(g, show: bool = True, randomize: bool = True):
    """Test if the given game is implemented correctly. If so, then save it."""
    g.sample_target()
    g.close()
    g.reset()
    if randomize: g.randomize()
    g.get_distance_to_target()
    g.get_blueprint()
    g.get_observation()
    g.step(1, 1)
    plt.close()
    
    # Save the final game
    g.save()
    
    # Show the result of the game
    if show:
        plt.figure()
        g.get_blueprint(show_player=True)
        plt.show()
        plt.close()


def create_dummy_game(i: int = 0, overwrite: bool = True, show: bool = True):
    cfg = get_shared_config()
    ROBOT_INIT_POS = Vec2d(cfg.game.x_axis / 2, cfg.game.y_axis / 2)  # Initial position of the drone
    spawn_f = SpawnSimple(game_config=cfg, train=True)
    spawn_f.add_location((1, 1))
    game = Game(
            config=cfg,
            game_id=get_game_id(i, experiment_id=0),
            overwrite=overwrite,
            spawn_func=spawn_f,
    )
    game.player = Robot(game=game)
    game.set_player_init_pos(p=ROBOT_INIT_POS)
    check_and_save_game(game, show=show)


def create_experiment_1(overwrite: bool = True, show: bool = True):
    """
    Experiment 1 tests the generalization capabilities of the genomes based on the relative direction the target is
     placed.
    
    TRAINING
     Training is done in a maze where the robot is in the center, and the targets are positioned around the agent. Each
     run has a single target sampled. The targets all are positioned 6 meters from the robot's initial position.
    
    TESTING
     Testing is done similar as was the case for training, however, in this scenario only 20 targets are sampled (where
     there were 36 for training). Only 4 of these 20 sampled target positions overlap with the training set.
    """
    cfg = get_shared_config()
    
    # Fixed parameters
    ROBOT_INIT_ANGLE = pi / 2  # Looking up
    ROBOT_INIT_POS = Vec2d(cfg.game.x_axis / 2, cfg.game.y_axis / 2)  # Initial position of the drone
    
    # Create the training game
    spawn_f_train = SpawnSimple(game_config=cfg, train=True)
    for i in range(0, 360, 10):  # Positions circular in hops of 10 degree
        angle = i / 180 * pi
        offset_x = 6 * cos(angle)
        offset_y = 6 * sin(angle)
        spawn_f_train.add_location((ROBOT_INIT_POS[0] + offset_x, ROBOT_INIT_POS[1] + offset_y))
    game = Game(
            config=cfg,
            player_noise=0,
            game_id=get_game_id(0, experiment_id=1),
            overwrite=overwrite,
            spawn_func=spawn_f_train,
            stop_if_reached=True,
            wall_bound=False,
    )
    game.player = Robot(game=game)
    game.set_player_init_angle(a=ROBOT_INIT_ANGLE)
    game.set_player_init_pos(p=ROBOT_INIT_POS)
    check_and_save_game(game, show=show)
    
    # Create the evaluation games, each of those only contains one target
    for i in range(1, 21):
        spawn_f_eval = SpawnSimple(game_config=cfg, train=False)
        angle = i / 10 * pi
        offset_x = 6 * cos(angle)
        offset_y = 6 * sin(angle)
        spawn_f_eval.add_location((ROBOT_INIT_POS[0] + offset_x, ROBOT_INIT_POS[1] + offset_y))
        game = Game(
                config=cfg,
                player_noise=0,
                game_id=get_game_id(i, experiment_id=1),
                overwrite=overwrite,
                spawn_func=spawn_f_eval,
                stop_if_reached=True,
                wall_bound=False,
        )
        game.player = Robot(game=game)
        game.set_player_init_angle(a=ROBOT_INIT_ANGLE)
        game.set_player_init_pos(p=ROBOT_INIT_POS)
        check_and_save_game(game, show=show, randomize=False)


def create_experiment_2(overwrite: bool = True, show: bool = True):
    """
    Experiment 2 tests the generalization capabilities of the genomes based on their initial distance from the target.
    
    TRAINING
     Training is done in a maze where the robot is in the center, and the targets are positioned around the agent. Each
     run has a single target sampled. The targets all are positioned 6 meters from the robot's initial position.
    
    TESTING
     Testing is done similar as was the case for training. For testing, half of the training's initial target-
     orientations are used (18 instead of 36). However, each orientation now has two targets, both with a different
     initial position from the robot's starting position. An inner-ring has distance 4, where the outer-ring has a
     distance of 8.
    """
    cfg = get_shared_config()
    
    # Fixed parameters
    ROBOT_INIT_ANGLE = pi / 2  # Looking up
    ROBOT_INIT_POS = Vec2d(cfg.game.x_axis / 2, cfg.game.y_axis / 2)  # Initial position of the drone
    
    # Create the training game
    spawn_f_train = SpawnSimple(game_config=cfg, train=True)
    for i in range(0, 360, 10):  # Positions circular in hops of 10 degree
        angle = i / 180 * pi
        offset_x = 6 * cos(angle)
        offset_y = 6 * sin(angle)
        spawn_f_train.add_location((ROBOT_INIT_POS[0] + offset_x, ROBOT_INIT_POS[1] + offset_y))
    game = Game(
            config=cfg,
            player_noise=0,
            game_id=get_game_id(0, experiment_id=2),
            overwrite=overwrite,
            spawn_func=spawn_f_train,
            stop_if_reached=True,
            wall_bound=False,
    )
    game.player = Robot(game=game)
    game.set_player_init_angle(a=ROBOT_INIT_ANGLE)
    game.set_player_init_pos(p=ROBOT_INIT_POS)
    check_and_save_game(game, show=show)
    
    # Create the inner circle of the evaluation games, each of those only contains one target
    for i in range(1, 19):
        spawn_f_eval = SpawnSimple(game_config=cfg, train=False)
        angle = i / 9 * pi  # Hops of 20°
        offset_x = 4 * cos(angle)
        offset_y = 4 * sin(angle)
        spawn_f_eval.add_location((ROBOT_INIT_POS[0] + offset_x, ROBOT_INIT_POS[1] + offset_y))
        game = Game(
                config=cfg,
                player_noise=0,
                game_id=get_game_id(i, experiment_id=2),
                overwrite=overwrite,
                spawn_func=spawn_f_eval,
                stop_if_reached=True,
                wall_bound=False,
        )
        game.player = Robot(game=game)
        game.set_player_init_angle(a=ROBOT_INIT_ANGLE)
        game.set_player_init_pos(p=ROBOT_INIT_POS)
        check_and_save_game(game, show=show, randomize=False)
    
    # Create the outer circle of the evaluation games, each of those only contains one target
    for i in range(1, 19):
        spawn_f_eval = SpawnSimple(game_config=cfg, train=False)
        angle = i / 9 * pi  # Hops of 20°
        offset_x = 8 * cos(angle)
        offset_y = 8 * sin(angle)
        spawn_f_eval.add_location((ROBOT_INIT_POS[0] + offset_x, ROBOT_INIT_POS[1] + offset_y))
        game = Game(
                config=cfg,
                player_noise=0,
                game_id=get_game_id(i + 100, experiment_id=2),
                overwrite=overwrite,
                spawn_func=spawn_f_eval,
                stop_if_reached=True,
                wall_bound=False,
        )
        game.player = Robot(game=game)
        game.set_player_init_angle(a=ROBOT_INIT_ANGLE)
        game.set_player_init_pos(p=ROBOT_INIT_POS)
        check_and_save_game(game, show=show, randomize=False)


def create_experiment_3(overwrite: bool = True, show: bool = True):
    """
    Experiment 3 tests both the generalization capabilities in adapting to newly spawned targets. In this scenario, new
     targets are spawned ones the previous are found, with the goal to find as many as possible.
    
    TRAINING
     During training, new targets are spawned based on the agent's current position. Each target will lay between 4 and
     8 meters from the agent and will always be positioned inside of the maze.
    
    TESTING
     Testing is done similar as was the case for training, with the main difference that the positions of the targets
     are defined beforehand, this to ensure a fair comparison between the different evaluations. 10 targets are created,
     with the idea that it would be impossible for the drone to ever reach this many targets.
    """
    cfg = get_shared_config()
    
    # Fixed parameters
    ROBOT_INIT_ANGLE = pi / 2  # Looking up
    ROBOT_INIT_POS = Vec2d(cfg.game.x_axis / 2, cfg.game.y_axis / 2)  # Initial position of the drone
    
    # Create the training game
    spawn_f_train = SpawnRandom(game_config=cfg.game, train=True)
    game = Game(
            config=cfg,
            player_noise=0,
            game_id=get_game_id(0, experiment_id=3),
            overwrite=overwrite,
            spawn_func=spawn_f_train,
            stop_if_reached=False,
            wall_bound=False,
    )
    game.player = Robot(game=game)
    game.set_player_init_angle(a=ROBOT_INIT_ANGLE)
    game.set_player_init_pos(p=ROBOT_INIT_POS)
    check_and_save_game(game, show=show)
    
    # Create 20 evaluation games
    for i in range(1, 21):
        spawn_f_eval = SpawnRandom(game_config=cfg.game, train=False)
        game = Game(
                config=cfg,
                player_noise=0,
                game_id=get_game_id(i, experiment_id=3),
                overwrite=overwrite,
                spawn_func=spawn_f_eval,
                stop_if_reached=False,
                wall_bound=False,
        )
        game.player = Robot(game=game)
        game.set_player_init_angle(a=ROBOT_INIT_ANGLE)
        game.set_player_init_pos(p=ROBOT_INIT_POS)
        check_and_save_game(game, show=show, randomize=False)


def create_experiment_6(overwrite: bool = True, show: bool = True):
    """
    Experiment 6 is another 'simplified' simulation in which the agents are trained on a harder environment than they
     are tested on. The reason why is to keep a lower threshold for the 'solution' genomes (only interested in those).
    
    TRAINING
     During training, new targets are spawned based on the agent's initial position. Each target will lay between 2 and
     10 meters from the agent and will always be positioned inside of the maze.
    
    TESTING
     Testing is done on 18 predefined targets position in a relative angle from the agent of k*20° and a distance of
     either 4, 6, or 8 meters (sequential alternating).
    """
    cfg = get_shared_config()
    
    # Fixed parameters
    ROBOT_INIT_ANGLE = pi / 2  # Looking up
    ROBOT_INIT_POS = Vec2d(cfg.game.x_axis / 2, cfg.game.y_axis / 2)  # Initial position of the drone
    
    # Create the training game
    spawn_f_train = SpawnRange(game_config=cfg.game, train=True, r_max=10, r_min=2)
    game = Game(
            config=cfg,
            player_noise=0,
            game_id=get_game_id(0, experiment_id=6),
            overwrite=overwrite,
            spawn_func=spawn_f_train,
            stop_if_reached=True,
            wall_bound=False,
    )
    game.player = Robot(game=game)
    game.set_player_init_angle(a=ROBOT_INIT_ANGLE)
    game.set_player_init_pos(p=ROBOT_INIT_POS)
    check_and_save_game(game, show=show)
    
    # Create 18 evaluation games
    for i in range(1, 19):
        spawn_f_eval = SpawnSimple(game_config=cfg.game, train=False)
        angle = i / 9 * pi  # Hops of 20°
        d = 4 if i % 3 == 0 else 6 if i % 3 == 1 else 8
        offset_x = d * cos(angle)
        offset_y = d * sin(angle)
        spawn_f_eval.add_location((ROBOT_INIT_POS[0] + offset_x, ROBOT_INIT_POS[1] + offset_y))
        game = Game(
                config=cfg,
                player_noise=0,
                game_id=get_game_id(i, experiment_id=6),
                overwrite=overwrite,
                spawn_func=spawn_f_eval,
                stop_if_reached=True,
                wall_bound=False,
        )
        game.player = Robot(game=game)
        game.set_player_init_angle(a=ROBOT_INIT_ANGLE)
        game.set_player_init_pos(p=ROBOT_INIT_POS)
        check_and_save_game(game, show=show, randomize=False)


if __name__ == '__main__':
    """
    Create game, option to choose from custom or auto-generated.
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--overwrite', type=bool, default=True)
    parser.add_argument('--show', type=int, default=False)
    args = parser.parse_args()
    
    # Point back to root
    os.chdir('../../')
    
    # Create the experiments
    # for i in [0, -1, -2, -3, -4]:
    #     create_dummy_game(i=i, overwrite=args.overwrite, show=args.show)
    # create_experiment_1(overwrite=args.overwrite, show=args.show)
    # create_experiment_2(overwrite=args.overwrite, show=args.show)
    # create_experiment_3(overwrite=args.overwrite, show=args.show)
    create_experiment_6(overwrite=args.overwrite, show=args.show)
