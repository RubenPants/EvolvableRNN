"""
monitor_genome.py

Shameless copy paste from monitor_genome_single_gru.
"""
import argparse
import os

import numpy as np

from config import Config
from environment.game import get_game
from genomes_gru.persist import load_genome
from population.utils.gene_util.gru import GruNodeGene
from population.utils.genome import Genome
from population.utils.network_util.feed_forward_net import make_net
from population.utils.visualizing.monitor_genome_single_gru import get_gru_states, merge, visualize_actuation, \
    visualize_candidate_hidden_state, visualize_distance, visualize_hidden_state, visualize_position, \
    visualize_reset_gate, visualize_update_gate
from utils.dictionary import *
from utils.myutils import get_subfolder


def main(genome: Genome, gid: int, duration: int = 60, debug: bool = False):
    # Check if valid genome (contains at least one hidden GRU, first GRU is monitored)
    assert len([n for n in genome.get_used_nodes().values() if type(n) == GruNodeGene]) >= 1
    
    # Get the game
    cfg = Config()
    cfg.game.duration = duration
    cfg.update()
    game = get_game(gid, cfg=cfg, noise=False)
    state = game.reset()[D_SENSOR_LIST]
    step_num = 0
    
    # Create the network
    net = make_net(genome=genome,
                   genome_config=cfg.genome,
                   batch_size=1,
                   initial_read=state,
                   )
    
    # Containers to monitor
    actuation = []
    distance = []
    position = []
    Ht = []
    Ht_tilde = []
    Rt = []
    Zt = []
    target_found = []
    score = 0
    
    # Initialize the containers
    actuation.append([0, 0])
    distance.append(state[0])
    position.append(game.player.pos.get_tuple())
    ht, ht_tilde, rt, zt = get_gru_states(net=net, x=np.asarray([state]))  # TODO: Has updated!
    Ht.append(ht)
    Ht_tilde.append(ht_tilde)
    Rt.append(rt)
    Zt.append(zt)
    if debug:
        print(f"Step: {step_num}")
        print(f"\t> Actuation: {(round(actuation[-1][0], 5), round(actuation[-1][1], 5))!r}")
        print(f"\t> Distance: {round(distance[-1], 5)}")
        print(f"\t> Position: {(round(position[-1][0], 2), round(position[-1][1], 2))!r}")
        print(f"\t> GRU states: "
              f"\t\tHt={round(Ht[-1], 5)}"
              f"\t\tHt_tilde={round(Ht[-1], 5)}"
              f"\t\tRt={round(Rt[-1], 5)}"
              f"\t\tZt={round(Zt[-1], 5)}")
    
    # Start monitoring
    while True:
        # Check if maximum iterations is reached
        if step_num == duration * cfg.game.fps: break
        
        # Determine the actions made by the agent for each of the states
        action = net(np.asarray([state]))
        
        # Check if each game received an action
        assert len(action) == 1
        
        # Proceed the game with one step, based on the predicted action
        obs = game.step(l=action[0][0], r=action[0][1])
        finished = obs[D_DONE]
        
        # Update the score-count
        if game.score > score:
            target_found.append(step_num)
            score = game.score
        
        # Update the candidate's current state
        state = obs[D_SENSOR_LIST]
        
        # Stop if agent reached target in all the games
        if finished: break
        step_num += 1
        
        # Update the containers
        actuation.append(action[0])
        distance.append(state[0])
        position.append(game.player.pos.get_tuple())
        ht, ht_tilde, rt, zt = get_gru_states(net=net, x=np.asarray([state]))
        Ht.append(ht)
        Ht_tilde.append(ht_tilde)
        Rt.append(rt)
        Zt.append(zt)
        if debug:
            print(f"Step: {step_num}")
            print(f"\t> Actuation: {(round(actuation[-1][0], 5), round(actuation[-1][1], 5))!r}")
            print(f"\t> Distance: {round(distance[-1], 5)}")
            print(f"\t> Position: {(round(position[-1][0], 2), round(position[-1][1], 2))!r}")
            print(f"\t> GRU states: "
                  f"\t\tHt={round(Ht[-1], 5)}"
                  f"\t\tHt_tilde={round(Ht[-1], 5)}"
                  f"\t\tRt={round(Rt[-1], 5)}"
                  f"\t\tZt={round(Zt[-1], 5)}")
    
    # Visualize the monitored values
    path = get_subfolder('genomes_gru/', 'images')
    path = get_subfolder(path, f'genome{genome.key}')
    path = get_subfolder(path, 'monitor')
    visualize_actuation(actuation,
                        target_found=target_found,
                        game_cfg=cfg.game,
                        save_path=f"{path}actuation.png")
    visualize_distance(distance,
                       target_found=target_found,
                       game_cfg=cfg.game,
                       save_path=f"{path}distance.png")
    visualize_hidden_state(Ht,
                           target_found=target_found,
                           game_cfg=cfg.game,
                           save_path=f"{path}hidden_state.png")
    visualize_candidate_hidden_state(Ht_tilde,
                                     target_found=target_found,
                                     game_cfg=cfg.game,
                                     save_path=f"{path}candidate_hidden_state.png")
    visualize_reset_gate(Rt,
                         target_found=target_found,
                         game_cfg=cfg.game,
                         save_path=f"{path}reset_gate.png")
    visualize_update_gate(Zt,
                          target_found=target_found,
                          game_cfg=cfg.game,
                          save_path=f"{path}update_gate.png")
    visualize_position(position,
                       game=game,
                       save_path=f"{path}trace.png")
    merge(f"Monitored genome={genome.key} on game={game.id}", path=path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--duration', type=int, default=25)  # Simulation duration
    parser.add_argument('--gid', type=int, default=60001)  # First evaluation game of experiment3
    parser.add_argument('--name', type=str, default='genome2')
    args = parser.parse_args()
    
    # Go back to root
    os.chdir("..")
    
    # Load in the genome
    g = load_genome(
            g_name=args.name,
    )
    
    # Perform the script
    main(
            genome=g,
            gid=args.gid,
            duration=args.duration,
    )
