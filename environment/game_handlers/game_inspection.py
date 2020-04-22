"""
inspect_games.py

Check if all games are created correctly.
"""
import os

from config import Config
from environment.game import Game

if __name__ == '__main__':
    os.chdir("../..")
    config = Config()
    for g_id in [1, 2, 3]:
        try:
            game = Game(
                    game_id=g_id,
                    config=config,
                    save_path="environment/games_db/",
                    overwrite=False,
                    silent=True,
            )
            game.close()
            game.reset()
            game.get_blueprint()
            game.get_observation()
            game.step(0, 0)
        except Exception:
            print(f"Bug in game: {g_id}, please manually redo this one")
