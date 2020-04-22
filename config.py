"""
config.py

Class containing all the used configurations.
"""
from configs.base_config import BaseConfig
from configs.bot_config import BotConfig
from configs.evaluation_config import EvaluationConfig
from configs.game_config import GameConfig
from configs.genome_config import GenomeConfig
from configs.population_config import PopulationConfig


class Config(BaseConfig):
    """The main config-file tossed around, which is a container for all the other config-files."""
    
    __slots__ = {
        'bot', 'evaluation', 'game', 'genome', 'population',
    }
    
    def __init__(self):
        self.bot: BotConfig = BotConfig()
        self.evaluation: EvaluationConfig = EvaluationConfig()
        self.game: GameConfig = GameConfig()
        self.genome: GenomeConfig = GenomeConfig()
        self.population: PopulationConfig = PopulationConfig()
        self.update()
    
    def read(self):
        """Override read-method to pretty-print result"""
        result = "Config:\n"
        for param in sorted(self.__slots__):
            result += f" > {getattr(self, param).read()}"
        return result
    
    def update(self, **kwargs):
        for param in self.__slots__:
            getattr(self, param).update(self)
