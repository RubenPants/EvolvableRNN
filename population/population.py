"""
population.py

This class is responsible for containing the population. It will simply be used as a container for the population's core
functionality such as its configuration and methods used to persist the population properly.
"""
import re
from glob import glob

from config import Config
from population.utils.genome import Genome
from population.utils.population_util.reproduction import DefaultReproduction
from population.utils.population_util.species import DefaultSpecies
from population.utils.population_util.stagnation import DefaultStagnation
from population.utils.reporter_util.reporting import ReporterSet, StdOutReporter
from population.utils.visualizing.genome_visualizer import draw_net
from utils.myutils import append_log, get_subfolder, load_pickle, store_pickle, update_dict


class Population:
    """ Container for each of the agent's control mechanisms. """
    
    __slots__ = {
        'name', 'folder_name', 'use_backup', 'best_fitness', 'best_genome', 'best_genome_hist', 'config', 'generation',
        'population', 'reporters', 'reproduction', 'species', 'species_hist', 'log_print',
    }
    
    def __init__(self,
                 name: str,
                 folder_name: str,
                 config: Config = None,
                 log_print: bool = True,
                 overwrite: bool = False,
                 use_backup: bool = True,
                 ):
        """
        The population will be concerned about everything directly and solely related to the population. These are
        reporters, persisting methods, evolution and more.
        
        :param name: Name of population, used when specific population must be summon
        :param folder_name: Name of the folder to which the population belongs (NEAT, GRU-NEAT, ...)
        :param config: Main config file
        :param log_print: Print during logging
        :param overwrite: Overwrite the current population
        :param use_backup: Use the backed-up populations found in subfolder 'population_backup' (not connected with git)
        """
        self.config: Config = config if config else Config()
        self.name = name
        self.folder_name = folder_name
        self.use_backup: bool = use_backup
        
        # Placeholders
        self.best_fitness: dict = dict()  # Container for the best fitness of each generation
        self.best_genome: Genome = None  # Current most fit genome
        self.best_genome_hist: dict = dict()  # Container for the best genome for each generation
        self.generation = 0  # Current generation of the population
        self.population = None  # Container for all the current used genomes
        self.reporters: ReporterSet = None  # Reporters that report during training, evaluation, ...
        self.reproduction: DefaultReproduction = None  # Reproduction mechanism of the population
        self.species: DefaultSpecies = None  # Container for all the species
        self.species_hist: dict = dict()  # Container for the elite player of each species at each generation
        self.log_print: bool = log_print  # By default, print result during logging
        
        # Try to load the population, create new if not possible
        if not self.load() or overwrite:
            self.create_population()
    
    def __str__(self):
        return self.name
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    def create_population(self):
        """Create a new population based on the given config file."""
        stagnation = DefaultStagnation(self.config.population, self.reporters)
        self.reporters = ReporterSet()
        self.reproduction = DefaultReproduction(self.reporters, stagnation)
        
        # Create a population from scratch, then partition into species
        self.population = self.reproduction.create_new(config=self.config,
                                                       num_genomes=self.config.population.pop_size)
        self.species = DefaultSpecies(reporters=self.reporters)
        self.species.speciate(config=self.config,
                              population=self.population,
                              generation=self.generation,
                              logger=self.log)
        
        # Add to each of the species its elites
        self.update_species_fitness_hist()
        
        # Use 'print' to output information about the run
        reporter = StdOutReporter()
        self.add_reporter(reporter)
        
        # Save newly made population
        self.save()
        
        # Write population configuration to file
        with open(f"population{'_backup' if self.use_backup else ''}/"
                  f"storage/"
                  f"{self.folder_name}/"
                  f"{self}/"
                  f"config.txt", 'w') as f:
            f.write(self.config.read())
    
    def evolve(self):
        """
        The evolution-process consists out of two phases:
          1) Reproduction
          2) Speciation
        
        This method manipulates the population itself, so nothing has to be returned
        """
        # Create the next generation from the current generation
        self.population = self.reproduction.reproduce(
                config=self.config,
                species=self.species,
                generation=self.generation,
                logger=self.log,
        )
        
        # Check for complete extinction
        if not self.species.species:
            self.reporters.complete_extinction(logger=self.log)
            
            # If requested by the user, create a completely new population, otherwise raise an exception
            self.population = self.reproduction.create_new(config=self.config,
                                                           num_genomes=self.config.population.pop_size)
        
        # Divide the new population into species
        self.species.speciate(config=self.config,
                              population=self.population,
                              generation=self.generation,
                              logger=self.log)
        
        # Add to each of the species its elites
        self.update_species_fitness_hist()
        
        # Increment generation count
        self.generation += 1
    
    def update_species_fitness_hist(self):
        """Add for each of the current species their fitness score."""
        for specie_id, specie in self.species.species.items():
            if specie_id not in self.species_hist: self.species_hist[specie_id] = dict()
            max_fitness = max({m.fitness if m.fitness else 0 for m in specie.members.values()})
            self.species_hist[specie_id][self.generation] = max_fitness
    
    def visualize_genome(self, debug=False, genome=None, show: bool = True):
        """
        Visualize the architecture of the given genome.
        
        :param debug: Add excessive genome-specific details in the plot
        :param genome: Genome that must be visualized, best genome is chosen if none
        :param show: Directly visualize the architecture
        """
        if not genome:
            genome = self.best_genome if self.best_genome else list(self.population.values())[0]
        name = f"genome_{genome.key}"
        sf = get_subfolder(f"population{'_backup' if self.use_backup else ''}/"
                           f"storage/"
                           f"{self.folder_name}/"
                           f"{self}/", 'images')
        sf = get_subfolder(sf, f'architectures{"_debug" if debug else ""}')
        draw_net(config=self.config.genome,
                 genome=genome,
                 debug=debug,
                 filename=f'{sf}{name}',
                 view=show)
    
    # ---------------------------------------------> FUNCTIONAL METHODS <--------------------------------------------- #
    
    def add_evaluation_result(self, eval_result, overwrite: bool = False):
        """Append the result of the evaluation."""
        sf = get_subfolder(f"population{'_backup' if self.use_backup else ''}/"
                           f"storage/"
                           f"{self.folder_name}/"
                           f'{self}/', 'evaluation')
        sf = get_subfolder(sf, f"{self.generation:05d}")
        update_dict(f'{sf}results', eval_result, overwrite=overwrite)
    
    def add_reporter(self, reporter):
        """
        Add a new reporter to the population's ReporterSet.
        """
        self.reporters.add(reporter)
    
    def remove_reporter(self, reporter):
        """
        Remove the given reporter from the population's ReporterSet.
        """
        self.reporters.remove(reporter)
    
    # ---------------------------------------------> PERSISTING METHODS <--------------------------------------------- #
    
    def save(self):
        """
        Save the population as the current generation.
        """
        # Create needed subfolder if not yet exist
        f = get_subfolder(f"population{'_backup' if self.use_backup else ''}/storage/", f'{self.folder_name}')
        if len(str(self).split("/")) > 1: get_subfolder(f, f'{str(self).split("/")[0]}')
        f = get_subfolder(f, f'{self}')
        f = get_subfolder(f, 'generations')
        
        # Save the population
        store_pickle(self, f'{f}gen_{self.generation:05d}')
        self.log(f"Population '{self}' saved! Current generation: {self.generation}")
    
    def load(self, gen=None):
        """
        Load in a game, specified by its current id.
        
        :return: True: game successfully loaded | False: otherwise
        """
        try:
            path = f"population{'_backup' if self.use_backup else ''}/" \
                   f"storage/" \
                   f"{self.folder_name}/" \
                   f"{self}/" \
                   f"generations/"
            if gen is None:
                # Load in all previous populations
                populations = glob(f"{path}gen_*")
                if not populations: raise FileNotFoundError
                
                # Find newest population and save generation number under 'gen'
                populations = [p.replace('\\', '/') for p in populations]
                regex = r"(?<=" + re.escape(f'{path}gen_') + ")[0-9]*"
                gen = max([int(re.findall(regex, p)[0]) for p in populations])
            
            # Load in the population under the specified generation
            pop = load_pickle(f'{path}/gen_{gen:05d}')
            self.best_fitness = pop.best_fitness
            self.best_genome = pop.best_genome
            self.best_genome_hist = pop.best_genome_hist
            self.config = pop.config
            self.generation = pop.generation
            self.population = pop.population
            self.reporters = pop.reporters
            self.reproduction = pop.reproduction
            self.species = pop.species
            self.species_hist = pop.species_hist
            self.log(f"\nPopulation '{self}' loaded successfully! Current generation: {self.generation}")
            return True
        except FileNotFoundError:
            return False
    
    def log(self, inp, print_result: bool = None):
        """
        Append input to the population's log-file.
        
        :param inp: Input that must be logged, supported types: str, json, dict.
        :param print_result: Print out the result that will be logged.
        """
        if print_result is None: print_result = self.log_print
        # Parse input
        if type(inp) == str:
            logged_inp = inp
        else:
            raise NotImplementedError(f"Given type '{type(inp)}' is not supported for logging.")
        
        # Print if requested
        if print_result: print(logged_inp)
        
        # Append the string to the file
        append_log(logged_inp, f"population{'_backup' if self.use_backup else ''}/"
                               f"storage/"
                               f"{self.folder_name}/"
                               f"{self}/"
                               f"logbook.log")
    
    def inject_genome(self, genome: Genome):
        """Inject the given genome into the population."""
        self.population[genome.key] = genome
