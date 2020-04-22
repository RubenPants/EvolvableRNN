"""
reproduction.py

Handles creation of genomes, either from scratch or by sexual or asexual reproduction from parents.
"""
from __future__ import division

import copy
from itertools import count
from random import choice

from neat.six_util import iteritems, itervalues

from config import Config
from population.utils.genome import Genome
from population.utils.population_util.species import DefaultSpecies
from population.utils.population_util.stagnation import DefaultStagnation
from population.utils.reporter_util.reporting import ReporterSet


class DefaultReproduction:
    """The default NEAT-python reproduction scheme: explicit fitness sharing with fixed-time species stagnation."""
    
    def __init__(self, reporters: ReporterSet, stagnation: DefaultStagnation):
        self.reporters = reporters  # Report on reproduction
        self.genome_indexer = count(1)  # Helps with indexing the genomes (creating unique keys)
        self.stagnation = stagnation  # Defines if specie has become stagnant or not
        self.elite_history = dict()  # Keeps of each generation the elites
    
    def create_new(self, config: Config, num_genomes):
        """Create a new (random initialized) population."""
        new_genomes = dict()
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            g = Genome(key, num_outputs=config.genome.num_outputs, bot_config=config.bot)
            g.configure_new(config.genome)
            new_genomes[key] = g
        return new_genomes
    
    @staticmethod
    def compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size):
        """Compute the proper number of offspring per species (proportional to fitness)."""
        af_sum = sum(adjusted_fitness)
        spawn_amounts = []
        for af, ps in zip(adjusted_fitness, previous_sizes):
            # Determine number of candidates in the specie, which is always at least the minimum specie-size
            s = max(min_species_size, af / af_sum * pop_size)
            
            # Adjust the number of candidates in the population via a weighted average over a specie's previous size,
            #  this is done to prevent unnecessary fluctuations in the specie-sizes
            #   * s is the number of candidates the specie will contain  (80% weight)
            #   * ps is the specie his previous size  (20% weight)
            # Example: ps=64, s=32, new specie size will then be 48 (=64-16)
            spawn_amounts.append(0.8 * s + 0.2 * ps)
        
        # Normalize the spawn amounts so that the next generation is roughly the requested population size
        total_spawn = sum(spawn_amounts)
        norm = pop_size / total_spawn
        spawn_amounts = [max(min_species_size, int(round(n * norm))) for n in spawn_amounts]
        return spawn_amounts
    
    def reproduce(self, config: Config, species: DefaultSpecies, generation: int, logger=None):
        """Handles creation of genomes, either from scratch or by sexual or asexual reproduction from parents."""
        # Check is one of the species has become stagnant (i.e. must be removed)
        remaining_fitness = []
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(config=config, species_set=species, gen=generation):
            # If specie is stagnant, then remove
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s, logger=logger)
            
            # Add the specie to remaining_species and save each of its members' fitness
            else:
                remaining_fitness.extend(m.fitness for m in itervalues(stag_s.members))
                remaining_species.append(stag_s)
        
        # If no species is left then force hard-reset
        if not remaining_species:
            species.species = dict()
            return dict()
        
        # Calculate the adjusted fitness, normalized by the minimum fitness across the entire population
        for specie in remaining_species:
            # Adjust a specie's fitness in a fitness sharing manner. A specie's fitness gets normalized by the number of
            #  members it has, this to ensure that a better performing specie does not takes over the population
            # A specie's fitness is determined by its most fit genome
            specie_fitness = max([m.fitness for m in specie.members.values()])
            specie_size = len(specie.members)
            specie.adjusted_fitness = specie_fitness / max(specie_size, config.population.min_specie_size)
        
        # Minimum specie-size is defined by the number of elites and the minimal number of genomes in a population
        spawn_amounts = self.compute_spawn(adjusted_fitness=[s.adjusted_fitness for s in remaining_species],
                                           previous_sizes=[len(s.members) for s in remaining_species],
                                           pop_size=config.population.pop_size,
                                           min_species_size=max(config.population.min_specie_size,
                                                                config.population.genome_elitism))
        
        # Setup the next generation by filling in the new species with their elites and offspring
        new_population = dict()
        species.species = dict()
        for spawn_amount, specie in zip(spawn_amounts, remaining_species):
            # If elitism is enabled, each species will always at least gets to retain its elites
            spawn_amount = max(spawn_amount, config.population.genome_elitism)
            assert spawn_amount > 0
            
            # Get all the specie's old (evaluated) members
            old_members = list(iteritems(specie.members))  # Temporarily save members of last generation
            specie.members = dict()  # Reset members
            species.species[specie.key] = specie
            
            # Sort members in order of descending fitness (i.e. most fit members in front)
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)
            
            # Make sure that all the specie's elites are added to the new generation
            if config.population.genome_elitism > 0:
                # Add the specie's elites to the global population
                for i, m in old_members[:config.population.genome_elitism]:
                    new_population[i] = m
                    spawn_amount -= 1
                
                # Add the specie's past elites as well if requested
                for i in range(min(len(specie.elite_list), config.population.genome_elite_stagnation - 1)):
                    gid, g = specie.elite_list[-(i + 1)]
                    if gid not in new_population:  # Only add genomes not yet present in the population
                        new_population[gid] = g
                        spawn_amount -= 1
                
                # Update the specie's elite_list
                specie.elite_list.append(old_members[0])
            
            # Check if the specie has the right to add more genomes to the population
            if spawn_amount <= 0: continue
            
            # Only use the survival threshold fraction to use as parents for the next generation, use at least all the
            #  elite of a population as parents
            reproduction_cutoff = max(round(config.population.parent_selection * len(old_members)),
                                      config.population.genome_elitism)
            
            # Since asexual reproduction, at least one parent must be chosen
            reproduction_cutoff = max(reproduction_cutoff, 1)
            parents = old_members[:reproduction_cutoff]
            
            # Add the elites again to the parent-set such that these have a greater likelihood of being chosen
            parents += old_members[:config.population.genome_elitism]
            
            # Fill the specie with offspring based, which is a mutation of the chosen parent
            while spawn_amount > 0:
                spawn_amount -= 1
                
                # Init genome dummy (values are overwritten later)
                gid = next(self.genome_indexer)
                child: Genome = Genome(gid, num_outputs=config.genome.num_outputs, bot_config=config.bot)
                
                # Choose the parents, note that if the parents are not distinct, crossover will produce a genetically
                #  identical clone of the parent (but with a different ID)
                p1_id, p1 = choice(parents)
                child.connections = copy.deepcopy(p1.connections)
                child.nodes = copy.deepcopy(p1.nodes)
                
                # Mutate the child
                child.mutate(config.genome)
                
                # Ensure that the child is connected
                while len(child.get_used_connections()) == 0:
                    child.mutate_add_connection(config.genome)
                
                # Add the child to the global population
                new_population[gid] = child
        
        return new_population
