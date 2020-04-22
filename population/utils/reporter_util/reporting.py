"""
reporting.py

Makes possible reporter classes, which are triggered on particular events and may provide information to the user,
may do something else such as checkpointing, or may do both.
"""
from __future__ import division, print_function

import time

from neat.math_util import mean, stdev
from neat.six_util import iterkeys, itervalues


class ReporterSet(object):
    """
    Keeps track of the set of reporters
    and gives methods to dispatch them at appropriate points.
    """
    
    def __init__(self):
        self.reporters = []
    
    def add(self, reporter):
        self.reporters.append(reporter)
    
    def remove(self, reporter):
        self.reporters.remove(reporter)
    
    def start_generation(self, gen, logger=None):
        for r in self.reporters: r.start_generation(gen, logger=logger)
    
    def end_generation(self, population, name, species_set, logger=None):
        for r in self.reporters: r.end_generation(population, name=name, species_set=species_set, logger=logger)
    
    def post_evaluate(self, population, species, best_genome, logger=None):
        for r in self.reporters: r.post_evaluate(population, species, best_genome, logger=logger)
    
    def post_reproduction(self, population, species, logger=None):
        for r in self.reporters: r.post_reproduction(population, species, logger=logger)
    
    def complete_extinction(self, logger=None):
        for r in self.reporters: r.complete_extinction(logger=logger)
    
    def found_solution(self, generation, best, logger=None):
        for r in self.reporters: r.found_solution(generation, best, logger=logger)
    
    def species_stagnant(self, sid, species, logger=None):
        for r in self.reporters: r.species_stagnant(sid, species, logger=logger)
    
    def info(self, msg, logger=None, print_result=True):
        for r in self.reporters: r.info(msg, logger=logger, print_result=print_result)


class BaseReporter(object):
    """Definition of the reporter interface expected by ReporterSet."""
    
    def start_generation(self, generation, logger=None):
        pass
    
    def end_generation(self, population, name, species_set, logger=None):
        pass
    
    def post_evaluate(self, population, species, best_genome, logger=None):
        pass
    
    def post_reproduction(self, population, species, logger=None):
        pass
    
    def complete_extinction(self, logger=None):
        pass
    
    def found_solution(self, generation, best, logger=None):
        pass
    
    def species_stagnant(self, sid, species, logger=None):
        pass
    
    def info(self, msg, logger=None, print_result=True):
        pass


class StdOutReporter(BaseReporter):
    """Uses `print` to output information about the run; an example reporter class."""
    
    def __init__(self):
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0
    
    def start_generation(self, generation, logger=None):
        self.generation = generation
        msg = f"\n********** Running generation {generation} **********\n"
        t = time.localtime()
        print(f"Current time: {t.tm_hour:02d}h-{t.tm_min:02d}m-{t.tm_sec:02d}s")
        logger(msg) if logger else print(msg)
        self.generation_start_time = time.time()
    
    def end_generation(self, population, name, species_set, logger=None):
        sids = list(iterkeys(species_set.species))
        sids.sort()
        msg = f"\nPopulation '{name}' with {len(population):d} members in {len(species_set.species):d} species:" \
              f"\n\t specie    age    size    fitness    adj fit    stag    repr size " \
              f"\n\t========  =====  ======  =========  =========  ======  ==========="
        logger(msg) if logger else print(msg)
        for sid in sids:
            s = species_set.species[sid]
            a = self.generation - s.created
            n = len(s.members)
            f = "--" if s.fitness is None else f"{s.fitness:.5f}"
            af = "--" if s.adjusted_fitness is None else f"{s.adjusted_fitness:.5f}"
            st = self.generation - s.last_improved
            sh = species_set.species[sid].representative.size()
            msg = f"\t{sid:^8}  {a:^5}  {n:^6}  {f:^9}  {af:^9}  {st:^6}  {sh!r:^11}"
            logger(msg) if logger else print(msg)
        logger("") if logger else print()
        
        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        if self.num_extinctions > 0:
            msg = f'Total extinctions: {self.num_extinctions:d}'
            logger(msg) if logger else print(msg)
        if len(self.generation_times) > 1:
            msg = f"Generation time: {elapsed:.3f} sec  ({average:.3f} average)"
            logger(msg) if logger else print(msg)
        else:
            msg = f"Generation time: {elapsed:.3f} sec"
            logger(msg) if logger else print(msg)
    
    def post_evaluate(self, population, species, best_genome, logger=None):
        fitnesses = [c.fitness for c in itervalues(population)]
        
        # Full population
        msg = f'Full population\'s fitness overview:' \
              f'\n\t-       best fitness: {max(fitnesses):3.5f}' \
              f'\n\t-       mean fitness: {mean(fitnesses):3.5f}' \
              f'\n\t-      worst fitness: {min(fitnesses):3.5f}' \
              f'\n\t- standard deviation: {stdev(fitnesses):3.5f}'
        logger(msg, print_result=False) if logger else print(msg)
        
        # Best genome
        best_species_id = species.get_species_id(best_genome.key)
        msg = f'Best genome overview:' \
              f'\n\t- fitness: {best_genome.fitness:3.5f}' \
              f'\n\t- size (hid, conn): {best_genome.size()!r}' \
              f'\n\t- genome id: {best_genome.key}' \
              f'\n\t- belongs to specie: {best_species_id}'
        if logger:
            logger(msg, print_result=False)
            print(f"Best genome '{best_genome.key}' "
                  f"with size {best_genome.size()!r} "
                  f"of specie: '{best_species_id}' "
                  f"has fitness: {best_genome.fitness:.5f}")
        else:
            print(msg)
    
    def complete_extinction(self, logger=None):
        self.num_extinctions += 1
        msg = "All species are extinct."
        logger(msg) if logger else print(msg)
    
    def found_solution(self, gen, best, logger=None):
        msg = f'\nBest individual in generation {gen} meets fitness threshold - size: {best.size()!r}'
        logger(msg) if logger else print(msg)
    
    def species_stagnant(self, sid, s, logger=None):
        msg = f"\nSpecies {sid} with {len(s.members)} members is stagnated (fitness: {round(s.fitness, 5)}): removed!"
        logger(msg) if logger else print(msg)
    
    def info(self, msg, logger=None, print_result=True):
        logger(msg, print_result=print_result) if logger else print(msg)
