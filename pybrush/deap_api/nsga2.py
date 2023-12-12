from deap import tools 
from deap.benchmarks.tools import hypervolume
import numpy as np
import functools

class DeapIndividual():
    """Class that wraps brush program for creator.Individual class from DEAP."""
    def __init__(self, prg):
        self.prg = prg

def nsga2(toolbox, NGEN, MU, CXPB, use_batch, verbosity, rnd_flt):
    # NGEN = 250
    # MU   = 100
    # CXPB = 0.9
    # rnd_flt: random number generator to sample crossover prob

    def calculate_statistics(ind):
        on_train = ind.fitness.values
        on_val   = toolbox.evaluateValidation(ind)

        return (*on_train, *on_val) 

    stats = tools.Statistics(calculate_statistics)

    stats.register("avg", np.nanmean, axis=0)
    stats.register("med", np.nanmedian, axis=0)
    stats.register("std", np.nanstd, axis=0)
    stats.register("min", np.nanmin, axis=0)
    stats.register("max", np.nanmax, axis=0)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'evals'] + \
                     [f"{stat} {partition} O{objective}"
                         for stat in ['avg', 'med', 'std', 'min', 'max']
                         for partition in ['train', 'val']
                         for objective in toolbox.get_objectives()]

    pop = toolbox.population(n=MU)

    # OBS: evaluate calls fit in the individual. It is different from using it to predict. The
    # function evaluateValidation don't call the fit
    fitnesses = toolbox.map(functools.partial(toolbox.evaluate), pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.survive(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)

    if verbosity > 0: 
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        batch = toolbox.getBatch() # batch will be a random subset only if it was not defined as the size of the train set.
                                   # everytime this function is called, a new random batch is generated.
        if (use_batch): # recalculate the fitness for the parents
            # use_batch is false if batch_size is different from train set size.
            # If we're using batch, we need to re-evaluate every model (without changing its weights).
            # evaluateValidation doesnt fit the weights
            fitnesses = toolbox.map( 
                functools.partial(toolbox.evaluateValidation, data=batch), pop)
        
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

        # Vary the population
        # offspring = tools.selTournamentDCD(pop, len(pop))
        parents = toolbox.select(pop, len(pop))
        # offspring = [toolbox.clone(ind) for ind in offspring]
        offspring = []
        for ind1, ind2 in zip(parents[::2], parents[1::2]):
            off1, off2 = None, None
            if rnd_flt() < CXPB: # either mutation or crossover. 
                off1, off2 = toolbox.mate(ind1, ind2)
            else:
                off1 = toolbox.mutate(ind1)
                off2 = toolbox.mutate(ind2)
            
            if off1 is not None: # Mutation worked. first we fit, then add to offspring
                # Evaluate (instead of evaluateValidation) to fit the weights of the offspring
                off1.fitness.values = toolbox.evaluate(off1) 
                if use_batch: # Adjust fitness to the same data as parents
                    off1.fitness.values = toolbox.evaluateValidation(off1, data=batch)
                offspring.extend([off1])

            if off2 is not None:
                off2.fitness.values = toolbox.evaluate(off2) 
                if use_batch:
                    off2.fitness.values = toolbox.evaluateValidation(off2, data=batch)
                offspring.extend([off2])

        # Select the next generation population (no sorting before this step, as 
        # survive==offspring will cut it in half)
        pop = toolbox.survive(pop + offspring, MU)

        pop.sort(key=lambda x: x.fitness, reverse=True)

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(offspring)+(len(pop) if use_batch else 0), **record)

        if verbosity > 0: 
            print(logbook.stream)

    if verbosity > 0: 
        print("Final population hypervolume is %f" % hypervolume(pop, [1000.0, 50.0]))

    archive = tools.ParetoFront() 
    archive.update(pop)

    return archive, logbook