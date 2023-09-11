from deap import tools 
from deap.benchmarks.tools import diversity, convergence, hypervolume
import numpy as np
import functools


def ga(toolbox, NGEN, MU, CXPB, use_batch, verbosity, rnd_flt):
    def calculate_statistics(ind):
        return (*ind.fitness.values, *toolbox.evaluateValidation(ind)) 

    stats = tools.Statistics(calculate_statistics)

    stats.register("avg", np.mean, axis=0)
    stats.register("med", np.median, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "avg (O1 train, O2 train, O1 val, O2 val)", \
                                     "med (O1 train, O2 train, O1 val, O2 val)", \
                                     "std (O1 train, O2 train, O1 val, O2 val)", \
                                     "min (O1 train, O2 train, O1 val, O2 val)", \
                                     "max (O1 train, O2 train, O1 val, O2 val)"

    pop = toolbox.population(n=MU)

    fitnesses = toolbox.map(functools.partial(toolbox.evaluate), pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)

    if verbosity > 0: 
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        batch = toolbox.getBatch() 
        if (use_batch):
            fitnesses = toolbox.map( 
                functools.partial(toolbox.evaluateValidation, data=batch), pop)
        
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

        # Vary the population
        parents = toolbox.select(pop, len(pop))

        offspring = []
        for ind1, ind2 in zip(parents[::2], parents[1::2]):
            off1, off2 = None, None
            if rnd_flt() < CXPB:
                off1, off2 = toolbox.mate(ind1, ind2)
            else:
                off1 = toolbox.mutate(ind1)
                off2 = toolbox.mutate(ind2)
            
            offspring.extend([off1 if off1 is not None else toolbox.Clone(ind1)])
            offspring.extend([off2 if off2 is not None else toolbox.Clone(ind2)])

        fitnesses = toolbox.map(functools.partial(toolbox.evaluate), offspring)
        if (use_batch): 
            fitnesses = toolbox.map(functools.partial(toolbox.evaluateValidation, data=batch), offspring)

        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population with offspring strategy
        pop = offspring

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(offspring)+(len(pop) if use_batch else 0), **record)

        if verbosity > 0: 
            print(logbook.stream)

    if verbosity > 0: 
        print("Final population hypervolume is %f" % hypervolume(pop, [1000.0, 50.0]))

    archive = tools.ParetoFront() 
    archive.update(pop)

    return archive, logbook