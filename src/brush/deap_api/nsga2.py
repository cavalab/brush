from deap import tools 
from deap.benchmarks.tools import diversity, convergence, hypervolume
import numpy as np
import functools


def nsga2(toolbox, NGEN, MU, CXPB, use_batch, verbosity, rng):
    # NGEN = 250
    # MU = 100
    # CXPB = 0.9

    def calculate_statistics(ind):
        on_train = ind.fitness.values
        on_val   = toolbox.evaluateValidation(ind)

        return (*on_train, *on_val) 

    stats = tools.Statistics(calculate_statistics)

    stats.register("ave train", np.mean, axis=0)
    stats.register("std train", np.std, axis=0)
    stats.register("min train", np.min, axis=0)

    stats.register("ave val", np.mean, axis=0)
    stats.register("std val", np.std, axis=0)
    stats.register("min val", np.min, axis=0)

    # stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "ave train", "std train", "min train", \
                     "ave val", "std val", "min val"

    pop = toolbox.population(n=MU)

    batch = toolbox.getBatch() # everytime this function is called, a new random batch is generated
    
    # OBS: evaluate calls fit in the individual. It is different from using it to predict. The
    # function evaluateValidation don't call the fit
    fitnesses = toolbox.map(functools.partial(toolbox.evaluate, data=batch), pop)
    
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
        if (use_batch): #batch will be random only if it is not the size of the entire train set. In this case, we dont need to reevaluate the whole pop
            batch = toolbox.getBatch()
            fitnesses = toolbox.map(functools.partial(toolbox.evaluate, data=batch), pop)
        
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

        # Vary the population
        # offspring = tools.selTournamentDCD(pop, len(pop))
        parents = toolbox.select(pop, len(pop))
        # offspring = [toolbox.clone(ind) for ind in offspring]
        offspring = []

        for ind1, ind2 in zip(parents[::2], parents[1::2]):
            if rng.random() < CXPB:
                off1, off2 = toolbox.mate(ind1, ind2)
            else:
                off1, off2 = ind1, ind2
                
            # avoid inserting empty solutions
            if off1 != None: off1 = toolbox.mutate(off1)
            if off1 != None: offspring.extend([off1])

            if off2 != None: off2 = toolbox.mutate(off2)
            if off2 != None: offspring.extend([off2])

        # archive.update(offspring)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(functools.partial(toolbox.evaluate, data=batch), invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.survive(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(offspring)+(len(pop) if use_batch else 0), **record)

        if verbosity > 0: 
            print(logbook.stream)

    if verbosity > 0: 
        print("Final population hypervolume is %f" % hypervolume(pop, [1000.0, 50.0]))

    archive = tools.ParetoFront() 
    archive.update(pop)

    return archive, logbook