
class DeapIndividual():
    """Class that wraps brush program for creator.Individual class from DEAP."""
    def __init__(self, prg):
        self.prg = prg

def cross(self, ind1, ind2):
    offspring = [] 

    for i,j in [(ind1,ind2),(ind2,ind1)]:
        off = creator.Individual(i.prg.cross(j.prg))
        # off.fitness.valid = False
        offspring.append(off)

    return offspring[0], offspring[1]