import random
import copy

def cull(genome_list, speciesStrength):
    # Takes a list of genomes. Removes most of the least fit members.
    # Retains most of the most fit members. Returns a list of genome
    # that is exactly half the length of the inputed list.
    # print 'genomeList', len(genome_list)
    genomeWeights = [(x.fitness,x) for x in genome_list]
    culledGenomes = []
    sortedGenomes = list(reversed(sorted(genomeWeights,key=lambda x: x[0])))
    try:
        lower_choice = [sortedGenomes[0], sortedGenomes[1]]
    except IndexError: # if there is only one member of species an IndexError is flagged.
        sortedGenomes.append(copy.deepcopy(sortedGenomes[0]))# Clones the sole survivor
        # Applies a randomly chosen mutation to the clone
        lower_choice = [sortedGenomes[0], sortedGenomes[1]]
        # Tadaaaaa we've saved a species!s

    cumulativeWeight = sum([x.fitness for x in genome_list])
    for i in range(int(len(genomeWeights)*speciesStrength)):
        randomNumber = random.uniform(0,cumulativeWeight)
        counter = 0
        while counter < len(sortedGenomes):
            if randomNumber < sortedGenomes[counter][0]:
                cumulativeWeight - sortedGenomes[counter][0]
            else:
                temp = sortedGenomes.pop(counter)
                culledGenomes.append(temp[1])
                break
            counter += 1

    if culledGenomes:
        return culledGenomes
    else:
        return [i[1] for i in lower_choice]
