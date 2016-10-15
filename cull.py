import random

def cull(genome_list, speciesStrength):
    # Takes a list of genomes. Removes most of the least fit members.
    # Retains most of the most fit members. Returns a list of genome
    # that is exactly half the length of the inputed list.
    genomeWeights = [x.fitness() for x in genome_list]
    culledGenomes = []
    sortedGenomes = list(reversed(sorted(genomeWeights)))
    cumulativeWeight = sum(genomeWeights)
    for i in range(int(len(genomeWeights)*speciesStrength)):
        randomNumber = random.uniform(0,cumulativeWeight)
        counter = 0
        while True:
            if randomNumber < sortedGenomes[counter]:
                cumulativeWeight - sortedGenomes[counter]
            else:
                culledGenomes.append(sortedGenomes.pop(counter))
                break
            counter += 1
    return culledGenomes
