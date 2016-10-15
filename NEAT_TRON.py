import random
from fuzzy_delete import fuzzy_delete # Ensure this works correctly
import numpy as np

outputs = ["UP", "DOWN", "LEFT", "RIGHT"]
# Constants for species assignment
c_DISJOINT = 1.0
C_WEIGHT = 0.4
INITIAL_GENOMES = 200

n = neatNN()
n.start()
print(n.species)

class neatNN(object):
    """ This class will aim to implement the NEAT algorithm to creating neural networks that can evolve structures capable of playing
    a strong game of Tron. It stores both the nodes and the connection between the nodes. """

    def __init__(self):
        self.innovation = 0
        self.species    = []           # List of genomes.

    def start(self):
        genomes = [genome(self, [(1,0),(-1,0),(0,1),(0,-1)], outputs, 2) for _ in range(INITIAL_GENOMES)]
        self.speciesReps = []
        self.species = self.sortToSpecies(genomes)


    def evaluate(self, board, headPos):
        board = self.processBoard(board, headPos)
        currentGenome = self.genomes(self.currentSpecies, self.currentGenome)
        currentGenome.fitness += 1

        #FEED board TO DECISION NEURAL NETWORK

    def processBoard(self, array, xy): #numpy array, NN head position
        og = np.copy(array)
        array.fill(0)
        array.resize(((2 * array.shape[0]) -1, (2 * array.shape[1]) -1), refcheck = False)
        array[0:og.shape[0],0:og.shape[1]] = og
        array = np.roll(array, (og.shape[0]-1)-xy[1], 0)
        array = np.roll(array, (og.shape[1]-1)-xy[0], 1)
        return array

    def breedingControl(self):
    """This is a method that oversees each stage of the breeding process."""
    nextGenerationUnsorted = []

    # Iterate over each species
    for species in self.genomes:
        breedingList = self.getBreedingPairs(species) 	# Culls the species of undesirables and returns a list of a tuple of breeding pairs
        for breedingPair in breedingList:
            nextGenerationUnsorted.append(self.breedGenomes(breedingPair[0], breedingPair[1]))

    retrrn nextGenerationUnorted



    def getBreedingPairs(self, genome):
        culledGenome = fuzzy_delete(genome)
        random.shuffle(culledGenome)
        return [(culledGenome[x],culledGenome[(x+1)%len(culledGenome)])  for x in range(len(culledGenome))]

    def breedGenome(self, parent1, parent2):
        """ASSUMES node1 and node2 are compatable parent nodes. This method will go through each parent and select genes weights to inherit at
        random from each of the parent. Genes not expressed in either parent will be carried forth to the child node!"""

        # Conception
        child = genome(self, [], outputs, 0) #create an 'empty' child.


        # Generating set with the union and intersect of A and B (all innovations in both A and B)
        innovationUnion = parent1.innovations | parent2.innovations
        innovationIntersect = parent1.innovations & parent2.innovations

        # Generating sets with difference of parent1, parent2
        parent1Diff = parent1.innovations - parent2.innovations
        parent2Diff = parent2.innovations - parent1.innovations

        for innovation in innovationUnion:

            if innovation in innovationIntersect:
                # Aliasing certain variables to be more easily obtained.
                inNode = parent1.nodes[parent1.genes[innovation].inNode]
                outNode = parent1.nodes[parent1.genes[innovation].outNode]
                inherited = random.choice(parent1.genes[innovation], parent2.genes[innovation])
                child.addNode(inNode)
                child.addNode(outNode)
                child.addConnection(inherited.inNode, inherited.outNode, inherited.weight, innovation)

            elif (innovation in parent1Diff) and (parent1.fitness > parent2.fitness):
                # Aliasing certain variables to be more easily obtained.
                inNode = parent1.nodes[parent1.genes[innovation].inNode]
                outNode = parent1.nodes[parent1.genes[innovation].outNode]
                inherited = parent1.genes[innovation]
                child.addNode(inNode)
                child.addNode(outNode)
                child.addConnection(inherited.inNode, inherited.outNode, inherited.weight, innovation)

            elif (innovation in parent2Diff) and (parent2.fitness > parent1.fitness):
                # Aliasing certain variables to be more easily obtained.
                inNode = parent2.nodes[parent2.genes[innovation].inNode]
                outNode = parent2.nodes[parent2.genes[innovation].outNode]
                inherited = parent2.genes[innovation]
                child.addNode(inNode)
                child.addNode(outNode)
                child.addConnection(inherited.inNode, inherited.outNode, inherited.weight, innovation)

        return child

    def getSpeciesReps(self):
        """Assumes species is a list of lists
        which where each list is a species."""

        self.speciesReps = []
        for species in self.species:
            self.selfspeciesReps.append(random.choice(species))

    def sortToSpecies(self, newGenomes):
        nextGeneration = [list([]) for _ in range(len(self.species))] # This is where we store the results of our sorting
        # Acquire a represetative for each species and place it in a vector

        i = 0    #  Initialising I to the first index of our list
        while i < len(newGenomes):  # until I hs gon through all genomes
            genome = newGenomes[i]  # newGenomes is the unsorted list of next generation of genomes
            genomeSpecies = genome.determineSpecies(self, self.speciesReps)
            if genomeSpecies != "NEW":
                nextGeneration[genomeSpecies].append(genome)  # If it belongs to a species assign it to the proper sub_list
            else:
                nextGeneration.append([genome]) # Adds it all to to the list of next generation of genomes list.
                self.speciesReps.append(genome) # We need a new species rep... there is only one genome to choose from
            i += 1
        return(nextGeneration) # nextGeneration is our table of tables!

    def nextGenome(self, winner):
        """Makes the transition to the next genome in running the neural network.
        If the genome is a winner it will assign it a certain bonus fitness."""
        currentGenome = self.genomes(self.currentSpecies, self.currentGenome)  # Aliasing current genome
        if winner:
            currentGenome.fitness += 400
        # Now we either increment to the next genome in the species or we go to the next species
        maxGenomesIndex = len(self.genomes[self.currentSpecies]) - 1  # This allows us to quickly check if we're out of bounds
        lastSpecies = len(self.genomes - 1)

        # Aliasing our booleans function.
        moreGenomes = self.currentGenome + 1 < maxGenomesIndex

        atLastSpecies = (self.currentSpecies == lastSpecies)

        if (moreGenomes):
            # We simply need to increment to the next member of the species
            self.currentGenome += 1  # Is there less redundant way to do this?
            return
        # Potential Breakpoint.
        elif (not MoreGenomes) and not atLastSpecies: # Checking to see if we are at the last species
            self.currentSpecies += 1
            self.currentGenome = 0 # Going back to the first genome
        else: # We have tested all genomes
            print("SYNTAX")

    def calculateDistance(self, g1, g2):
        """Somewhat of a quick ad dirty implementation of our Distance function
        as defined by Stanley and Miikkulainen, but ignoring the distinction between disjoint and excess genes, just using 'disjoint' to encompass both"""
        longestGenome = max(len(g1.innovations), len(g2.innovations))
        disjointGeneCount = len(g1.innovations ^ g2.innovations)  # Symmetric Difference
        averageWeightDiff = self.getAvgWeightDiff(g1, g2)  # Get the average weights of each node and sums them up :Dhv
        distance = (C_DISJOINT*disjointGeneCount)/longestGenome + C_WEIGHT * averageWeightDiff
        return distance

    def getAvgWeightDiff(self, g1, g2):
        tot = 0
        num = 0
        for innovation in (g1.innovations & g2.innovations):
            num += 1
            tot += abs(g1.nodes[innovation]-g2.nodes[innovation])
        if num > 0:
            return tot / num
        else:
            return 0

class genome(object):
    #initialInputs is a tuple, first value is number, second is a list of coord-pairs for the inputs. initialOutputs is a list of names
    def __init__(self, father, initialInputs, initialOutputs, initialConnections):
        self.nodes = {}
        self.genes = {}
        self.innovations = set()
        self.fitness = 0

        for i in range(len(initialInputs)):
            self.nodes[i] = inputNode("in", initialInputs[i])

        for i in range(len(initialOutputs)):
            self.nodes[i + len(initialInputs)] = outputNode(initialOutputs[i])

        for i in range(initialConnections):
            self.genes[father.innovation] = gene(random.randint(-1, len(initialInputs)), random.randint(0, len(initialOutputs)), father.innovation)
            father.innovation += 1
            self.innovations.add(father.innovation)

    def addNode(self, node):
        """Adds a node outside the context of mutation"""
        if node.name not in self.nodes:
            self.node[node.name] = node

    def addConnection(self, inNode, outNode, weight, innovation):
        """Adds a connection to the genome outside of the context of incrementing the global innovation variable"""
        self.genome.append(gene(inNode, outNode, innovation, weight)) # This should be sufficient to add a connection

    def mutateAddNode(self, ):
        """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        """

    def mutateAddConnecion(self, father, inNode, outNode, weight):
        """Adds a connecion between two nodes"""
        self.genome.append(gene(inNode, outNode, father.innovation, weight))
        father.innovation += 1
        self.innovations.add(father.innovation)

    def flipConnection(self, connection):
        """Flips a connection to being either enabled or disabled"""
        connection.connected = not(connection.connected)

    def determineSpecies(self, father, speciesReps):
        """Compares the genome to representative members of each species. The first species that satisfies the threshhold will be the
        species that the genome is sorted into.

        speciesVec is a list containing representatives from each species"""
        bestFit = 0
        DELTA_THRESH = 1000     # To do -----------------------------------------------------------
        """Somewhat of """
        for species, rep in enumerate(speciesReps):
            delta = father.calculateDistance(self, rep)     # Im overdoing creating methods
            if (delta < DELTA_THRESH):
                return species

        return "NEW" # This should signal to the calling environment to create a new species.


    def rewardWinning(self):
        """If the genome is a winner it will increase the fitness by a certain amount"""

    def __repr__(self):
        s = ""
        for n in self.nodes:
            s += str(n) + ", "
        s += "\n"
        for g in self.genes:
            s += str(n) + ", "
        return s + "]"

class gene(object):
    """Stores information regarding connexions between nodes."""

    def __init__(self, inNode, outNode, innovation, weight = random.uniform(-1, 1), enabled = True):
        # Wow look at these fancy initialisations.
        self.inNode = inNode
        self.outNode = outNode
        self.innovation = innovation
        self.weight = weight
        self.connected  = enabled

    def __repr__(self):
        return str([self.inNode, self.outNode, self.innovation, self.weight, self.enabled])

class inputNode(object):
    def __init__(self, name, pos, value = 0):
        self.name = name
        self.pos = pos
        self.value = value
    def __repr__(self):
        return str([self.name, self.pos])

class hiddenNode(object):
    def __init__(self, name, value = 0):
        self.name = name
        self.value = value
    def __repr__(self):
        return str([self.name])

class outputNode(object):
    def __init__(self, name, value = 0):
        self.name = name
        self.value = value
    def __repr__(self):
        return str([self.name])
