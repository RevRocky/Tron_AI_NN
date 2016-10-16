from __future__ import division
import math
import random
import numpy as np
import time
import pygame

#from cull import cull
from saveToFile import *
#from fileIO import *

outputs = ["U", "D", "L", "R"]   # The four legal moves in tron
# Constants for species assignment
C_DISJOINT = 1.0
C_WEIGHT = 0.4
INITIAL_GENOMES = 5
MUTATE_INPUT_THRESH = 4/3  # 1/75 of random numbers between 0 and 100 ill fall in interval (0, 4/3)
MUTATE_NODE_THRESH = MUTATE_INPUT_THRESH + 3  # Preserves probabilties
MUTATE_CONN_THRESH = MUTATE_NODE_THRESH + 5


class neatNN(object):
    """ This class will aim to implement the NEAT algorithm to creating neural networks that can evolve structures capable of playing
    a strong game of Tron. It stores both the nodes and the connection between the nodes. """

    def __init__(self):
        self.innovation = 0
        self.species    = []           # List of genomes.
        self.tron = Tron()             # Initialises a tron instance
        self.size = (160,160)

    def start(self):
        size = self.size
        self.unsortedNextGen = [genome(self, [(size[0], size[1] - 1), (size[0] - 2, size[1] - 1), (size[0] - 1, size[1]), (size[0] - 1, size[1] - 2)], outputs, 2) for _ in xrange(INITIAL_GENOMES)]

    def learningLoop(self):
        """This method controls the main learningLoop of our neural network
        Order of Operations is

        1) Sort Genomes into Species
        2) Iterate over species, sub-iterate over genomes, play tron.
        3) Tron gives fitness. As you are receiving fitness data compute the species' average.
        4) Make an array of the species' scores and compute
        5) Culling Undesirables
        6) Breeding and add to unsorted list
        7) Mutate (Random Chance)
        """

        self.speciesReps = []
        self.species = self.sortToSpecies(self.unsortedNextGen) # Sorts each species into genomes. Does unsortedNextGen need to be an attribute or not

        arrayMaker = []
        maxFitness = 0
        avgFitness = 0
        count = 0
        for species in self.species:
            specFitTot = 0
            for genome in species:
                count += 1
                fitness, winning = self.evaluate(genome)
                if winning:
                    fitness += 400  # How much we increase fitness by.
                    genome.fitness = fitness
                specFitTot +=  fitness
                if fitness > maxFitness: 
                    bestGenome = genome
            avgFitness += specFitTot
            if len(species) > 0:
                arrayMaker.append(specFitTot / len(species))
            else:
                arrayMaker.append(0)
        
        avgFitness = avgFitness / count

        # Getting the relative fitnesses of each species!
        array = np.array(arrayMaker)
        array = (array / np.amax(array)) / 0.4 + 0.15
        for i, p in enumerate(array):
            self.unsortedNextGen += self.breedingControl(self.species[i], p)

        for genome in self.unsortedNextGen:
            num = random.uniform(0, 100)
            if num < MUTATE_INPUT_THRESH:
                genome.mutateAddNode(self)
            elif num < MUTATE_NODE_THRESH:
                genome.mutateAddNode(self)
            elif num < MUTATE_CONN_THRESH:
                genome.mutateAddConnection(self)
        
        pickletoDisk(self.species, "EcoSystem")  # Saves the whole eco system.
        pickletoDisk(bestGenome, "strongestMember")
        return maxFitness, avgFitness

    def sortToSpecies(self, newGenomes):
        nextGeneration = [list([]) for _ in xrange(len(self.species))] # This is where we store the results of our sorting
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
            tot += abs(g1.genes[innovation].weight-g2.genes[innovation].weight)
        if num > 0:
            return tot / num
        else:
            return 0

    def cull(self, genome_list, speciesStrength):
        genomeWeights = [(x.fitness,x) for x in genome_list]
        culledGenomes = []
        sortedGenomes = list(reversed(sorted(genomeWeights,key=lambda x: x[0])))
        return sortedGenomes[0:int(len(sortedGenomes) * speciesStrength)]

    def evaluate(self, genome):
        """Plays Tron using a NN archetecture specified by the genome that is passed in.
        It will also get moves for the opponent"""
        tron = self.tron
        boardSize = (160, 160)
        tron.start(boardSize)
        headPos = (159, 159)
        gameWinner = False
        fitness = 0
        while (not gameWinner):
            processedBoard = self.processBoard(tron.board, headPos)
            genomeMove = genome.getMove(processedBoard)
            fitness = fitness + 1
            # Assumes opponent wants an unadulterated board. This is a function that will interface with the AI the opponent is playing against
            headPos, winner = tron.tick(genomeMove)  # board is the current board sate. nextTick a tuple with first position being 1 if NN wins, second position if opponent wins
            if 1 in winner:
                gameWinner = True
        # If the first cell in our tuple is 1 the NN has won
        tron.quit()
        if winner[1] == 1:
            nnWinner = True
        else:  # The opponent has won
            nnWinner = False
        return fitness, nnWinner

    def processBoard(self, inp, xy): #numpy array, NN head position
        array = np.copy(inp)
        og = np.copy(array)
        array.fill(0)
        array.resize(((2 * array.shape[0]) -1, (2 * array.shape[1]) -1), refcheck = False)
        array = array.astype(int)
        array[0:og.shape[0],0:og.shape[1]] = og
        array = np.roll(array, (og.shape[0]-1)-xy[1], 0)
        array = np.roll(array, (og.shape[1]-1)-xy[0], 1)
        return array

    def averageFitness(self, species):
        """This returns the average strength of each species"""
        return(sum(species)/len(species))

    def breedingControl(self, species, percentKept):
        """This is a method that oversees each stage of the breeding process.
        This is called from within a for loop in the main function!"""
        nextGenerationUnsorted = []

        # Iterate over each species
        breedingList = self.getBreedingPairs(species, percentKept) 	# Culls the species of undesirables and returns a list of a tuple of breeding pairs

        for breedingPair in breedingList: # breeding our actual input pairs
            nextGenerationUnsorted.append(self.breedGenome(breedingPair[0], breedingPair[1]))
        return nextGenerationUnsorted # An unsorted list of our next

    def getBreedingPairs(self, species, percentKept):
        culledGenome = self.cull(species, percentKept)
        random.shuffle(culledGenome)

        return [(culledGenome[x][1],culledGenome[(x+1)%len(culledGenome)][1])  for x in xrange(len(culledGenome))]

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
                inNode = parent1.genes[innovation].inNode
                outNode = parent1.genes[innovation].outNode

                inherited = choice(parent1.genes[innovation].weight, parent2.genes[innovation].weight)
                child.addNode(inNode)
                child.addNode(outNode)
                child.addConnection(inNode, outNode, innovation, inherited)

            elif (innovation in parent1Diff) and (parent1.fitness >= parent2.fitness):
                # Aliasing certain variables to be more easily obtained.
                inNode = parent1.genes[innovation].inNode
                outNode = parent1.genes[innovation].outNode
                inherited = parent1.genes[innovation]
                child.addNode(inNode)
                child.addNode(outNode)
                child.addConnection(inherited.inNode, inherited.outNode, innovation, inherited.weight)

            elif (innovation in parent2Diff) and (parent2.fitness >= parent1.fitness):
                # Aliasing certain variables to be more easily obtained.
                inNode = parent2.genes[innovation].inNode
                outNode = parent2.genes[innovation].outNode
                inherited = parent2.genes[innovation]
                child.addNode(inNode)
                child.addNode(outNode)
                child.addConnection(inherited.inNode, inherited.outNode, innovation, inherited.weight)

        return child

    def getSpeciesReps(self):
        """Assumes species is a list of lists
        which where each list is a species."""

        self.speciesReps = []
        for species in self.species:
            self.selfspeciesReps.append(random.choice(species))

    def rep(self):
        for i, s in enumerate(self.species):
            print "Species", i
            for i, genome in enumerate(s):
                print "\tGenome", i
                for n in genome.nodes:
                    print "\t\tNode", n.index
                    print "\t\t\tPos", n.pos
                    print "\t\t\tButton", n.button
                for i, g in enumerate(genome.genes.values()):
                    print "\t\tGene", i
                    print "\t\t\tIn", g.inNode
                    print "\t\t\tOut", g.outNode
                    print "\t\t\tInnovation", g.innovation
                    print "\t\t\tWeight", g.weight
                    print "\t\t\tConnected", g.connected


class genome(object):
    #initialInputs is a tuple, first value is number, second is a list of coord-pairs for the inputs. initialOutputs is a list of names
    def __init__(self, parent, initialInputs, initialOutputs, initialConnections):
        self.nodes = []
        self.genes = {}
        self.innovations = set()
        self.fitness = 0
        self.maxNodeNumber = -1

        for i in xrange(len(initialInputs)):
            self.addNode(inputNode(self.maxNodeNumber + 1, self, initialInputs[i]))
            self.maxNodeNumber += 1

        for i in xrange(len(initialOutputs)):
            self.addNode(outputNode(self.maxNodeNumber + 1, self, initialOutputs[i]))
            self.maxNodeNumber += 1

        for i in xrange(initialConnections):
            inPut = self.nodes[random.randint(0, len(initialInputs) - 1)]
            outPut = self.nodes[random.randint(len(initialInputs), len(initialInputs) + len(initialOutputs) - 2)]
            self.addVirginConnection(inPut,outPut, parent.innovation, random.uniform(-1, 1) )
            parent.innovation += 1

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def addNode(self, node):
        """Adds a node outside the context of mutation"""
        if node not in self.nodes:
            self.nodes.append(node)
            self.maxNodeNumber += 1
            self.nodes[-1].index = self.maxNodeNumber
            return node

    def addVirginConnection(self, inNode, outNode, innovation, weight):
        """Adds a new, virgin connection between two nodes."""
        self.genes[innovation] = gene(inNode, outNode, innovation, weight) # This should be sufficient to add a connection
        self.genes[innovation].inNode.outLinks.append(self.genes[innovation])
        self.genes[innovation].outNode.inLinks.append(self.genes[innovation])
        self.innovations.add(innovation)



    def addConnection(self, inNode, outNode, innovation, weight):
        """Adds a connection to the genome outside of the context of incrementing the global innovation variable"""
        self.genes[innovation] = gene(inNode, outNode, innovation, weight) # This should be sufficient to add a connection
        self.genes[innovation].inNode.outLinks.append(self.genes[innovation])
        self.genes[innovation].outNode.inLinks.append(self.genes[innovation])
        self.innovations.add(innovation)
        inNode.refresh()
        outNode.refresh()

    def mutateAddNode(self, parent):
        """Adds a node, splitting a connection"""
        g = random.choice(self.genes.values())  # Picks a random connction
        self.setEnable(g, False)  # disables it
        newNode = self.addNode(hiddenNode(self.maxNodeNumber, self)) # Adds a new hidden node
        self.maxNodeNumber += 1 # Increments self.maxNodeNumber
        self.addConnection(g.inNode, newNode, parent.innovation, 1) #
        parent.innovation += 1
        self.addConnection(newNode, g.outNode, parent.innovation, g.weight)
        parent.innovation += 1


    def mutateAddConnection(self, parent):
        """Adds a connecion between two previously unconnected nodes"""
        n1 = random.choice(self.nodes)
        n2 = random.choice(self.nodes)
        if n1.nType != "out" and n2.nType != "in":
            self.addConnection(n1, n2, parent.innovation, random.uniform(-1, 1))
            parent.innovation += 1

    def mutateAddInput(self, parent):
        """Add an input node"""
        x = random.randint(-1, self.size[0])  # -1 represents a bias unit
        if x == -1:  #iff we have a bias unit
            y = -1
        else: #Regular input
            y = random.randint(0, self.size[1])
        self.addNode(inputNode(self.maxNodeNumber + 1, (x,y)))
        self.maxNodeNumber += 1
        self.addConnection(self.maxNodeNumber, n2, parent.innovation, random.uniform(-1, 1))
        parent.innovation += 1

    def setEnable(self, connection, value):
        """Flips a connection to being either enabled or disabled"""
        connection.connected = value

    def determineSpecies(self, father, speciesReps):
        """Compares the genome to representative members of each species. The first species that satisfies the threshhold will be the
        species that the genome is sorted into.

        speciesVec is a list containing representatives from each species"""
        bestFit = 0
        DELTA_THRESH = 50     # To do -----------------------------------------------------------
        """Somewhat of """
        for species, rep in enumerate(speciesReps):
            delta = father.calculateDistance(self, rep)     # Im overdoing creating methods
            if (delta < DELTA_THRESH):
                return species

        return "NEW" # This should signal to the calling environment to create a new species.

    def rewardWinning(self):
        """If the genome is a winner it will increase the fitness by a certain amount"""
        self.fitness += 400  # Increments fitness by what_ever fitness value we deem appropriate

    def getMove(self, processedBoard):
        """Feedforward one tick at a time of our neural network. Needs to save data between ticks."""
        # Creating lists of my input and output nodes. This should probably be done elsewhere
        outputNodes = []
        for node in self.nodes:
            if node.nType == 'in':
                node.defineInputNode(processedBoard)  # Defines an actual value for our input node
            elif node.nType == 'out':
                outputNodes.append(node)
        # Now we make just one step each time through our neural network.
        for node in self.nodes:
            if node.nType != "out":
                for gene in node.outLinks:
                    theta = gene.weight # aliasing our theta value
                    gene.outNode.inValue += (gene.inNode.outValue * theta)

        # There may be a way to put this into our prior loop but right now I'd rather it explictly be seperate
        for node in self.nodes:
            if (node.nType != 'in'):
                node.outValue = self.sigmoid(node.inValue)  # Copies the output value into our input value
                node.inValue = 0  # Resetting the inValue of an innode

        # Eww One Liners... But c'est hackathon
        return self.maxNode(outputNodes).button  # This is the button press associated with the ideal move


    def maxNode(self, nodes):
        """Assumes that nodes is some object that is an iterable containing node Objects.
        Max Node is the node with the biggest output value."""

        maxNodeValue = -100

        for node in nodes:
            if node.outValue > maxNodeValue:
                maxNode = node
        return maxNode  # This is the node with the biggest value.

class gene(object):
    """Stores information regarding connexions between nodes."""

    def __init__(self, inNode, outNode, innovation, weight, enabled = True):
        # Wow look at these fancy initialisations.
        self.inNode = inNode
        self.outNode = outNode
        self.innovation = innovation
        self.weight = weight
        self.connected  = enabled

class inputNode(object):
    def __init__(self, index, parent, pos, value = 0):
        self.index = index
        self.pos = pos # pos is a tuple that contains the cell that we are looking at relative to the top left corner of adjusted grid. (-1, -1 signifies a bias node)
        self.inValue = 0
        self.button = None
        self.outValue = value # This is just to make code play nicely with eachother
        self.nType = "in"
        self.outLinks = []
        self.parent = parent


    def refresh(self):
        self.outLinks = []
        for g in self.parent.genes.values():
            if g.inNode == self.index:
                self.outLinks.append(g)

    def forward(self, i):
        return self.parent.nodes[self.parent.genes[i].outNode]
    def backward(self, i):
        return self.parent.nodes[self.parent.genes[x].inNode]

    def defineInputNode(self, processedBoard):
        """Uses the coordinate stored in the input node and grabs the value from that coordinate on the board"""
        if self.pos[0] == -1: # Handling our bias node
            self.outValue = 1
        else:
            self.outValue = processedBoard[self.pos[0]][self.pos[1]]

class hiddenNode(object):
    def __init__(self, index, parent, value = 0):
        self.index = index
        self.inValue = value
        self.outValue = value
        self.button = None
        self.pos = None
        self.nType = "hid"
        self.parent = parent
        self.inLinks = []
        self.outLinks = []

    def refresh(self):
        self.inLinks = []
        self.outLinks = []

        # Assumes that parents had their genes updated to reflect the
        # new connections after inserting a new hidden node.
        for g in self.parent.genes.values():
            if g.outNode == self.index:
                self.inLinks.append(g)  # appending the gene
            if g.inNode == self.index:
                self.outLinks.append(g) # appending the gene to our out links

    def forward(self, i):
        return self.parent.nodes[self.parent.genes[i].outNode]
    def backward(self, i):
        return self.parent.nodes[self.parent.genes[x].inNode]


class outputNode(object):
    def __init__(self, index, parent, button, value = 0):
        self.index = index
        self.button = button
        self.inValue = value
        self.outValue = value
        self.pos = None
        self.nType = "out"
        self.parent = parent
        self.inLinks = []

    def refresh(self):
        self.inLinks = []
        for g in self.parent.genes.values():
            if g.outNode == self.index:
                self.inLinks.append(g)


# In[264]:

class Tron(object):
    def start(self, size):
        self.board = np.empty(size)
        #self.board.shape(size)
        self.board.fill(2)
        self.size = size
        self.AI = True
        width = self.size[0]
        height = self.size[1]
        minimumDifference = 15
        while True:
            self.p2 = random.randint(width / 4, 3 * (width / 4)), random.randint(height / 4, 3 * (height / 4))
            self.p1 = random.randint(width / 4, 3 * (width / 4)), random.randint(height / 4, 3 * (height / 4))
            if not(abs(self.p2[0] - self.p1[0]) < minimumDifference or abs(self.p2[1] - self.p1[1]) < minimumDifference):
                break
        self.d1 = (1, 0)
        self.d2 = (-1, 0)
        pygame.init()
        self.d = pygame.display.set_mode((self.size[0] * 4,self.size[1]*4))
        self.s = pygame.Surface(size)
        self.clock = pygame.time.Clock()
    
    def quit(self):
        pygame.quit()

    def add(self, a, b):
        return (a[0]+b[0], a[1]+b[1])

    def oob(self, p, w, h):
        if p[0] >= w or p[0] < 0 or p[1] >= h or p[1] < 0:
            return True
        return False

    def tick(self, move):
        self.oldd1 = self.d1
        if move == "U": self.d1 = (0, -1)
        if move == "D": self.d1 = (0, 1)
        if move == "L": self.d1 = (-1, 0)
        if move == "R": self.d1 = (1, 0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                playing = False
            if event.type == pygame.KEYDOWN:
                if event.key==pygame.K_w:
                    self.d2 = 0, -1
                elif event.key==pygame.K_s:
                    self.d2 = 0, 1
                elif event.key==pygame.K_a:
                    self.d2 = -1, 0
                elif event.key==pygame.K_d:
                    self.d2 = 1, 0

        if self.AI:
            self.d2 = ai(self.board, self.p1, self.p2, self.oldd1, self.d2, self.size[0], self.size[1])
        
        oldp2 = self.p2

        self.p1 = add(self.p1, self.d1)
        self.p2 = add(self.p2, self.d2)
        
        self.p1 = int(self.p1[0]), int(self.p1[1])
        self.p2 = int(self.p2[0]), int(self.p2[1])

        if self.p1 == self.p2:
            return self.p1, (0, 1)
        else:
            if self.oob(self.p1, self.size[0], self.size[1]) or self.board[self.p1[0]][self.p1[1]] != 2:
                return self.p1, (0, 1)
            else:
                self.s.set_at(self.p1, (0, 0, 255))
            if self.oob(self.p2, self.size[0], self.size[1]) or self.board[self.p2[0]][self.p2[1]] != 2:
                return self.p1, (1, 0)
            else:
                self.s.set_at(self.p2, (255, 0, 0))

        self.board[self.p1[0]][self.p1[1]] = 0
        self.board[self.p2[0]][self.p2[1]] = 1
        self.board[oldp2[0]][oldp2[1]] = -1
        
        self.clock.tick(30)

        pygame.transform.scale(self.s, (self.size[0] * 4, self.size[1] * 4), self.d)
        pygame.display.flip()

        return self.p1, (0,0)


# In[265]:
def choice(gen1, gen2):
    num = random.randint(0, 10)
    if num % 2 == 0:
        return gen1
    else:
        return gen2


def add(a, b):
    return (a[0]+b[0], a[1]+b[1])
def sub(a, b):
    return (a[0]-b[0], a[1]-b[1])

def oob(p, w, h):
    if p[0] >= w or p[0] < 0 or p[1] >= h or p[1] < 0:
        return True
    return False

def sign(n):
    if n > 0: return 1
    if n < 0: return -1
    return 0

def ai(board, opPos, yourPos, d1, d2, width, height):
    aim = x, y = sub(add(add(opPos, d1), d1), yourPos)
    if abs(x) > abs(y):
        optimal = sign(x), 0
        optimal2 = 0, sign(y)
    else:
        optimal = 0, sign(y)
        optimal2 = sign(x), 0
    if (not oob(add(yourPos, optimal), width, height)) and board[add(yourPos, optimal)[0]][add(yourPos, optimal)[1]] == 2:
        d2 = optimal
    elif (not oob(add(yourPos, optimal2), width, height)) and board[add(yourPos, optimal2)[0]][add(yourPos, optimal2)[1]] == 2:
        d2 = optimal2
    elif (not oob(add(yourPos, (optimal2[0] * -1, optimal2[1] * -1)), width, height)) and board[add(yourPos, (optimal2[0] * -1, optimal2[1] * -1))[0]][add(yourPos, (optimal2[0] * -1, optimal2[1] * -1))[0]] == 2:
        d2 = optimal2[0] * -1, optimal2[1] * -1
    elif (not oob(add(yourPos, (optimal[0] * -1, optimal[1] * -1)), width, height)) and board[add(yourPos, (optimal[0] * -1, optimal[1] * -1))[0]][add(yourPos, (optimal[0] * -1, optimal[1] * -1))[0]] == 2:
        d2 = optimal[0] * -1, optimal[1] * -1

    return d2


# In[266]:

def main():
    MAX_GENERATIONS = 3
    
    clear_file()

    
    trainingNets = neatNN()
    trainingNets.start()
    for generation in xrange(MAX_GENERATIONS):
        startTime = time.time()
        print "Generation", generation, "begin"
        maxF, avgF = trainingNets.learningLoop()
        endTime = time.time()
        record(generation, endTime - startTime, avgF, maxF)
    

main()


