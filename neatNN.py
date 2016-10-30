# pylint: disable=redefined-builtin

from __future__ import division
from __future__ import print_function
from builtins import range
import math
import random
import numpy as np

outputs = ["U", "D", "L", "R"]   # The four legal moves in tron
DISJOINT = 1.0 # Constants for species assignment

WEIGHT = 0.4
INITIAL_GENOMES = 5
MUTATE_INPUT_THRESH = 5  # Constants for mutation chance out of 100
MUTATE_NODE_THRESH = MUTATE_INPUT_THRESH + 5
MUTATE_CONN_THRESH = MUTATE_NODE_THRESH + 5
EVAL_LOOP_MAX = 5

class neatNN(object):
  """ This class will aim to implement the NEAT algorithm to creating neural networks that can
  evolve structures capable of playing a strong game of Tron. It stores both the nodes and the
  connection between the nodes. """

  def __init__(self, framework):
    self.innovation = 0
    self.framework = framework
    self.size = self.framework.size
    self.species = []
    self.speciesReps = []

  def start(self, saveData = None):
    size = self.size
    if saveData is None:
      p1 = (size[0], size[1] - 1) # define positions of four possible starting input nodes
      p2 = (size[0] - 2, size[1] - 1)
      p3 = (size[0] - 1, size[1])
      p4 = (size[0] - 1, size[1] - 2)
      g = [genome(self, [p1, p2, p3, p4], outputs, 2)]
      self.unsortedNextGen = g * INITIAL_GENOMES
    else:
      self.unsortedNextGen = saveData

  def doGeneration(self):
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

    self.speciesReps = self.getSpeciesReps()
    self.species = self.sortToSpecies(self.unsortedNextGen) # Sorts each species into genomes

    array = []
    maxFitness = 0
    avgFitness = 0
    count = 0
    for species in self.species:
      specFitTot = 0
      for g in species:
        if self.framework.quitFlag:
          return 0,0,self.unsortedNextGen,None
        count += 1
        fitness, winning = self.evaluate(g)
        if winning:
          fitness += 20  # How much we increase fitness by.
          g.fitness = fitness
        specFitTot +=  fitness
        if fitness > maxFitness:
          maxFitness = fitness
          bestGenome = g
      avgFitness += specFitTot
      if len(species) > 0:
        array.append(specFitTot / len(species))
      else:
        array.append(0)

    avgFitness = avgFitness / count

    # Getting the relative fitnesses of each species!
    array = np.array(array)
    array = (array / np.amax(array)) / 0.4 + 0.15
    for i, p in enumerate(array):
      add = self.breedingControl(self.species[i], p)
      if add != []:
        self.unsortedNextGen += add

    for g in self.unsortedNextGen:
      num = random.uniform(0, 100)
      if num < MUTATE_INPUT_THRESH:
        g.mutateAddInput(self)
      elif num < MUTATE_NODE_THRESH:
        g.mutateAddNode(self)
      elif num < MUTATE_CONN_THRESH:
        g.mutateAddConnection(self)

    return maxFitness, avgFitness, self.unsortedNextGen, bestGenome

  def sortToSpecies(self, newGenomes):
    nextGeneration = [list([]) for _ in range(len(self.species))]

    i = 0
    while i < len(newGenomes):
      g = newGenomes[i]  # newGenomes is the unsorted list of next generation of genomes
      genomeSpecies = g.determineSpecies(self, self.speciesReps)
      if genomeSpecies != "NEW":
        nextGeneration[genomeSpecies].append(g)  # Assign the genome to the proper species
      else:
        nextGeneration.append([g])
        self.speciesReps.append(g) # Add a new species rep.
      i += 1
    return nextGeneration # nextGeneration is our table of tables!

  def calculateDistance(self, g1, g2):
    """Somewhat of a quick and dirty implementation of our Distance function
    as defined by Stanley and Miikkulainen, but ignoring the distinction between disjoint and excess
    genes, just using 'disjoint' to encompass both"""
    longestGenome = max(len(g1.innovations), len(g2.innovations))
    disjointGeneCount = len(g1.innovations ^ g2.innovations)  # Symmetric Difference
    averageWeightDiff = self.getAvgWeightDiff(g1, g2)  # Gets the average difference in weights
    distance = (DISJOINT*disjointGeneCount)/longestGenome + WEIGHT * averageWeightDiff
    return distance

  def getAvgWeightDiff(self, g1, g2):
    tot = 0
    num = 0
    for innovation in g1.innovations & g2.innovations:
      num += 1
      tot += abs(g1.genes[innovation].weight-g2.genes[innovation].weight)
    if num > 0:
      return tot / num
    else:
      return 0

  def cull(self, genome_list, speciesStrength):
    genomeWeights = [(x.fitness,x) for x in genome_list]
    sortedGenomes = list(reversed(sorted(genomeWeights,key=lambda x: x[0])))
    return sortedGenomes[0:int(len(sortedGenomes) * speciesStrength)]

  def evaluate(self, g):
    """Plays Tron using a NN architecture specified by the genome that is passed in.
    It will also get moves for the opponent"""
    tron = self.framework
    tron.start()
    headPos = (159, 159)
    gameWinner = False
    fitness = 0
    while not gameWinner:
      processedBoard = self.processBoard(tron.board, headPos)
      for _ in range(EVAL_LOOP_MAX):
        genomeMove = g.getMove(processedBoard)
      fitness = fitness + 1
      headPos, winner = tron.tick(genomeMove, g) #give the framework a move, and get the result.
      if 1 in winner:
        gameWinner = True
    # If the first cell in our tuple is 1 the NN has won
    nnWinner = winner[0]
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
    return sum(species)/len(species)

  def breedingControl(self, species, percentKept):
    """This is a method that oversees each stage of the breeding process.
    This is called from within a for loop in the main function!"""
    nextGenerationUnsorted = []

    # Iterate over each species
    breedingList = self.getBreedingPairs(species, percentKept)

    for breedingPair in breedingList: # breeding our actual input pairs
      nextGenerationUnsorted.append(self.breedGenome(breedingPair[0], breedingPair[1]))
    return nextGenerationUnsorted # An unsorted list of our next

  def getBreedingPairs(self, species, percentKept):
    culled = self.cull(species, percentKept)
    random.shuffle(culled)

    return [(culled[i][1],culled[(i+1)%len(culled)][1])  for i, _ in enumerate(culled)]

  def breedGenome(self, parent1, parent2):
    """ASSUMES parent1 and parent2 are compatible parent nodes. This method will go through each
    parent and select genes weights to inherit at random from each of the parent. Genes not
    expressed in either parent will be carried forth to the child node!"""

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

        options = [parent1.genes[innovation].weight, parent2.genes[innovation].weight]
        inherited = random.choice(options)
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

    reps = []
    for species in self.species:
      reps.append(random.choice(species))

    return reps

class genome(object):
  def __init__(self, parent, initialInputs, initialOutputs, initialConnections):
    self.nodes = []
    self.genes = {}
    self.innovations = set()
    self.fitness = 0
    self.maxNodeNumber = -1
    self.parent = parent

    for i, _ in enumerate(initialInputs):
      self.addNode(inputNode(self.maxNodeNumber + 1, self, initialInputs[i]))
      self.maxNodeNumber += 1

    for i, _ in enumerate(initialOutputs):
      self.addNode(outputNode(self.maxNodeNumber + 1, self, initialOutputs[i]))
      self.maxNodeNumber += 1

    for i in range(initialConnections):
      inNode = self.nodes[random.randint(0, len(initialInputs) - 1)]
      nodeNum = random.randint(len(initialInputs), len(initialInputs) + len(initialOutputs) - 2)
      outNode = self.nodes[nodeNum]
      self.addConnection(inNode, outNode, parent.innovation, random.uniform(-1, 1))
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

  def addConnection(self, inNode, outNode, innovation, weight):
    """Adds a connection to the genome without incrementing the global innovation variable"""
    self.genes[innovation] = gene(inNode, outNode, innovation, weight) # Add a connection
    self.genes[innovation].inNode.outLinks.append(self.genes[innovation])
    self.genes[innovation].outNode.inLinks.append(self.genes[innovation])
    self.innovations.add(innovation)
    inNode.refresh()
    outNode.refresh()

  def mutateAddNode(self, parent):
    """Adds a node, splitting a connection"""
    g = random.choice(list(self.genes.values()))  # Picks a random connction
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
    x = random.randint(-1, parent.size[0] * 2 - 1)  # -1 represents a bias unit
    if x == -1:  #if we have a bias unit
      y = -1
    else: #Regular input
      y = random.randint(0, parent.size[1] * 2 - 1)
    out = random.choice(self.nodes)
    count = 0
    while out.nType == "in" or count > 10:
      out = random.choice(self.nodes)
      count += 1
    if count > 10:
      return None
    node1 = self.addNode(inputNode(self.maxNodeNumber + 1, self, (x,y)))
    self.maxNodeNumber += 1
    self.addConnection(node1, out, parent.innovation, random.uniform(-1, 1))
    parent.innovation += 1

  def setEnable(self, connection, value):
    """Flips a connection to being either enabled or disabled"""
    connection.connected = value

  def determineSpecies(self, father, speciesReps):
    """Compares the genome to representative members of each species. The first species that
    satisfies the threshold will be the species that the genome is sorted into."""

    DELTA_THRESH = 50   # To do

    for species, rep in enumerate(speciesReps):
      delta = father.calculateDistance(self, rep)
      if delta < DELTA_THRESH:
        return species

    return "NEW" # Signal to the calling environment to create a new species.

  def rewardWinning(self):
    """If the genome is a winner it will increase the fitness by a certain amount"""
    self.fitness += 400  # Increments fitness by what_ever fitness value we deem appropriate

  def getMove(self, processedBoard):
    """Feed forward one tick at a time of our neural network. Needs to save data between ticks."""
    # Creating lists of my input and output nodes. This should probably be done elsewhere
    outputNodes = []
    for node in self.nodes:
      if node.nType == 'in':
        node.getInputValue(processedBoard)  # Defines an actual value for our input node
      elif node.nType == 'out':
        outputNodes.append(node)
    # Now we make just one step each time through our neural network.
    for node in self.nodes:
      if node.nType != "out":
        for g in node.outLinks:
          theta = g.weight # aliasing our theta value
          g.outNode.inValue += (g.inNode.outValue * theta)

    for node in self.nodes:
      if node.nType != 'in':
        node.outValue = self.sigmoid(node.inValue)  # Copies the output value into our input value
        node.inValue = 0  # Resetting the inValue of an input node

    return self.maxNode(outputNodes).button  # Return the ideal move

  def maxNode(self, nodes):
    """Assumes that nodes is an iterable containing node Objects.
    Max Node is the node with the biggest output value."""

    maxNodeValue = -100

    for node in nodes:
      if node.outValue > maxNodeValue:
        maxNode = node
    return maxNode  # This is the node with the biggest value.

class gene(object):
  """Stores information regarding connexions between nodes."""

  def __init__(self, inNode, outNode, innovation, weight, enabled = True):
    # Wow look at these fancy initializations.
    self.inNode = inNode
    self.outNode = outNode
    self.innovation = innovation
    self.weight = weight
    self.connected  = enabled

class inputNode(object):
  def __init__(self, index, parent, pos, value = 0):
    self.index = index
    self.pos = pos # relative to top-left corner of processed board, (-1, -1) signifies a bias node
    self.inValue = 0
    self.button = None
    self.outValue = value # This is just to make code play nicely with each other
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
    return self.parent.nodes[self.parent.genes[i].inNode]

  def getInputValue(self, processedBoard):
    """Gets the incoming value for an input node"""
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
    return self.parent.nodes[self.parent.genes[i].inNode]

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
