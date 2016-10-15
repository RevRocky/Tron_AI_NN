import random

class neatNN(object):
		""" This class will aim to implement the NEAT algorithm to creating neural networks that can evolve structures capable of playing 
	a strong game of Tron. It stores both the nodes and the connection between the nodes. """

	def __init__(self):
		self.innovation = 0

	def breedGenome(parent1, parent2):
		"""ASSUMES node1 and node2 are compatable parent nodes. This method will go through each parent and select genes weights to inherit at
		random from each of the parent. Genes not expressed in either parent will be carried forth to the child node!"""
		
		# Conception
		child = genome()


		# Generating set with the union and intersect of A and B (all innovations in both A and B)
		innovationUnion 			= parent1.innovations | parent2.innovations
		innovationIntersect			= parent1.innovations & parent2.innovations
		
		# Generating sets with XOR of parent1, parent2
		parent1XOR					= parent1.innovations - parent2.innovations
		parent2XOR					= parent2.innovations - parent1.innovations

		for innovation in innovationUnion:

			if innovation in innovationIntersect:
				# Aliasing certain variables to be more easily obtained.
				inNode 			= parent1.genes[].inNode	# TODO how can I obtain the specific gene that I want to refer to
				outNode 		= parent1.genes[].outNode
				inheritedWeight = random.choice(parent1.genes[].weight, parent2.genes[].)


				child.addConnection()


class genome(object):
 	#initialInputs is a tuple, first value is number, second is a list of coord-pairs for the inputs. initialOutputs is a list of names
 	def __init__(self, father, initialInputs, initialOutputs, initialConnections): 
	self.nodes = []
	self.genes = []
	self.innovations = set()
	
	for i in range(len(initialInputs)):
		self.nodes.append(("in", i))
    
	for i in range(len(initialOutputs)):
		self.nodes.append(("out", i + len(initialInputs)))
	
	for i in range(initialConnections):
		self.genes.append(gene(random.randint(0, initialInputs[0] + 1), out, innovation))
		father.innovation += 1

	def addnode(self,)
		"""Adds a node outside the context of mutation"""

	def addConnection(self, node1, node2, weight, innovation)
		"""Adds a connection to the genome outside of the context of incrementing the innovation number"""
		self.genome.append(gene(inNode, outNode, innovation, weight)) # This should be sufficient to add a connection
			

	def mutateAddNode(self, ):
		"""                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        """

	def mutateAddConnexion(self, node1, node2, innovation):
		"""Adds a connexion between node and node two"""
		self.genome.append(gene(inNode, outNode, innovation))
		# TODO Figure out how to properly increment innovation counts....
		innovation 			 += 1 			# Increment our innovation number
		self.connectionCount += 1 			# 
		return innovation


	def flipConnection(connection):
		"""Flips a connection to being either enabled or disabled"""
		connection.connected = 	not(connection.connected)

class gene(object):
	"""Stores information regarding connexions between nodes."""

	def __init__(inNode, outNode, innovation, weight = random.random(-1, +1)):
		# Wow look at these fancy initialisations. 
		self.inNode 	= inNode
		self.outNode 	= outNode
		self.innovation = innovation
		self.weight 	= weight
		self.connected  = True