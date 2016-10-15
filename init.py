import random

class neatNN(object):
	__init__(self):
		self.innovation = 0

class genome(object):
 	__init__(self, father, initialInputs, initialOutputs, initialConnections): #initialInputs is a tuple, first value is number, second is a list of coord-pairs for the inputs. initialOutputs is a list of names
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
