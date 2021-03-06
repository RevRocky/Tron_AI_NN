import pickle

def record(generationNumber, timeSpentTraining, averageFitness, maxFitness, fName):
  s = ""
  for d in [generationNumber, timeSpentTraining, averageFitness, maxFitness]:
    s += str(d) + ","
  with open(fName + ".txt", 'a') as f:
    f.write(s[:-2] + "\n")

def clear_file(fName, fName2):
  with open(fName + ".pkl", 'w') as f:
    f.write("")
  with open(fName2, 'w') as f:
    f.write("")

def pickleToDisk(population, fName):
  """This function pickles the entire population object hierarchy to disk"""
  pickle.dump(population, open(fName + ".pkl", 'wb'), protocol=2)

def readPickledFile(fName):
  return pickle.load(open(fName + ".pkl", 'rb'))
