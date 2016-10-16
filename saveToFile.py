
# coding: utf-8

# In[ ]:

import pickle
import csv

# In[8]:

def record(generationNumber, timeSpentTraining, averageFitness, maxFitness):
    with open('record.csv', 'ab') as file:
        csvWriteHead = csv.writer(file, delimiter = ' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvWriteHead.writerow([generationNumber, timeSpentTraining, averageFitness, maxFitness])
        
def clear_file():
    with open('record.csv', 'wb') as file:
        csvWriteHead = csv.writer(file, delimiter = ' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvWriteHead.writerow('')
    
# In[ ]:

def pickletoDisk(population, mfile = "./Backups/Population"):
    """This function pickles the entire population object heiarchy to disk"""
    with open(mfile, 'wb') as inFile:
        pickle.dump(population, inFile)

def readPickledFile(mfile = "./Backups/Population"):
    with open(mfile, 'rb') as inFile:
        return pickle.load(inFile)  # This should return an original object heiarchy that would be assigned to NN_Neat.species
