import pickle
import pandas as pd 

filename = 'data/smiles.pickle'
infile = open(filename,'rb')
smiles = pickle.load(infile)
infile.close()

filename = 'data/class_one.pickle'
infile = open(filename,'rb')
class_one = pickle.load(infile)
infile.close()

filename = 'data/class_two.pickle'
infile = open(filename,'rb')
class_two = pickle.load(infile)
infile.close()

target = []

for i in range(len(class_two)):
	if class_one[i][0] == 0:
		target.append(1)
	else:
		target.append(0)

df = pd.DataFrame(list(zip(smiles, target)),
               columns =['SMILES', 'Activity'])

df.to_csv('data/HIV-Activity.csv')


