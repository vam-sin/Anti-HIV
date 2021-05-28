import torch 
from dgllife.data import HIV
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
import pickle

dataset = HIV(smiles_to_bigraph, CanonicalAtomFeaturizer())

print(len(dataset))

smiles = []
class_one = []
class_two = []

for i in range(len(dataset)):
	print(i)
	print(dataset[i][2].numpy(), dataset[i][3].numpy())
	smiles.append(dataset[i][0])
	class_one.append(dataset[i][2].numpy())
	class_two.append(dataset[i][3].numpy())

filename = 'smiles.pickle'
outfile = open(filename,'wb')
pickle.dump(smiles,outfile)
outfile.close()

filename = 'class_one.pickle'
outfile = open(filename,'wb')
pickle.dump(class_one,outfile)
outfile.close()

filename = 'class_two.pickle'
outfile = open(filename,'wb')
pickle.dump(class_two,outfile)
outfile.close()