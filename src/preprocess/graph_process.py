from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
import pandas as pd 
import dgl 
from dgl.data.utils import save_graphs

# import data
train = pd.read_csv('../../data/train.csv')
train_smiles = train["SMILES"]
test = pd.read_csv('../../data/test.csv')
test_smiles = test["SMILES"]

# make graphs from smiles
node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
edge_featurizer = CanonicalBondFeaturizer(bond_data_field='h')

train_g = []
test_g = [] 

for i in range(len(train_smiles)):
	print(i)
	g = smiles_to_bigraph(smiles=train_smiles[i],node_featurizer=node_featurizer,edge_featurizer=edge_featurizer)
	g = dgl.add_self_loop(g)
	train_g.append(g)

for i in range(len(test_smiles)):
	print(i)
	g = smiles_to_bigraph(smiles=test_smiles[i],node_featurizer=node_featurizer,edge_featurizer=edge_featurizer)
	g = dgl.add_self_loop(g)
	test_g.append(g)

# g_train_batch = dgl.batch(train_g)
# g_test_batch = dgl.batch(test_g)

save_graphs("../../data/train_g.bin", train_g)
save_graphs("../../data/test_g.bin", test_g)

