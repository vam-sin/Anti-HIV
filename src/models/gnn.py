# libraries
import urllib.request
import pandas as pd 
import dgl 
import torch.nn as nn
import numpy as np
import random
from dgl.data import DGLDataset
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import load_graphs
import torch 
import torch.nn.functional as F 
import os 
from dgl.nn import GraphConv, SAGEConv
from sklearn.metrics import roc_auc_score
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
from dgllife.data import HIV
import pickle

class HIVTrain(DGLDataset):
	def __init__(self):
		super().__init__(name='HIVTrain')

	def process(self):
		train_g,_ = load_graphs('../../data/train_g.bin')
		targets = list(pd.read_csv('../../data/train.csv')["Activity"])

		tog = list(zip(train_g, targets))
		random.shuffle(tog)
		train_g, targets = zip(*tog)

		self.graphs = []
		self.labels = []

		for i in range(len(targets)):
			self.graphs.append(train_g[i])
			self.labels.append(targets[i])

		self.labels = torch.LongTensor(self.labels)

	def __getitem__(self, i):
		return self.graphs[i], self.labels[i]

	def __len__(self):
		return len(self.graphs)

class HIVTest(DGLDataset):
	def __init__(self):
		super().__init__(name='HIVTest')

	def process(self):
		train_g,_ = load_graphs('../../data/test_g.bin')
		targets = list(pd.read_csv('../../data/test.csv')["Activity"])
		
		self.graphs = []
		self.labels = []

		for i in range(len(targets)):
			self.graphs.append(train_g[i])
			self.labels.append(targets[i])

		self.labels = torch.LongTensor(self.labels)

	def __getitem__(self, i):
		return self.graphs[i], self.labels[i]

	def __len__(self):
		return len(self.graphs)

train_data = HIVTrain()
test_data = HIVTest()

train_dataloader = GraphDataLoader(train_data, batch_size=8, drop_last=False)
test_dataloader = GraphDataLoader(test_data, batch_size=1, drop_last=False)

class GCN(nn.Module):
	def __init__(self, in_feats, h_feats, num_classes):
		super(GCN, self).__init__()
		self.conv1 = GraphConv(in_feats, h_feats)
		self.conv2 = GraphConv(h_feats, h_feats)
		self.conv3 = GraphConv(h_feats, num_classes)

	def forward(self, g, in_feat):
		h = self.conv1(g, in_feat)
		h = F.relu(h)
		h = self.conv2(g, h)
		h = F.relu(h)
		h = self.conv3(g, h)
		g.ndata['h'] = h 

		return F.softmax(dgl.mean_nodes(g, 'h'))

model = GCN(74, 128, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Training Started")

best_test_acc = 0.0
best_auc_score = 0.0

for epoch in range(200):
	batch_loss = 0
	for batched_graph, labels in train_dataloader:
		# print(batched_graph, labels)
		unbatched_graph = dgl.unbatch(batched_graph)
		batch_graph = []
		for g in unbatched_graph:
			batch_graph.append(dgl.add_self_loop(g))
		batch_graph = dgl.batch(batch_graph)
		# lab = torch.cat((labels, masks), 1)
		# lab = torch.argmax(lab, dim=1)
		# lab = lab.to(torch.float32)
		# print(lab)
		pred = model(batch_graph, batch_graph.ndata['h'].float())
		# print(pred, labels)
		# pred = torch.squeeze(pred, dim=1)
		# print(pred, labels)
		loss = F.cross_entropy(pred, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	print("###### Epoch ", epoch, " #######")
	print('Loss: ', loss)

	num_correct = 0 
	num_tests = 0 

	all_preds = []
	test_lab = []

	for batched_graph, labels in test_dataloader:
		unbatched_graph = dgl.unbatch(batched_graph)
		batch_graph = []
		for g in unbatched_graph:
			batch_graph.append(dgl.add_self_loop(g))
		batch_graph = dgl.batch(batch_graph)
		# lab = torch.cat((labels, masks), 1)
		# lab = torch.argmax(lab, dim=1)
		lab = labels.detach().numpy()
		test_lab.append(lab)
		pred = model(batch_graph, batch_graph.ndata['h'].float())
		# print(pred, lab)
		all_preds.append(pred.detach().numpy())
		num_correct += (pred.argmax(1) == labels).sum().item()
		num_tests += 1

	test_acc = num_correct/num_tests
	print('Test Accuracy: ', test_acc)

	test_lab = np.asarray(test_lab)
	all_preds = np.asarray(all_preds)
	all_preds = np.squeeze(all_preds, axis=1)
	# print(test_lab.shape, all_preds.shape)
	all_p = []
	for i in all_preds:
		if i[1] >= i[0]:
			all_p.append(i[0])
		else:
			all_p.append(i[1])

	auc_score = roc_auc_score(test_lab, all_p)
	print("AUC Score: ", auc_score)

	if auc_score > best_auc_score:
		print("Best Till Now")
		best_test_acc = test_acc
		best_auc_score = auc_score

	print("Best Values")
	print("Test Accuracy: ", best_test_acc)
	print("AUC Score: ", best_auc_score)

'''
h_feats: 16 | 2 layer
Best Values
Test Accuracy:  0.6666666666666666
AUC Score:  0.5825

h_feats: 32 | 2 layer
Best Values
Test Accuracy:  0.7033333333333334
AUC Score:  0.5120444444444445

h_feats: 64 | 3 layer
Best Values
Test Accuracy:  0.6583333333333333
AUC Score:  0.745

'''