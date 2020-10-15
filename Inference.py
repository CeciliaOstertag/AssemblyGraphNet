import torch
import numpy as np
import torch_geometric
from torch_geometric.nn import DataParallel
from torch_geometric.data import Data, DataLoader, DataListLoader, Dataset

#from torch_scatter import scatter_mean

from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.utils import shuffle

from matplotlib import pyplot as plt
import networkx as nx
import json
import cv2
import os
from itertools import chain
import random
import time
import re


#os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.backends.cudnn.enabled == True
torch.backends.cudnn.benchmark = True

import math
from torch.optim.optimizer import Optimizer


class Adam16(Optimizer):
	# Source: https://gist.github.com/ajbrock/075c0ca4036dc4d8581990a6e76e07a3
	# This version of Adam keeps an fp32 copy of the parameters and 
	# does all of the parameter updates in fp32, while still doing the
	# forwards and backwards passes using fp16 (i.e. fp16 copies of the 
	# parameters and fp16 activations).
	#
	# Note that this calls .float().cuda() on the params such that it 
	# moves them to gpu 0--if you're using a different GPU or want to 
	# do multi-GPU you may need to deal with this.
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        params = list(params)
        super(Adam16, self).__init__(params, defaults)
        # for group in self.param_groups:
            # for p in group['params']:
        
        self.fp32_param_groups = [p.data.float().to(p.device) for p in params]
        if not isinstance(self.fp32_param_groups[0], dict):
            self.fp32_param_groups = [{'params': self.fp32_param_groups}]

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group,fp32_group in zip(self.param_groups,self.fp32_param_groups):
            for p,fp32_p in zip(group['params'],fp32_group['params']):
                if p.dtype == torch.float32: # do not convert batch norm weights to float16
                	continue
                if p.grad is None:
                    continue
                    
                grad = p.grad.data.float()
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], fp32_p)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
            
                # print(type(fp32_p))
                fp32_p.addcdiv_(-step_size, exp_avg, denom)
                p.data = fp32_p.half()

        return loss

class myDataParallel(DataParallel):
	def forward(self, data_list):
		if len(data_list) == 0:
			logging.warning('DataParallel received an empty data list, which may result in unexpected behaviour.')
			return None

		if not self.device_ids or len(self.device_ids) == 1:  # Fallback
			data = Batch.from_data_list(data_list, follow_batch=self.follow_batch).to(self.src_device)
			return self.module(data)

		for t in chain(self.module.parameters(), self.module.buffers()):
			if t.device != self.src_device:
				raise RuntimeError(('Module must have its parameters and buffers on device {} but found one of them on device {}.').format(self.src_device, t.device))

		inputs = self.scatter(data_list, self.device_ids)
		replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
		outputs = self.parallel_apply(replicas, inputs, None)
		gathered = self.gather(outputs, self.output_device)
		return outputs, gathered[1], gathered[2] #list of graphs, result, resultsoft
            
class myData(Data):
	def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, pos=None, norm=None, face=None, **kwargs):
		self.x = x
		self.edge_index = edge_index
		self.edge_attr = edge_attr
		self.y = y
		self.pos = pos
		self.norm = norm
		self.face = face
		for key, item in kwargs.items():
			if key == 'num_nodes':
				self.__num_nodes__ = item
			else:
				self[key] = item


		if torch_geometric.is_debug_enabled():
			self.debug()

class myDataset(Dataset):
	def __init__(self, root_dir, nb_samples=None, mode="Train"):
		self.root_dir = root_dir
		self.list_files = []
		self.list = os.listdir(self.root_dir)
		self.list.sort(key=lambda f: (int(re.match(r".*png_(\d*).*(\w*)\.pt",f).group(1)),re.match(r".*png_(\d*)(\w*)\.pt",f).group(2)))

		for i in range(0,len(self.list),3):
			attr_file = self.list[i]
			idx_file = self.list[i+1]
			x_file = self.list[i+2]
			self.list_files.append((attr_file, idx_file, x_file))
		
		random.seed(42)
		random.shuffle(self.list_files)
		self.datatest = self.list_files[:200]
		self.dataval = self.list_files[200:700:]
		self.datatrain = self.list_files[700:]
		
		if mode == "Test":
			self.list_files = self.datatest
		elif mode == "Val":
			self.list_files = self.dataval
		else:
			self.list_files = self.datatrain
		print("Mode: "+mode)
		print("Nb of samples: "+str(len(self.list_files)))	
			
	def __len__(self):
		return len(self.list_files)
		
	def __getitem__(self, idx):
		edge_attr_gt = torch.load(self.root_dir+self.list_files[idx][0])
		edge_index = torch.load(self.root_dir+self.list_files[idx][1]).numpy()
		edge_index = np.swapaxes(edge_index,0,1)
		edge_index, edge_attr_gt = shuffle(edge_index, edge_attr_gt)
		edge_index = np.swapaxes(edge_index,0,1)
		edge_index = torch.tensor(edge_index, dtype=torch.uint8)
		x = torch.load(self.root_dir+self.list_files[idx][2])
		
		l = 46
		#x = x[:,l:x.shape[1]-l,l:x.shape[2]-l,:]
		
		shape = (edge_attr_gt.shape[0],edge_attr_gt.shape[1])
		ones = np.ones(shape)
		dummy_edge_attr = torch.tensor(ones, dtype=torch.uint8)

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		data = myData(edge_index = edge_index.to(device), edge_attr = dummy_edge_attr.to(device), x = x.to(device), y = edge_attr_gt.to(device))
		del edge_attr_gt
		del edge_index
		del x
		return data
		
class myMixedDataset(Dataset):
	def __init__(self, root_dir, nb_samples=None, mode="Test"):
		self.root_dir = root_dir
		self.list_files = []
		self.list = os.listdir(self.root_dir)
		self.list.sort(key=lambda f: (int(re.match(r".*png_(\d*).*(\w*)\.pt",f).group(1)),re.match(r".*png_(\d*)(\w*)\.pt",f).group(2)))

		for i in range(0,len(self.list),3):
			attr_file = self.list[i]
			idx_file = self.list[i+1]
			x_file = self.list[i+2]
			self.list_files.append((attr_file, idx_file, x_file))
		
		random.seed(42)
		random.shuffle(self.list_files)
		self.datatest = self.list_files[:200]
		self.dataval = self.list_files[200:700:]
		self.datatrain = self.list_files[700:]
		
		if mode == "Test":
			print("Testing with Mixed dataset. Warning: Ground Truth will be meaningless")
			self.list_files = self.datatest
		else:
			print("Error: Mixed dataset only usable for testing")
			self.list_files = None
		print("Mode: "+mode)
		print("Nb of samples: "+str(len(self.list_files)))	
			
	def __len__(self):
		return len(self.list_files)
		
	def __getitem__(self, idx):
		edge_attr_gt_0 = torch.load(self.list_files[idx][0]).numpy()
		edge_index_0 = torch.load(self.list_files[idx][1]).numpy()
		edge_attr_gt_1 = torch.load(self.list_files[idx+1][0]).numpy()
		edge_index_1 = torch.load(self.list_files[idx+1][1]).numpy()
		edge_index_1 += 15
		
		edge_index = np.concatenate((edge_index_0,edge_index_1),axis=1)
		edge_attr_gt = np.concatenate((edge_attr_gt_0,edge_attr_gt_1))
		
		edge_index = np.swapaxes(edge_index,0,1)
		edge_index, edge_attr_gt = shuffle(edge_index, edge_attr_gt)
		edge_index = np.swapaxes(edge_index,0,1)
		edge_index = torch.tensor(edge_index, dtype=torch.uint8)
		edge_attr_gt = torch.tensor(edge_attr_gt, dtype=torch.uint8)
		x_0 = torch.load(self.list_files[idx][2]).numpy()
		x_1 = torch.load(self.list_files[idx+1][2]).numpy()
		#r = np.random.choice([True, False], len(x_0))
		#print(r)
		#x = np.array([x_0[i] if r[i] == True else x_1[i] for i in range(len(x_0))]) # patches are either from image 0 or image 1
		x = np.concatenate((x_0,x_1))
		x = torch.tensor(x, dtype=torch.uint8)
		
		shape = (edge_attr_gt.shape[0],edge_attr_gt.shape[1])
		ones = np.ones(shape)
		dummy_edge_attr = torch.tensor(ones, dtype=torch.uint8)

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		data = myData(edge_index = edge_index.to(device), edge_attr = dummy_edge_attr.to(device), x = x.to(device), y = edge_attr_gt.to(device))
		del edge_attr_gt
		del edge_index
		del x
		return data

class ConvNet(torch.nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		self.BN1 = torch.nn.BatchNorm2d(3)
		self.C1 = torch.nn.Conv2d(3,4,3).half()
		self.BN2 = torch.nn.BatchNorm2d(4)
		self.C2 = torch.nn.Conv2d(4,4,3).half()
		self.BN3 = torch.nn.BatchNorm2d(4)
		self.C3 = torch.nn.Conv2d(4,4,3).half()
		self.BN4 = torch.nn.BatchNorm2d(4)
		self.C4 = torch.nn.Conv2d(4,8,3).half()
		self.BN5 = torch.nn.BatchNorm2d(8)
		self.C5 = torch.nn.Conv2d(8,8,3).half()
		self.BN6 = torch.nn.BatchNorm2d(8)
		self.reLU = torch.nn.ReLU().half()
		self.MP = torch.nn.MaxPool2d(2).half()
	
	def forward(self, x):
		#print("!")
		#print("1 ",x.shape)
		x = self.BN1(x)
		x = self.C1(x)
		x = self.reLU(x)
		x = self.MP(x)
		
		#print("2 ",x.shape)
		x = self.BN2(x)
		x = self.C2(x)
		x = self.reLU(x)
		x = self.MP(x)
		#print("3 ",x.shape)
		x = self.BN3(x)
		x = self.C3(x)
		x = self.reLU(x)
		x = self.MP(x)
		#print("4 ",x.shape)
		x = self.BN4(x)
		x = self.C4(x)
		x = self.reLU(x)
		x = self.MP(x)
		
		#print("5 ",x.shape)
		x = self.BN5(x)
		x = self.C5(x)
		x = self.reLU(x)
		x = self.MP(x)
		x = self.BN6(x)

		return x

class ConvNet2(torch.nn.Module):
	def __init__(self):
		super(ConvNet2, self).__init__()
		self.BN1 = torch.nn.BatchNorm2d(3)
		self.C1 = torch.nn.Conv2d(3,4,3).half()
		self.BN2 = torch.nn.BatchNorm2d(4)
		self.C2 = torch.nn.Conv2d(4,4,3).half()
		self.BN3 = torch.nn.BatchNorm2d(4)
		self.reLU = torch.nn.ReLU().half()
		self.MP = torch.nn.MaxPool2d(2).half()
	
	def forward(self, x):
		#print("!")
		#print("1 ",x.shape)
		x = self.BN1(x)
		x = self.C1(x)
		x = self.reLU(x)
		x = self.MP(x)
		
		#print("2 ",x.shape)
		x = self.BN2(x)
		x = self.C2(x)
		x = self.reLU(x)
		x = self.MP(x)
		#print("3 ",x.shape)
		x = self.BN3(x)

		return x

class NodeModel(torch.nn.Module):
	def __init__(self):
		super(NodeModel, self).__init__()
		self.net = ConvNet()
		self.lin0 = torch.nn.Linear(8*6*6, 64).half()
		self.lin1 = torch.nn.Linear(64,32).half()
		self.lin2 = torch.nn.Linear(32,16).half()
		self.lin3 = torch.nn.Linear(16,5).half()
		self.ReLU = torch.nn.ReLU().half()
		self.soft = torch.nn.Softmax().half()
		self.flat = torch.nn.Flatten().half()
	
	def forward(self, x, y):
		size = 256
		
		x = x.view(-1,3,size,size)
		y = y.view(-1,3,size,size)
		#print(x.shape) #pb with images with a lot of edges, need to scale images better
		#p = input("....")
		res_1 = self.net(x)
		res_2 = self.net(y)
		del x
		del y
		res = res_1 - res_2
		res = torch.abs(res)
		res = self.flat(res)
		#res = res_1 - res_2
		del res_1
		del res_2
		res = self.lin0(res)
		res = self.ReLU(res)
		res = self.lin1(res)
		res = self.ReLU(res)
		res = self.lin2(res)
		res = self.ReLU(res)
		res = self.lin3(res)
		#print(res.shape)
		ressoft = self.soft(res)
		return res, ressoft

class NodeModel2(torch.nn.Module):
	def __init__(self):
		super(NodeModel2, self).__init__()
		self.net = ConvNet2()
		self.lin0 = torch.nn.Linear(4*78*62, 64).half()
		self.lin1 = torch.nn.Linear(64,32).half()
		self.lin2 = torch.nn.Linear(32,16).half()
		self.lin3 = torch.nn.Linear(16,5).half()
		self.ReLU = torch.nn.ReLU().half()
		self.soft = torch.nn.Softmax().half()
		self.flat = torch.nn.Flatten().half()
	
	def forward(self, x, y):
		size = 256
		thic = 40
		x = x.view(-1,3,size,size)
		y = y.view(-1,3,size,size)
		x11 = x[:, :, 0:thic,:]
		y22 = y[:, :, -thic:,:]
		x12 = x[:, :, -thic:,:]
		y21 = y[:, :, 0:thic,:]
		x13 = x[:, :,:,0:thic]
		y24 = y[:, :,:,-thic:]
		x14 = x[:, :,:,-thic:]
		y23 = y[:, :,:,0:thic]
		
		del x
		del y
		H = torch.cat([y22, x11], 2)
		B = torch.cat([x12, y21], 2)
		G = torch.cat([y24, x13], 3)
		G = torch.rot90(G, 1, [2, 3])
		D = torch.cat([x14, y23], 3)
		D = torch.rot90(D, 1, [2, 3])
		res = torch.cat([H, B, G, D], 2)
		res = self.net(res)
		res = self.flat(res)
		#print(res.shape)
		res = self.lin0(res)
		res = self.ReLU(res)
		res = self.lin1(res)
		res = self.ReLU(res)
		res = self.lin2(res)
		res = self.ReLU(res)
		res = self.lin3(res)
		#print(res.shape)
		ressoft = self.soft(res)
		return res, ressoft		

class EdgeModel(torch.nn.Module):
	def __init__(self):
		super(EdgeModel, self).__init__()
		self.convnet = NodeModel2()

	def forward(self, data):
		# source, target: [E, F_x], where E is the number of edges.
		# edge_attr: [E, F_e]
		# u: [B, F_u], where B is the number of graphs.
		# batch: [E] with max entry B - 1.
		row, col = data['edge_index']
		src = data['x'][row.to(torch.long)]
		dest = data['x'][col.to(torch.long)]
		src = src / 255.
		dest = dest / 255.
		src = src.to(torch.half)
		dest = dest.to(torch.half)
		#plt.imshow(src[1000].cpu().numpy())
		#plt.show()
		
		result, resultsoft = self.convnet(src, dest)
		del row
		del col
		del src
		del dest

		outdata = myData(edge_index = data['edge_index'], edge_attr = result, x = data['x'])
		return outdata, result, resultsoft


def to_networkx(data, node_attrs=None, edge_attrs=None, edge_probs=None, to_undirected=False, remove_self_loops=False):
	if to_undirected:
		G = nx.Graph()
	else:
		G = nx.DiGraph()

	G.add_nodes_from(range(data.num_nodes))
	values = {key: data[key].squeeze().tolist() for key in data.keys}
	edge_index = data['edge_index']
	labels = {0:"H", 1:"B", 2:"G", 3:"D", 4:"_"}

	i = 0
	for k in range(edge_index.shape[1]):
		G.add_edge(edge_index[0,k].item(), edge_index[1,k].item())
		G[edge_index[0,k].item()][edge_index[1,k].item()]["pos"] = labels[edge_attrs[i]]
		G[edge_index[0,k].item()][edge_index[1,k].item()]["proba"] = edge_probs[i]
		i = i + 1

	for j, node in enumerate(G.nodes(data=True)):
		node[1]["img"] = node_attrs[j,:,:]

	return G
	
# this function is used to convert networkx to Cytoscape.js JSON format
# returns string of JSON
def convert2cytoscapeJSON(G, out, filter_thr=0.7):
	# load all nodes into nodes array
	final = {}
	final["items"] = []
	deleted = 0
	for node in G.nodes():
		nx = {}
		nx["group"] = "nodes"
		nx["data"] = {}
		nx["data"]["id"] = node
		nx["data"]["label"] = node
		nx["position"] = {"x":G.nodes[node]["coord"][0]*10, "y":-G.nodes[node]["coord"][1]*10}
		final["items"].append(nx.copy())
	#load all edges to edges array
	for edge in G.edges():
		pb = G.edges[edge[0],edge[1]]["proba"].round(2)
		if pb > filter_thr:
			nx = {}
			nx["group"] = "edges"
			nx["data"]={}
			nx["data"]["id"]="E"+str(edge[0])+str(edge[1])
			nx["data"]["label"]=G.edges[edge[0],edge[1]]["pos"]
			nx["data"]["proba"]=G.edges[edge[0],edge[1]]["proba"].round(2)
			nx["data"]["source"]=edge[0]
			nx["data"]["target"]=edge[1]
			final["items"].append(nx)
		else:
			deleted += 1
	print("Filtered "+str(deleted)+" edges: proba < "+str(filter_thr))
	with open(out, 'w') as outfile:
		outfile.write("var input = ")
	with open(out, 'a') as outfile:
		json.dump(final, outfile)
	return final


###############################################################

path = "./graphs2/"
trained_model = None
if path == "./graphs3/":
	print("Using Natural images")
	trained_model = "model_trained.pt"
elif path == "./graphs2/":
	print("Using Papyrus images")
	trained_model = "model_trained_papyrus_.pt"	
else:
	print("Error: Unknown dataset, No model trained for this dataset")
	
nb_samples = len(os.listdir(path)) // 3
b_size = 1
#dataset_test = myMixedDataset(path, nb_samples, "Test") # mix two images
dataset_test = myDataset(path, nb_samples, "Test")
loader_test = DataLoader(dataset_test, batch_size=b_size, shuffle=True)


#pause = input(".........")

model = EdgeModel()
state_dict = torch.load(trained_model)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
weights = [1., 1., 1., 1., 0.1] 
class_weights=torch.HalfTensor(weights).cuda()
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

with torch.no_grad():
	model.eval()  


	i=0
	t_all = 0
	total_test_acc = 0
	total_test_f1 = np.array([0,0,0,0,0],dtype=np.float)
	tot_test_loss=0
	for data_test in loader_test:
		start_time = time.time()
		outdata, out, outsoft = model(data_test)
		t = time.time() - start_time
		print("Time (s): ",t)
		if i > 0:
			t_all += t
		y = data_test['y']
		test_loss = criterion(out, torch.argmax(y, dim=1))
		output = outsoft.cpu().detach().numpy()
		amax = np.amax(output, axis=1)
		argmax = np.argmax(output, axis=1)
		gt = np.argmax(y.cpu().detach().numpy(), axis=1)
		f1 = f1_score(gt, argmax, average=None)
		acc = balanced_accuracy_score(gt, argmax)
		tot_test_loss += test_loss.item()
		f1 = f1_score(gt, argmax, average=None)
		total_test_f1 += f1
		acc = balanced_accuracy_score(gt, argmax)
		total_test_acc += acc
		print("F1 per class ",f1)
		print("Balanced accuracy ",acc)
		i += 1

print("Test Loss: ",tot_test_loss/float(i+1))
print("Test F1: ",total_test_f1/float(i+1))
print("Test Acc: ",total_test_acc/float(i+1))
print("Average execution time (s): ",t_all/(i)) #exclude first batch, way longer than the rest


edge_attributes = np.asarray([val for val in argmax],dtype=np.float)
edge_probas = np.asarray([val for val in amax],dtype=np.float)
edge_attributes_targ = np.asarray([val for val in gt],dtype=np.float)
outgraph = myData(x = outdata['x'],edge_index = outdata['edge_index'],edge_attr = edge_attributes)
G = to_networkx(outgraph, node_attrs=outgraph.x, edge_attrs=edge_attributes, edge_probs=edge_probas)
pb = np.ones(edge_probas.shape)
data_targ = myData(x = data_test['x'],edge_index = data_test['edge_index'],edge_attr = edge_attributes_targ)
G_targ = to_networkx(data_targ, node_attrs=data_targ.x, edge_attrs=edge_attributes_targ, edge_probs = pb)

# Draw reconstructed image using the corrdinates of patches

fig=plt.figure(figsize=(10,10))
ax=plt.subplot()
ax.set_aspect('auto')

pos = {}
toRemove = []
for node_idx in G.nodes():
	#print("idx ", node_idx)
	#if node_idx not in pos.keys():
	origin = [0,0]
	if node_idx == 0: #origine
		G.nodes[node_idx]["coord"] = origin
		pos[node_idx] = G.nodes[node_idx]["coord"]
	for out_edge in G.out_edges(node_idx, data=True):
		#print(out_edge)
		if out_edge[2]["pos"] == "_":
			#print("nope")
			toRemove.append((out_edge[0],out_edge[1]))
			origin[0] += 5
			G.nodes[out_edge[1]]["coord"] = origin
			pos[out_edge[1]] = G.nodes[out_edge[1]]["coord"]
		else:
			if ("coord" not in G.nodes[out_edge[0]]) : #new node, not linked to any previous ones
				origin[0] += 5
				G.nodes[out_edge[0]]["coord"] = origin
				pos[out_edge[0]] = G.nodes[out_edge[0]]["coord"]
			if ("coord" not in G.nodes[out_edge[1]]) : #or (G.nodes[out_edge[1]]["coord"] == [0,0]):
				#print("S "+str(out_edge[0])+" T "+str(out_edge[1]))
				G.nodes[out_edge[1]]["coord"] = [0,0]
				#print( G.nodes[out_edge[0]]["coord"])
				if out_edge[2]["pos"] == "H":
					G.nodes[out_edge[1]]["coord"][0] = G.nodes[out_edge[0]]["coord"][0]
					G.nodes[out_edge[1]]["coord"][1] = G.nodes[out_edge[0]]["coord"][1] + 1
				elif out_edge[2]["pos"] == "B":
					G.nodes[out_edge[1]]["coord"][0] = G.nodes[out_edge[0]]["coord"][0]
					G.nodes[out_edge[1]]["coord"][1] = G.nodes[out_edge[0]]["coord"][1] - 1
				elif out_edge[2]["pos"] == "G":
					G.nodes[out_edge[1]]["coord"][0] = G.nodes[out_edge[0]]["coord"][0] - 1
					G.nodes[out_edge[1]]["coord"][1] = G.nodes[out_edge[0]]["coord"][1] 
				elif out_edge[2]["pos"] == "D":
					G.nodes[out_edge[1]]["coord"][0] = G.nodes[out_edge[0]]["coord"][0] + 1
					G.nodes[out_edge[1]]["coord"][1] = G.nodes[out_edge[0]]["coord"][1] 
				pos[out_edge[1]] = G.nodes[out_edge[1]]["coord"]
				#print(G.nodes[out_edge[1]]["coord"])
				#p = input("....")
			else:
				continue

print("Nb of edges: ",len(G.edges()))
print("Removing: ",len(toRemove))
G.remove_edges_from(toRemove)
print("Nb of edges: ",len(G.edges()))

jsonG = convert2cytoscapeJSON(G,'graph.json', filter_thr=0.8)
"""
pos = {}
toRemove = []
for node_idx in G_targ.nodes():
	print(node_idx)
	#if node_idx not in pos.keys():
	if node_idx == 0: #origine
		G_targ.nodes[node_idx]["coord"] = [0,0]
		pos[node_idx] = G_targ.nodes[node_idx]["coord"]
	for out_edge in G_targ.out_edges(node_idx, data=True):
		if out_edge[2]["pos"] == "_":
			toRemove.append((out_edge[0],out_edge[1]))
		else:
			if ("coord" not in G_targ.nodes[out_edge[1]]) : #or (G.nodes[out_edge[1]]["coord"] == [0,0]):
				#print("S "+str(out_edge[0])+" T "+str(out_edge[1]))
				G_targ.nodes[out_edge[1]]["coord"] = [0,0]
				if out_edge[2]["pos"] == "H":
					G_targ.nodes[out_edge[1]]["coord"][0] = G_targ.nodes[out_edge[0]]["coord"][0]
					G_targ.nodes[out_edge[1]]["coord"][1] = G_targ.nodes[out_edge[0]]["coord"][1] + 1
				elif out_edge[2]["pos"] == "B":
					G_targ.nodes[out_edge[1]]["coord"][0] = G_targ.nodes[out_edge[0]]["coord"][0]
					G_targ.nodes[out_edge[1]]["coord"][1] = G.nodes[out_edge[0]]["coord"][1] - 1
				elif out_edge[2]["pos"] == "G":
					G_targ.nodes[out_edge[1]]["coord"][0] = G_targ.nodes[out_edge[0]]["coord"][0] - 1
					G_targ.nodes[out_edge[1]]["coord"][1] = G_targ.nodes[out_edge[0]]["coord"][1] 
				elif out_edge[2]["pos"] == "D":
					G_targ.nodes[out_edge[1]]["coord"][0] = G_targ.nodes[out_edge[0]]["coord"][0] + 1
					G_targ.nodes[out_edge[1]]["coord"][1] = G_targ.nodes[out_edge[0]]["coord"][1] 
				pos[out_edge[1]] = G_targ.nodes[out_edge[1]]["coord"]
				#print(G.nodes[out_edge[1]]["coord"])
				#p = input("....")
			else:
				continue

print("Nb of edges: ",len(G_targ.edges()))
print("Removing: ",len(toRemove))
G_targ.remove_edges_from(toRemove)
print("Nb of edges: ",len(G_targ.edges()))

jsonG_targ = convert2cytoscapeJSON(G_targ, 'graph_targ.json')
"""
nx.draw_networkx_nodes(G,pos)
labels = {0:"p1",1:"p2",2:"p3"}
nx.draw_networkx_labels(G,pos, labels=labels)
nx.draw_networkx_edges(G,pos,ax=ax, arrowstyle='->', arrowsize=5, node_size=10)
edge_labels = dict([((u,v,),d['pos']) for u,v,d in G.edges(data=True)])
nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels, label_pos=0.3)

trans=ax.transData.transform
trans2=fig.transFigure.inverted().transform

piesize=0.1 # this is the image size
p2=piesize/2.0
for node in G.nodes(data=True):
	xx,yy=trans(node[1]["coord"]) # figure coordinates
	xa,ya=trans2((xx,yy)) # axes coordinates
	a = plt.axes([xa-p2,ya-p2, piesize, piesize])
	a.set_aspect('equal')
	try:
		img=node[1]["img"]
		img = img.cpu().numpy()
		B = img[:,:,0]
		G = img[:,:,1]
		R = img[:,:,2]
		img = np.stack((R,G,B),axis=2)
		cv2.imwrite("./patches/"+str(node[0])+".jpg", img)
		#a.imshow(img,cmap="gray")
		a.axis('off')
	except KeyError:
		print("img error")
		continue



