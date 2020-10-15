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

import csv
from itertools import zip_longest
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
		self.lin0 = torch.nn.Linear(4*78*62, 64).half() #4*78*62 if no erosion, else 4*78*39
		self.lin1 = torch.nn.Linear(64,32).half()
		self.lin2 = torch.nn.Linear(32,16).half()
		self.lin3 = torch.nn.Linear(16,5).half()
		self.ReLU = torch.nn.ReLU().half()
		self.soft = torch.nn.Softmax().half()
		self.flat = torch.nn.Flatten().half()
		self.drop = torch.nn.Dropout(0.5)
	
	def forward(self, x, y):
		size = 256 #- 92
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
		#print(res.shape)
		res = self.flat(res)
		#print(res.shape)
		res = self.lin0(res)
		res = self.ReLU(res)
		res = self.lin1(res)
		res = self.ReLU(res)
		res = self.lin2(res)
		res = self.ReLU(res)
		res = self.drop(res)
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
def convert2cytoscapeJSON(G, out):
    # load all nodes into nodes array
    final = {}
    final["items"] = []
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
        nx = {}
        nx["group"] = "edges"
        nx["data"]={}
        nx["data"]["id"]=str(edge[0])+str(edge[1])
        nx["data"]["label"]=G.edges[edge[0],edge[1]]["pos"]
        nx["data"]["proba"]=G.edges[edge[0],edge[1]]["proba"]
        nx["data"]["source"]=edge[0]
        nx["data"]["target"]=edge[1]
        final["items"].append(nx)
        
    with open(out, 'w') as outfile:
    	json.dump(final, outfile)
    return final


###############################################################

path = "./graphs2/"
nb_samples = len(os.listdir(path)) // 3 #useless, need to remove
dataset_train = myDataset(path, nb_samples, "Train")
b_size = 12
loader_train = DataListLoader(dataset_train, batch_size=b_size, shuffle=True)
dataset_val	= myDataset(path, nb_samples, "Val")
loader_val = DataListLoader(dataset_val, batch_size=b_size, shuffle=True)


#pause = input(".........")

model = EdgeModel()
model = myDataParallel(model)        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
weights = [0.8, 0.8, 0.8, 0.8, 0.1] 
class_weights=torch.HalfTensor(weights).cuda()
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = Adam16(model.parameters(), lr=0.0001)  


loss_history = []
acc_history = []
val_loss_history = []
val_acc_history = []
epochs = 30
max_val_acc = 0
for epoch in range(epochs):
	print("Epoch "+str(epoch+1)+"/"+str(epochs))
	start_time = time.time() 
	total_acc = 0
	total_f1 = np.array([0,0,0,0,0],dtype=np.float)
	tot_train_loss=0
	model.train()
	for i,data_list in enumerate(loader_train):

		optimizer.zero_grad()
		outdata, out, outsoft = model(data_list)
		y = torch.cat([data['y'] for data in data_list]).to(out.device)
		loss = criterion(out, torch.argmax(y, dim=1))
		output = outsoft.cpu().detach().numpy()
		amax = np.amax(output, axis=1)
		argmax = np.argmax(output, axis=1)
		gt = np.argmax(y.cpu().detach().numpy(), axis=1)
		#p = input("......")

		loss.backward()
		optimizer.step()
		
		tot_train_loss += loss.item()
		f1 = f1_score(gt, argmax, average=None)
		total_f1 += f1
		acc = balanced_accuracy_score(gt, argmax)
		total_acc += acc

	loss_history.append(tot_train_loss/float(i+1))
	acc_history.append(total_acc/float(i+1))
	
	print("\nTrain Loss: ",tot_train_loss/float(i+1))
	print("Train F1: ",total_f1/float(i+1))
	print("Train Acc: ",total_acc/float(i+1))	
		
	total_val_acc = 0
	total_val_f1 = np.array([0,0,0,0,0],dtype=np.float)
	tot_val_loss=0
	with torch.no_grad():
		model.eval()
		for j,data_list in enumerate(loader_val):

			optimizer.zero_grad()
			outdata, out, outsoft = model(data_list)
			y = torch.cat([data['y'] for data in data_list]).to(out.device)
			val_loss = criterion(out, torch.argmax(y, dim=1))
			output = outsoft.cpu().detach().numpy()
			amax = np.amax(output, axis=1)
			argmax = np.argmax(output, axis=1)
			gt = np.argmax(y.cpu().detach().numpy(), axis=1)
		
			tot_val_loss += val_loss.item()
			f1 = f1_score(gt, argmax, average=None)
			total_val_f1 += f1
			acc = balanced_accuracy_score(gt, argmax)
			total_val_acc += acc

		val_loss_history.append(tot_val_loss/float(j+1))
		val_acc_history.append(total_val_acc/float(j+1))
	
		print("Val Loss: ",tot_val_loss/float(j+1))
		print("Val F1: ",total_val_f1/float(j+1))
		print("Val Acc: ",total_val_acc/float(j+1))	
		print("\n")
		if total_val_acc/float(j+1) > max_val_acc:
			max_val_acc = total_val_acc/float(j+1)
			torch.save(model.state_dict(), "./model_trained_papyrus_.pt")
			print("model saved")
	print("Time (s): ",(time.time() - start_time))
	
d = [loss_history, val_loss_history,acc_history,val_acc_history]
export_data = zip_longest(*d, fillvalue = '')
with open('model_papyrus_.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
	  wr = csv.writer(myfile)
	  wr.writerow(("loss", "valloss","acc","valacc"))
	  wr.writerows(export_data)
myfile.close()

"""
plt.figure()
plt.plot(loss_history)
plt.plot(val_loss_history)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train loss','Val loss'], loc='upper left')
plt.figure()
plt.plot(acc_history)
plt.plot(val_acc_history)
plt.ylabel('Acc')
plt.xlabel('Epoch')
plt.legend(['Train acc','Val acc'], loc='lower left')
plt.show()
"""
