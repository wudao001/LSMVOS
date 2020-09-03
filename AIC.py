
import torch.nn as nn
import torch
class AIC(nn.Module):
	"""docstring for AIC"""
	def __init__(self,input,hidden):
		super(AIC, self).__init__()
		self.conv1 = nn.Conv2d(input, hidden, kernel_size=3, padding=1)
		self.x1 = nn.Conv2d(hidden, hidden, kernel_size=[1,3], padding=[0,1])
		self.x2 = nn.Conv2d(hidden, hidden, kernel_size=[1,5], padding=[0,2])
		self.x3 = nn.Conv2d(hidden, hidden, kernel_size=[1,7], padding=[0,3])
		self.y1 = nn.Conv2d(hidden, hidden, kernel_size=[3,1], padding=[1,0])
		self.y2 = nn.Conv2d(hidden, hidden, kernel_size=[5,1], padding=[2,0])
		self.y3 = nn.Conv2d(hidden, hidden, kernel_size=[7,1], padding=[3,0])
		self.conv2 = nn.Conv2d(hidden, input, kernel_size=3, padding=1)
		self.weights =  nn.Conv2d(hidden, 6, kernel_size=1)
		self.relu = nn.ReLU(inplace=True)
		self.bn = nn.InstanceNorm2d(hidden)
		self.softmax = torch.softmax
		self.bn2 = nn.InstanceNorm2d(input)
	def forward(self, x):
		residual = x
		x = self.conv1(x)
		w = self.weights(x)
		wx = self.softmax(w[:,:3],1)
		wy = self.softmax(w[:,3:],1)
		x1 = self.x1(x)
		x2 = self.x2(x)
		x3 = self.x3(x)
		x = x1.transpose(0,1).mul(wx[:,0])+x2.transpose(0,1).mul(wx[:,1])+x3.transpose(0,1).mul(wx[:,2])
		x = x.transpose(0,1)
		x = self.bn(x)
		x = self.relu(x)
		y1 = self.y1(x)
		y2 = self.y2(x)
		y3 = self.y3(x)
		x = y1.transpose(0,1).mul(wy[:,0])+y2.transpose(0,1).mul(wy[:,1])+y3.transpose(0,1).mul(wy[:,2])
		x = x.transpose(0,1)
		x = self.bn(x)
		x = self.relu(x)
		x = self.conv2(x)
		x +=residual
		x = self.bn2(x)
		x = self.relu(x)
		return x
