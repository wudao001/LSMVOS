import torch
import torch.utils.data as data
import numpy as np
import os
from glob import glob
import random
from transform import transform
from PIL import Image
class Davis(data.Dataset):
	"""docstring for Davis"""
	def __init__(self,root,year,mode,batch_size = None):
		super(Davis, self).__init__()
		self.root = root
		index = {'train' : '/train.txt','trainval' : '/trainval.txt','val' : '/val.txt', 'test-challenge' : '/test-challenge.txt','test-dev' : '/test-dev.txt'}
		self.year = year
		self.mode = mode
		self.batch_size = batch_size
		with open(self.root+'/ImageSets/'+self.year+index[mode]) as f:
			train_file = f.readlines()
		dirs = [self.root + '/JPEGImages/480p/' + name[0:-1] for name in train_file]
		dirs.sort()
		image_names = []
		for dir in dirs:
			files = glob(dir + '/*.*')
			files.sort()
			image_names.append(files)
		self.image_paths = [[],[]]
		self.image_paths2 = image_names
		self.idx = np.zeros(batch_size,dtype=np.int32)
		for i in range(batch_size):
			self.idx[i] = i
		self.idx2 = np.zeros(batch_size,dtype=np.int32)
		self.idx3 = 0
		self.idx4 = np.zeros(batch_size,dtype=np.int32)
		self.idx5 = batch_size-1

		for i in range(self.__len__()):
			self.sort(i)
		
		self.transform = transform()
	def __getitem__(self,index):
		image = self.image_paths[0][index]
		iskey = self.image_paths[1][index]
		if iskey==2:
			return torch.zeros(1,3,480,854),torch.zeros(1,1,2000,2000),2,'None',[0,0]
		img = Image.open(image).convert('RGB')
		if iskey==0:
			lab = Image.open(image.replace('JPEGImages', 'Annotations').replace('jpg','png')).convert('P')
		else:
			lab = Image.fromarray(np.zeros((480,854),dtype=np.uint8)).convert('P')
		size = img.size
		imgs = self.transform([img,lab])
		img = imgs[0].unsqueeze(0)
		lab = imgs[1].unsqueeze(0)
		if self.year == '2016':
			lab = (lab!=0).to(torch.float32)+(lab==2).to(torch.float32)
		return img,lab,iskey,image.replace('\\','/'),size
	def sort(self,index):
		if self.idx[self.idx3]>len(self.image_paths2)-1:
			if self.idx3<self.batch_size-1:
				self.idx3 +=1
			else:
				self.idx3 = 0
			self.image_paths[0].append(None)
			self.image_paths[1].append(2)
			return 
		image = self.image_paths2[self.idx[self.idx3]][self.idx2[self.idx3]]
		iskey = 1
		if self.idx2[self.idx3] ==0:
			iskey = 0
		if self.idx2[self.idx3]<len(self.image_paths2[self.idx[self.idx3]])-1:
			self.idx2[self.idx3] +=1
		else:
			self.idx2[self.idx3] = 0
			self.idx[self.idx3] = self.idx5+1
			self.idx5 +=1
		if self.idx3<self.batch_size-1:
			self.idx3 +=1
		else:
			self.idx3 = 0
		self.image_paths[0].append(image)
		self.image_paths[1].append(iskey)
		return 
	
	def __len__(self):
		count = 0
		idx = torch.zeros(self.batch_size).to(torch.int32)
		for i in self.image_paths2:
			idx[idx.min(0)[1]]+=len(i)
		count = idx.max().item()*self.batch_size
		return count
