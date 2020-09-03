from PIL import Image
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms import Compose as Compose_source
import torch.nn.functional as FF
import numpy as np
class Compose(Compose_source):
	def __init__(self, transforms):
		super(Compose, self).__init__(transforms=transforms)


	def __call__(self, img, **kwargs):
		for t in self.transforms:
			img = t(img, **kwargs)
		return img

class norm(object):
	def __init__(self):
		self.mean = [0.485, 0.456, 0.406]
		self.std = [0.229, 0.224, 0.225]
		self.norm = Normalize(self.mean, self.std)

	def __call__(self, sample, **kwargs):
		for idx, tmp in enumerate(sample):
			if tmp.size()[0] == 1:
				sample[idx] = tmp
			else:
				sample[idx] = self.norm(tmp)
		return sample

class resize(object):
	"""docstring for resize"""
	def __init__(self):
		super(resize, self).__init__()
	def __call__(self, sample, **kwargs):	
		for idx, tmp in enumerate(sample):
			if tmp.size(0)!=1:
				sample[idx] = FF.interpolate(tmp.unsqueeze(0),size = [480,854],mode='bilinear',align_corners=True).squeeze(0)
		return sample		
class totensor(object):
	"""docstring for totensor"""
	def __init__(self):
		super(totensor, self).__init__()
		self.totensor = ToTensor()
	def __call__(self, sample, **kwargs):
		for idx, tmp in enumerate(sample):
			sample[idx] =self.totensor(tmp)
		return sample
class pad(object):
	"""docstring for pad"""
	def __init__(self):
		super(pad, self).__init__()
	def __call__(self, sample, **kwargs):
		for idx, tmp in enumerate(sample):
			if tmp.size(0)==1:
				a,b = tmp.size()[-2:]
				sample[idx] = FF.pad(tmp, (0, 2000 - b, 0, 2000 - a, ),value=2)
		return sample
def transform():
		return Compose([
			totensor(),
			resize(),
			pad(),
			norm()
			])

