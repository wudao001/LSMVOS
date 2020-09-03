
import torch.nn as nn
import math
import torch
import torch.nn.functional as FF
from res2net import res2net50_26w_8s
torch.backends.cudnn.benchmark = True
import numpy as np
from AIC import AIC

def makerlayer(inplanes,planes,ksize=3,d=1,groups=1):
	p = int((ksize - 1) / 2)
	return nn.Sequential(
        nn.Conv2d(inplanes, planes, ksize, padding=p, dilation=d, groups=groups),
        nn.InstanceNorm2d(planes),
        nn.ReLU(),

    )
class Global(nn.Module):
	"""docstring for Global"""
	def __init__(self):
		super(Global, self).__init__()
		self.softmax = torch.softmax
	def global_corr(self,img,key_feature):
		ker = key_feature
		size = ker.size()
		co = []
		for i in range(size[0]):
			temp_ker = ker[i].view(size[1], size[2] * size[3]).transpose(0, 1)#C,H,W->H*W,C
			temp_ker = temp_ker.unsqueeze(2).unsqueeze(3)
			co.append(FF.conv2d(img[i].unsqueeze(0), temp_ker.contiguous())) #H*W,H,W
		return torch.cat(co,0)
	def forward(self, key,curr,mask,mode = 0):
		if mode==0 or mode ==1:
			B,C,H,W = curr.size()
			key = FF.normalize(key)
			curr = FF.normalize(curr)
			co = self.global_corr(curr,key)#B,H*W,H,W
		if mode ==1:
			return co
		if mode == 2:
			co = key
			B,C,H,W = co.size()
		mask = FF.adaptive_max_pool2d(mask,[H,W])
		mask = mask.contiguous().view(B,H*W,1,1)
		co_f = co.mul(mask).contiguous().view(B,H*W,H,W)
		co_f = co_f.topk(256,1)[0]#.sum(1).div(co_sum).unsqueeze(1)
		co_b = co.mul(1-mask).contiguous().view(B,H*W,H,W)
		co_b = co_b.topk(256,1)[0]#.sum(1).div(co_sum).unsqueeze(1)
		return co_f,co_b
class Local(nn.Module):
	"""docstring for Local"""
	def __init__(self,Using_Correlation,scale = 10):
		super(Local, self).__init__()
		self.upsample = FF.interpolate
		self.Using_Correlation = Using_Correlation
		if Using_Correlation=='True':
			from correlation_package.correlation import Correlation
			self.corr = Correlation(pad_size=8, kernel_size=1, max_displacement=8, stride1=1, stride2=1, corr_multiply=1)
		self.scale = scale
	def local_corr(self,imgs,kers,scale = 10,k=8):
		size = imgs.size()
		kers = FF.pad(kers,(k,k,k,k),'reflect')
		res = []
		for i in range(int((size[-1]+scale-1)/scale)):
			f_co_row = []
			temp_img = imgs[:,:,:,i*scale:(i*scale+min(scale,size[-1]-i*scale))]
			temp_size = temp_img.size()
			temp_img = self.upsample(temp_img,[size[-2]*(2*k+1),min(scale,size[-1]-i*scale)*(2*k+1)])
			temp_ker = kers[:,:,:,i*scale:(i*scale+min(scale,size[-1]-i*scale)+2*k)]
			temp_ker = temp_ker.unfold(3,2*k+1,1).contiguous().view(size[0],size[1],temp_ker.size()[2],-1)
			temp_ker = temp_ker.unfold(2,2*k+1,1).transpose(2,3).contiguous().view(size[0],size[1],temp_ker.size()[3],-1).transpose(2,3)

			co = torch.mul(temp_ker,temp_img)
			co = co.sum(1).unsqueeze(1)
			temp_co = co.split(2*k+1,-1)

			for cow in range(len(temp_co)):
				f_co =temp_co[cow].contiguous().view(size[0],-1,(2*k+1)*(2*k+1)).transpose(1,2)
				f_co_row.append(f_co)
			f_co_row = torch.cat(f_co_row,-1)
			f_co_row = f_co_row.contiguous().view(size[0],-1,temp_size[-1],temp_size[-2]).transpose(2,3)
			res.append(f_co_row)

		res = torch.cat(res,-1)
		return res
	def forward(self, pre,curr,mask,mode = 0):
		if mode==0 or mode ==1:

			pre = FF.normalize(pre)
			curr = FF.normalize(curr)
			B,C,H,W = pre.size()
			if self.Using_Correlation=='True':
				co = self.corr(curr,pre)*256
			else:
				co = self.local_corr(imgs = curr, kers = pre,scale =self.scale ,k=8)#B,31*31,H,W
			
		
		if mode ==1:
			return co
		if mode ==2:
			co = pre
			B,C,H,W = pre.size()
		k = 8
		mask = FF.adaptive_max_pool2d(mask,[H,W])
		mask = FF.pad(mask,(k,k,k,k))
		mask = mask.unfold(3,2*k+1,1).contiguous().view(B,1,mask.size()[2],-1)
		mask = mask.unfold(2,2*k+1,1).transpose(2,3).contiguous().view(B,1,mask.size()[3],-1).transpose(2,3)
		mask = mask.view(B,H,(2*k+1),W,(2*k+1)).permute(0,2,4,1,3).contiguous().view(B,(2*k+1)*(2*k+1),H,W)
		f = co.mul(mask)
		f = f.topk(256,1)[0]#.sum(1).div(co_sum).unsqueeze(1)
		b = co.mul(1-mask)
		b = b.topk(256,1)[0]#.sum(1).div(co_sum).unsqueeze(1)
		return f,b
class Encoder(nn.Module):
	"""docstring for Encoder"""
	def __init__(self):
		super(Encoder, self).__init__()
		model = res2net50_26w_8s(pretrained=False)
		#3->64
		self.conv1 = model.conv1#25m
		self.bn1 = model.bn1
		self.relu = model.relu
		self.maxpool = model.maxpool
		#64->256
		self.layer1 = model.layer1#25m*3=75m*3=225m
		#256->512
		self.layer2 = model.layer2#12m*4=48m*3=144m
		#512->1024
		self.layer3 = model.layer3#6m*6=36m*3=108m
		#1024->2048
		self.layer4 = model.layer4#3m*3=9m*3=27m
		for p in self.parameters():
			p.requires_grad=False
		self.smooth4 = AIC(256,64)#1m+76m=77m
		self.smooth3 = AIC(256,64)#6m+66m=72m
		self.smooth2 = AIC(256,64)#25m+244m=259m
		self.smooth1 = AIC(64,64)#25m+244m=259m

		self.l55 = makerlayer(2048,256,ksize = 1)
		self.l44 = makerlayer(1024,256,ksize = 1)#1m
		self.l33 = makerlayer(512,256,ksize = 1)#6m
		self.l22 = makerlayer(256,256,ksize = 1)#25m
		self.l21 = makerlayer(256,64,ksize = 1)
		self.l11 = makerlayer(64,64,ksize = 1)#25m
		self.upsample = FF.interpolate
	def add_upsumple(self,x,y):
		x = self.upsample(x,size=y.size()[-2:],mode='bilinear',align_corners=True)
		return x+y
	def forward(self, img):
		x = self.conv1(img)
		x = self.bn1(x)
		x1 = self.relu(x)
		x = self.maxpool(x1)

		x2 = self.layer1(x)
		x3 = self.layer2(x2)
		x4 = self.layer3(x3)
		x5 = self.layer4(x4)


		p4 = self.add_upsumple(self.l55(x5),self.l44(x4))
		p4 = self.smooth4(p4)
		p3 = self.add_upsumple(p4,self.l33(x3))
		p3 = self.smooth3(p3)
		p2 = self.add_upsumple(p3,self.l22(x2))
		p2 = self.smooth2(p2)
		p1 = self.add_upsumple(self.l21(p2),self.l11(x1))
		p1 = self.smooth1(p1)

		
		
		
		
		return p3,p2,p1
class KV(nn.Module):
	"""docstring for KV"""
	def __init__(self,inputs,out_key):
		super(KV, self).__init__()
		self.Key = AIC(inputs,out_key)
	def forward(self, x):	
		return self.Key(x)
class Refine(nn.Module):
	"""docstring for Refine"""
	def __init__(self,inputs,hidden):
		super(Refine, self).__init__()
		self.aic1 = AIC(inputs,hidden)#makerlayer(inputs,inputs)#
		self.aic2 = AIC(inputs,hidden) #makerlayer(inputs,inputs)#
	def forward(self, x,y):
		x = FF.interpolate(x, size = y.size()[-2:], mode='bilinear', align_corners=True)
		x = self.aic1(x)
		x = x+y
		x = self.aic2(x)
		return x
class Decoder(nn.Module):
	"""docstring for Decoder"""
	def __init__(self):
		super(Decoder, self).__init__()
		self.ref1 = Refine(256,64)
		self.ref2 = Refine(64,64)
		self.conv1 = nn.Conv2d(256, 64, kernel_size=(3,3), padding=(1,1), stride=1)
		self.pred = nn.Conv2d(64, 1, kernel_size=(3,3), padding=(1,1), stride=1)
		self.relu = nn.ReLU(inplace=False)
		self.upsample = FF.interpolate
		self.drop2 = nn.Dropout2d(p = 0.5)
	def forward(self, p3,p2,p1):	
		
		p2 = self.ref1(p3,p2)
		p2 = self.conv1(p2)
		p1 = self.ref2(p2,p1)
		p = self.pred(p1)
		p = torch.sigmoid(p)
		return p

class LSMVOS(nn.Module):
	"""docstring for LSMVOS"""
	def __init__(self,Using_Correlation,scale = 10):
		super(LSMVOS, self).__init__()
		self.encoder = Encoder()
		self.glob = Global()
		self.loc = Local(Using_Correlation = Using_Correlation,scale = scale)
		self.g_kv = KV(256,64)
		self.l_kv = KV(256,64)
		self.decoder = Decoder()
		self.conv1 = nn.Conv2d(256*5+1, 256, kernel_size=(3,3), padding=(1,1), stride=1)
		self.aic1 = AIC(256,64)
		
		self.upsample = FF.interpolate
		self.drop2 = nn.Dropout2d(p = 0.5)
	def forward(self, img,kg1=None,klp=None,key_mask=None,pre_mask=None,first=2):
		
		
		if first==0:
			p3,p2,p1 = self.encoder(img)
			kg = self.g_kv(p3)
			kl = self.l_kv(p3)
			return kg.detach(),kl.detach()
		elif first ==1:
			p3,p2,p1 = self.encoder(img)
			kg = self.g_kv(p3)
			kl = self.l_kv(p3)
			lc = self.loc(pre = klp,curr = kl,mask =None,mode = 1)
			gc = self.glob(key = kg1,curr = kg,mask = None,mode=1)
			
			return gc,lc,p3,p2,p1,kl.detach()
		else:
			gf,gb = self.glob(key = kg1,curr = None,mask = key_mask,mode=2)
			lf,lb = self.loc(pre = klp,curr = None,mask =pre_mask,mode = 2)
			p3,p2,p1 = img
			pre_mask = FF.adaptive_max_pool2d(pre_mask,p3.size()[-2:])
			p3 = torch.cat([p3,gf,gb,lf,lb,pre_mask],1)
			p3 = self.drop2(p3)
			p3 = self.conv1(p3)
			p3 = self.aic1(p3)
			p = self.decoder(p3,p2,p1)
			return p

