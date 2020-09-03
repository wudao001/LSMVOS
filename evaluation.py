
import torch
import torch.nn.functional as FF
import torch.utils.data as data
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from DAVIS import *
import torch.optim as optim
from LSMVOS import LSMVOS
import numpy as np
import argparse
torch.backends.cudnn.benchmark = True
save_img = True
parser = argparse.ArgumentParser(description='LSMVOS')
parser.add_argument('--deviceID', nargs='+', type=int, default=[0], help='设备ID')
parser.add_argument('--Using_Correlation', default=False, help='是否安装Correlation')
parser.add_argument('--scale', type=int, default=10, help='图片分组像素')
parser.add_argument('--batch_size', type=int, default=1, help='批量大小')
parser.add_argument('--year',  default='2016', help='Davis年份')
parser.add_argument('--mode',  default='val', help='数据集名称')
parser.add_argument('--path', help='模型地址')
parser.add_argument('--root', help='数据集地址')
def mask(img):
	img = (img!=0).to(torch.float32)
	return img
def bbox_crop(img,bbox):
	return img[:,:,bbox[0]:bbox[2],bbox[1]:bbox[3]]
def bbox_uncrop(img, bbox): 
	msk = FF.pad(img, (bbox[1], 854 - bbox[3], bbox[0], 480 - bbox[2], ))
	return msk
def resize(img):
	 img = FF.interpolate(img.to(torch.float32),size = [480,854],mode='bilinear',align_corners=True)
	 return img
class Eval(object):
	"""docstring for Eval"""
	def __init__(self,model,batch_size,year,mode):
		super(Eval, self).__init__()
		self.model = model
		self.upsample = FF.interpolate
		self.batch_size = batch_size
		self.year = year
		self.mode = mode
		self.kg1 = [[]]*batch_size
		self.klp = [[]]*batch_size
		self.pre_mask = [[]]*batch_size
		self.key_mask = [[]]*batch_size
		self.folder = [[]]*batch_size
		self.ids = [[]]*batch_size
		self.pred = [[]]*batch_size
	def oneshot(self,lab):
		x = (lab[:,:,:,:1]==2).nonzero()[:,-2].min()
		y = (lab[:,:,:1,:]==2).nonzero()[:,-1].min()
		lab = lab[:,:,:x,:y]
		ids = [x.tolist() for x in list(set(lab.cpu().numpy().reshape(-1)))]
		ids.sort()
		labs = []
		for idx in ids:
			labs.append(mask(resize(lab==idx)))
		return labs,ids
	def getlab(self,lab,obj):
		x = (lab[:,:,:,:1]==2).nonzero()[:,-2].min()
		y = (lab[:,:,:1,:]==2).nonzero()[:,-1].min()
		lab = lab[:,:,:x,:y]
		lab = mask(lab==obj)
		return lab
	def rmpad(self,lab):
		x = (lab[:,:,:,:1]==2).nonzero()[:,-2].min()
		y = (lab[:,:,:1,:]==2).nonzero()[:,-1].min()
		lab = lab[:,:,:x,:y]
		return lab
	def __call__(self,data,palette):
		key = [[],[],[]]
		curr = [[],[],[]]
		imgs = data[0].cuda()
		labs = data[1].cuda()
		for i in range(self.batch_size):
			if data[2][i]==0:
				key_img = resize(imgs[i])
				key_lab = labs[i]
				key_object_labs,ids = self.oneshot(key_lab)
				self.ids[i] = ids
				key[0].append(key_img)
				key[1].append(key_object_labs[1:])
				key[2].append(i)
				if save_img:
					folder = './DAVIS_' + self.year+'_' + self.mode+'/'+data[3][i].split('/')[-2]
					if not os.path.exists(folder):
						os.makedirs(folder)
					size = [data[4][1][i],data[4][0][i]]
					slab = torch.cat(key_object_labs,1)[0]
					slab = FF.adaptive_max_pool2d(slab.unsqueeze(0),[480,854]).squeeze(0)
					slab = slab.cpu().numpy()
					slab = np.argmax(slab,axis=0).astype(np.uint8)
					slab = Image.fromarray(slab)
					slab.putpalette(palette)
					slab.save(folder+'/'+data[3][i].split('/')[-1].replace('jpg','png'),palette=palette)
					self.folder[i] = folder
			elif data[2][i]==1:
				img = resize(imgs[i])
				lab = labs[i]
				
				object_labs = []
				for idx in self.ids[i]:
					labi = self.getlab(lab,idx)
					object_labs.append(labi)
				curr[0].append(img)
				curr[1].append(object_labs)
				curr[2].append(i)
		if key[0]:
			key_imgs = torch.cat(key[0],0)
			key_labs =	key[1]
			with torch.no_grad():
				kg1,klp=self.model(img=key_imgs,first=0)
			key_mask = key_labs
			for i in range(len(key[2])):
				self.kg1[key[2][i]] = kg1[i].unsqueeze(0)
				self.klp[key[2][i]] = klp[i].unsqueeze(0)
				self.pre_mask[key[2][i]] = torch.cat(key_mask[i],1).squeeze(0)
				self.key_mask[key[2][i]] = key_mask[i]
		if curr[0]:
			imgs = torch.cat(curr[0],0)
			kg1 = []
			key_mask = []
			klp = []
			pre_mask = []
			count = 0
			torch.cuda.empty_cache()
			for video_num in range(imgs.size()[0]):
				kg1.append(self.kg1[curr[2][video_num]])
				klp.append(self.klp[curr[2][video_num]])
			kg1 = torch.cat(kg1,0)
			klp = torch.cat(klp,0)
			gc,lc,p3,p2,p1,kl = self.model(img=imgs,kg1=kg1,key_mask=None,
			 				klp = klp,pre_mask=None,first=1)
			for video_num in range(imgs.size()[0]):
				self.klp[curr[2][video_num]] = kl[video_num].unsqueeze(0)
			kg1 = []
			key_mask = []
			klp = []
			pre_mask = []
			imgs = [[],[],[]]
			pred = []
			for video_num in range(len(curr[2])):
				for obj_num in range(len(self.key_mask[curr[2][video_num]])):
					key_mask.append(FF.adaptive_max_pool2d(self.key_mask[curr[2][video_num]][obj_num],[480,854]))
					pre_mask.append(FF.adaptive_max_pool2d(self.pre_mask[curr[2][video_num]][obj_num].unsqueeze(0).unsqueeze(0),[480,854]))
					kg1.append(gc[video_num].unsqueeze(0)) 
					klp.append(lc[video_num].unsqueeze(0))
					imgs[0].append(p3[video_num].unsqueeze(0))
					imgs[1].append(p2[video_num].unsqueeze(0))
					imgs[2].append(p1[video_num].unsqueeze(0))
					count +=1
					if count ==self.batch_size:
						kg1 = torch.cat(kg1,0)
						key_mask = torch.cat(key_mask,0)
						klp = torch.cat(klp,0)
						pre_mask = torch.cat(pre_mask,0)
						imgs = [torch.cat(imgs[0],0),torch.cat(imgs[1],0),torch.cat(imgs[2],0)]
						p = self.model(img=imgs,kg1=kg1,key_mask=key_mask,
			 				klp = klp,pre_mask=pre_mask,first=2)
						count = 0
						pred.append(p)
						kg1 = []
						key_mask = []
						klp = []
						pre_mask = []
						imgs = [[],[],[]]
			if count!=0:
				kg1 = torch.cat(kg1,0)
				key_mask = torch.cat(key_mask,0)
				klp = torch.cat(klp,0)
				pre_mask = torch.cat(pre_mask,0)
				imgs = [torch.cat(imgs[0],0),torch.cat(imgs[1],0),torch.cat(imgs[2],0)]
				
				p = self.model(img=imgs,kg1=kg1,key_mask=key_mask,
			 		klp = klp,pre_mask=pre_mask,first=2)
				pred.append(p)
			pred = torch.cat(pred,0)
			count = 0
			iou = []
			for video_num in range(len(curr[2])):
				predi = []
				for obj_num in range(len(self.key_mask[curr[2][video_num]])):
					predi.append(pred[count])
					count +=1
				size = [data[4][1][curr[2][video_num]],data[4][0][curr[2][video_num]]]
				predi = torch.cat(predi,0)
				
				predi = FF.interpolate(predi.unsqueeze(0),size = size,mode='bilinear',align_corners=True).squeeze(0)
				self.pre_mask[curr[2][video_num]] = predi
				predi = torch.cat([mask((predi>0.6).sum(0)==0).unsqueeze(0),predi],0)
				
				predi = predi.max(0)[1].to(torch.int32)
				slab = transforms.ToPILImage()(predi.cpu()).convert('P')
				slab.putpalette(palette)
				slab.save(self.folder[curr[2][video_num]]+'/'+data[3][curr[2][video_num]].split('/')[-1].replace('jpg','png'))
				
				pre_objs = []
				for i in range(len(self.ids[curr[2][video_num]])):
					if  i==0:
						continue
					pre_obj_i = (predi==i).to(torch.float32)
					pre_objs.append(pre_obj_i)
				
				self.pre_mask[curr[2][video_num]] =torch.stack(pre_objs,0)
			return 
		return None
	def parameters(self):
		return self.model.parameters()
	def state_dict(self):
		return self.model.state_dict()

		
def main(model,root,batch_size,year,mode):
	batch_size =batch_size
	dataset=Davis(root=root,year=year,mode=mode,batch_size=batch_size)
	model.eval()
	model = Eval(model,batch_size,year,mode)
	dataset_size=len(dataset)
	data_loader = DataLoader(dataset=dataset, num_workers=8, batch_size=batch_size, shuffle=False, pin_memory=True)
	num = 0
	palette_path = 'palette.txt'
	with open(palette_path) as f:
		palette = f.readlines()
	palette = list(np.asarray([[int(p) for p in pal[0:-1].split(' ')] for pal in palette]).reshape(768))
	for idx,data in enumerate(data_loader):
		num += 1
		with torch.no_grad():
			model(data,palette)
			print("%d/%d" % (num,((dataset_size+batch_size-1)//batch_size)))
	return 
	


if __name__ == '__main__':
	arg = parser.parse_args()
	Using_Correlation = arg.Using_Correlation
	scale = arg.scale
	path = arg.path
	deviceID = arg.deviceID
	batch_size = arg.batch_size
	year = arg.year
	mode = arg.mode
	root = arg.root
	model=LSMVOS(Using_Correlation,scale).cuda()

	model_dict = model.state_dict()
	pretrained_dict = torch.load(path)
	pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
	model_dict.update(pretrained_dict)
	model.load_state_dict(model_dict)
	
	if deviceID !=[0]:
		model = torch.nn.DataParallel(model, device_ids=deviceID)
	#checkpoint = torch.load(path)
	#model.load_state_dict(checkpoint)
	#model.cuda()
	main(model=model,root=root,batch_size=batch_size,year=year,mode=mode)
