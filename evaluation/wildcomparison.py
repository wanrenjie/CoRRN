import torch
import torchvision.utils as vutils
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import PSNR, MatrixToImage
from torch.autograd import Variable
from PIL import Image
import torch.nn as nn
import scipy.io as sio
import os
import torchvision
from network10 import ReflectionNetwork

path = '/home/rjwan/HELPOTHER/test1/mixture3'
destination = '/home/rjwan/HELPOTHER/test2/mixture4'
torch.cuda.set_device(1)
#files = os.listdir(path)
vgg = torchvision.models.vgg16_bn(pretrained = True)
vgglist = list(vgg.features.children())
#print("==>Loading model")
model = ReflectionNetwork(vgglist)

checkpoint = torch.load("/home/rjwan/HELPOTHER/checkpoint_80.pth.tar")
#checkpoint = torch.load("/home/Test25/model2/checkpoint_81.pth.tar")
model.load_state_dict(checkpoint['state_dict'])

model.eval()

filelist1 = [file for file in os.listdir(path) if file.endswith('.jpg')]
os.makedirs(destination)

for file in filelist1:
	img = Image.open(path + "/" + file)
	img = np.asarray(img, dtype ='float32')
	img = img.transpose(2, 0, 1)
	inputB = torch.from_numpy(img).float()
	[c,h,w] = inputB.size()
	inputB = inputB.unsqueeze(0)
	inputB = inputB/255
	inputB = Variable(inputB.cuda())
	model = model.cuda()
	output = model(inputB)
	outp = MatrixToImage(output[0].data.cpu().numpy().reshape(c,h,w).transpose(1,2,0))
	outp.save(destination+"/"+file)
'''
for file in files:
	files2 = os.listdir(path+"/"+file)
	
	img = Image.open(path+"/"+file+"/m.jpg")
	#img = img.resize((288,224), Image.ANTIALIAS)
	img = np.asarray(img, dtype = 'float32')
	img = img.transpose(2,0,1)
	inputB = torch.from_numpy(img).float()
	inputB = inputB.unsqueeze(0)
	inputB = inputB/255
	inputB = Variable(inputB)
	model = model.cuda()
	inputB = inputB.cuda()
	output = model(inputB)
	#outp = MatrixToImage(output[0].data.cpu().numpy().reshape(3, 224, 288).transpose(1,2,0))
	outp = MatrixToImage(output[0].data.cpu().numpy().reshape(3, 288, 224).transpose(1,2,0))
	outp.save(path+"/"+file+"/"+"our6.jpg")	
'''	
