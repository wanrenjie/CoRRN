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

torch.cuda.set_device(0)
path = '/home/rjwan/PostcardDataset'
vgg = torchvision.models.vgg16_bn(pretrained = True)
vgglist = list(vgg.features.children())
model = ReflectionNetwork(vgglist)

checkpoint = torch.load("/mnt/UserData/rjwan/CoRRN/checkpoint_80.pth.tar")
model.load_state_dict(checkpoint['state_dict'])

model.eval()
aa = list(model.children())
print(aa[10])
'''
for file in files:
	files2 = os.listdir(path+"/"+file)
	for file2 in files2:
		files3 = os.listdir(path+"/"+file+"/"+file2)
		for file3 in files3:
			files4 = os.listdir(path+"/"+file+"/"+file2+"/"+file3)
			img = Image.open(path+"/"+file+"/"+file2+"/"+file3+"/m.png")
			img = img.resize((288,224), Image.ANTIALIAS)
			img = np.asarray(img, dtype = 'float32')
			img = img.transpose(2,0,1)
			inputB = torch.from_numpy(img)
			inputB = inputB.unsqueeze(0)
			inputB = inputB/255
			inputB = Variable(inputB)
			model = model.cuda()
			inputB = inputB.cuda()
			output = model(inputB)
			outp = MatrixToImage(output[0].data.cpu().numpy().reshape(3, 224, 288).transpose(1,2,0))
			outp.save(path+"/"+file+"/"+file2+"/"+file3+"/"+"our.png")
'''