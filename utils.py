import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from scipy.ndimage import imread
import numpy as np
import time, math
from PIL import Image

def PSNR(pred, gt, shave_border = 0):
	height, width = pred.shape[:2]
	pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
	gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
	imdiff = pred - gt
	rmse = math.sqrt(np.mean(imdiff ** 2))
	if rmse == 0:
		return 100
	return 20 * math.log10(255.0 / rmse)
	
def MatrixToImage(data):
	data = data*255
	new_im = Image.fromarray(data.astype(np.uint8))
	return new_im
	
def MatrixToImage2(data):
	data = data*5
	new_im = Image.fromarray(data.astype(np.uint8))
	return new_im	
	
def rgb2gray(img):
	(batch, channel, height, width) = img.size()
	gImg = Variable(torch.ones(batch, 1, height, width)).cuda()
	
	for i in range(batch):
		grayimg = 0.2989 * img[i,0,:,:] + 0.5870 * img[i,1,:,:] + 0.1140 * img[i,2,:,:]
		gImg[i,0,:,:] = grayimg
	
	return gImg
	
def MMDcompute(x, y, alpha = 1):
	
	#x = x.view(x.size(0), x.size(2) * x.size(3))
	#y = y.view(y.size(0), y.size(2) * y.size(3))
		
	xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

	rx = (xx.diag().unsqueeze(0).expand_as(xx))
	ry = (yy.diag().unsqueeze(0).expand_as(yy))
	
	rx1 = (xx.diag().unsqueeze(0).expand_as(torch.Tensor(y.size(0),x.size(0))))
	ry1 = (yy.diag().unsqueeze(0).expand_as(torch.Tensor(x.size(0),y.size(0))))
	
		# K = torch.exp(- alpha * (rx.t() + rx - 2*xx))
		# L = torch.exp(- alpha * (ry.t() + ry - 2*yy))
		# P = torch.exp(- alpha * (rx1.t() + ry1 - 2*zz))

	K = (torch.exp(- 0.5*alpha * (rx.t() + rx - 2*xx)) + torch.exp(- 0.1*alpha * (rx.t() + rx - 2*xx)) \
		+ torch.exp(- 0.05*alpha * (rx.t() + rx - 2*xx)))/3
	L = (torch.exp(- 0.5*alpha * (ry.t() + ry - 2*yy)) + torch.exp(- 0.1*alpha * (ry.t() + ry - 2*yy)) \
		+ torch.exp(- 0.05*alpha * (ry.t() + ry - 2*yy)))/3
	P = (torch.exp(- 0.5*alpha * (rx1.t() + ry1 - 2*zz)) + torch.exp(- 0.1*alpha * (rx1.t() + ry1 - 2*zz)) \
		+ torch.exp(- 0.05*alpha * (rx1.t() + ry1 - 2*zz)))/3

	beta1 = (1./(x.size(0)*x.size(0)))
	beta2 = (1./(y.size(0)*y.size(0)))
	gamma = (2./(x.size(0)*y.size(0))) 

	return beta1 * torch.sum(K) + beta2 * torch.sum(L) - gamma * torch.sum(P)	