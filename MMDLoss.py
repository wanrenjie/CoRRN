import torch
import torch.nn.functional as F
import numpy as np
from utils import rgb2gray,MMDcompute
import torch.nn as nn
from torch.autograd import Variable

class MMDLoss(torch.nn.Module):

	def __init__(self):
		super(MMDLoss, self).__init__()
		
		self.xconv = nn.Conv2d(1, 1, kernel_size = 3, stride = 1, padding = 1).cuda()
		self.xconv.bias.data.zero_()
		self.xconv.weight.data[0,0,:,:] = torch.FloatTensor([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]).cuda()
		for param in self.xconv.parameters():
			param.requires_grad = False
			
		self.yconv = nn.Conv2d(1, 1, kernel_size = 3, stride = 1, padding = 1).cuda()
		self.yconv.bias.data.zero_()
		self.yconv.weight.data[0,0,:,:] = torch.FloatTensor([[-1, -2, -1],[0, 0, 0],[1, 2, 1]]).cuda()
		for param in self.yconv.parameters():
			param.requires_grad = False
	
	def forward(self, im1, im2):
		im1g = rgb2gray(im1)
		im2g = rgb2gray(im2)
		
		im1gx = self.xconv(im1g)
		im1gy = self.yconv(im1g)
		
		im2gx = self.xconv(im2g)
		im2gy = self.yconv(im2g)
		
		(batch, channel, height, width) = im1.size()
		
		im1xd = F.softmax(im1gx.view(-1, height*width), dim = 1)
		im2xd = F.softmax(im2gx.view(-1, height*width), dim = 1)
		
		im1yd = F.softmax(im1gy.view(-1, height*width), dim = 1)
		im2yd = F.softmax(im2gy.view(-1, height*width), dim = 1)
		
		self.loss = MMDcompute(im1xd, im2xd) + MMDcompute(im1yd, im2yd)
		
		return self.loss
'''		
	def MMDcompute(x2222, y2222, alpha = 1):
	
		print(x2222)
		x = x.view(x.size(0), x.size(2) * x.size(3))
		y = y.view(y.size(0), y.size(2) * y.size(3))
		
		print(x.size())
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
'''		

