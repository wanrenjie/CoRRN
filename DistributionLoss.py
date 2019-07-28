import torch
import torch.nn.functional as F
import numpy as np
from utils import rgb2gray
import torch.nn as nn
from torch.autograd import Variable

class DistributionLoss(torch.nn.Module):
	def __init__(self):
		super(DistributionLoss, self).__init__()
		
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
			
		self.criterion = nn.KLDivLoss()
		
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
		im1xd = torch.log(im1xd)
		
		im1yd = F.softmax(im1gy.view(-1, height*width), dim = 1)
		im2yd = F.softmax(im2gy.view(-1, height*width), dim = 1)
		im1yd = torch.log(im1yd)
		
		self.loss = self.criterion(im1xd+0.001, im2xd+0.001)+ self.criterion(im1yd+0.001, im2yd+0.001)
		#print(self.loss)
		return self.loss