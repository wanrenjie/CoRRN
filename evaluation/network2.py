import torch
import torch.nn as nn
import os
import sys
import torchvision
import torchvision.models as models
from torch.autograd import Variable
#VGG+DECONVOLUTION+edgenetwork+FEATURCONSTRUCTION

class BasicConv2d(nn.Module):
	
	def __init__(self, in_planes, out_planes, kernel_size, stride, padding = 0):
		
		super(BasicConv2d, self).__init__()
		self.conv = nn.Conv2d(in_planes, out_planes,
								kernel_size = kernel_size, stride = stride,
								padding = padding, bias = False
							 ) # verify bias false
		self.bn = nn.BatchNorm2d(out_planes,
								eps=0.001, # value found in tensorflow
								momentum=0.1, # default pytorch value
								affine=True)
		self.relu = nn.ReLU(inplace = True)

	def forward(self, x):
		
		x = self.relu(self.bn(self.conv(x)))
		return x

class Conv2dUnit(nn.Module):
	def __init__(self, in_planes, out_planes, kernel_size, stride, padding = 0):
		super(Conv2dUnit, self).__init__()
		
		self.conv = nn.ConvTranspose2d(in_planes, out_planes,
								kernel_size=kernel_size, stride=stride,
								padding=padding, bias=False) # verify bias false
		self.bn = nn.BatchNorm2d(out_planes,
								eps=0.001, # value found in tensorflow
								momentum=0.1, # default pytorch value
								affine=True)
		self.relu = nn.ReLU(inplace = True)

	def forward(self, x):
		out = self.relu(self.bn(self.conv(x)))
		return out
		
class BasicTransConv2d(nn.Module):
	
	def __init__(self, in_planes, out_planes, kernel_size, stride, padding = 0):
		
		super(BasicTransConv2d, self).__init__()
		self.transconv = nn.ConvTranspose2d(in_planes, out_planes,
											kernel_size = kernel_size, stride = stride,
											padding = padding, bias = False)
		self.bn = nn.BatchNorm2d(out_planes,
								eps=0.001, # value found in tensorflow
								momentum=0.1, # default pytorch value
								affine=True)
		self.relu = nn.ReLU(inplace = True)
		
	def forward(self, x):
		
		x = self.relu(self.bn(self.transconv(x)))
		return x

class featureExtractionB(nn.Module):

	def __init__(self, in_planes):
		super(featureExtractionB, self).__init__()
		self.path1 = nn.Sequential(
			BasicConv2d(in_planes, 128, kernel_size = 1, stride = 1, padding = 0),
			BasicConv2d(128, 192, kernel_size = 7, stride = 1, padding = 3)
		)
		
		self.path2 = nn.Sequential(
			BasicConv2d(in_planes, 128, kernel_size = 1, stride = 1, padding = 0),
			BasicConv2d(128, 192, kernel_size = 3, stride = 1, padding = 1)
		)
		
		self.path3 = nn.Sequential(
			BasicConv2d(in_planes, 128, kernel_size = 1, stride = 1, padding = 0),
			BasicConv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
		)
		
		self.path4 = nn.Sequential(
			BasicConv2d(in_planes, 128, kernel_size = 1, stride = 1, padding = 0),
			BasicConv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
			BasicConv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
		)

	def forward(self, x):
		Path1 = self.path1(x)
		Path2 = self.path2(x)
		Path3 = self.path3(x)
		Path4 = self.path4(x)
		
		out = torch.cat((Path1, Path2, Path3, Path4), 1)
		return out

class featureExtrationA(nn.Module): #192, k256/2, l256/2, m192/3, n192/3, p96/3, q192/3
		
	def __init__(self, in_planes):
		super(featureExtrationA, self).__init__() 
		
		self.path1 = nn.Sequential(
			BasicConv2d(in_planes, 96, kernel_size = 1, stride = 1, padding = 0),
			BasicConv2d(96, 192, kernel_size = 7, stride = 1, padding = 3)
		)
		
		self.path2 = BasicConv2d(in_planes, 192, kernel_size = 3, stride = 1, padding = 1)
		
		self.path3 = nn.Sequential(
			BasicConv2d(in_planes, 256, kernel_size = 1, stride = 1, padding = 0),
			BasicConv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
			BasicConv2d(256, 192, kernel_size = 3, stride = 1, padding = 1)
		)
		
	def forward(self, x):
		x1 = self.path1(x)
		x2 = self.path2(x)
		x3 = self.path3(x)
		out = torch.cat((x1, x2, x3), 1)
		
		return out

class ReflectionNetwork(nn.Module):
	
	def __init__(self, model):
		super(ReflectionNetwork, self).__init__()
		#ReflectionNetwork
		self.model = model
		#x 128*128
		self.convs1R = nn.Sequential(*self.model[0:7]) #64*64
		self.convs2R = nn.Sequential(*self.model[7:14]) #32*32
		self.convs3R = nn.Sequential(*self.model[14:24]) #16*16
		self.convs4R = nn.Sequential(*self.model[24:34])	#8*8
		self.convs5R = nn.Sequential(*self.model[34:44]) #4*4
		
		self.conv6R = BasicConv2d(512, 256, kernel_size = 3, stride = 1, padding = 1)
		
		self.featureExtractionA = featureExtrationA(256)
		
		self.deconv0R = nn.Sequential(
			BasicTransConv2d(576, 256, kernel_size = 4, stride = 2, padding = 1)
		)
		
		self.deconv1R = nn.Sequential(
			BasicTransConv2d(1536-128, 128, kernel_size = 4, stride = 2, padding = 1)
		)
		
		self.featureExtractionB = featureExtractionB(768-64)
		
		self.deconv2R = nn.Sequential(
			BasicTransConv2d(640, 64, kernel_size = 4, stride = 2, padding = 1)
		)
		
		self.deconv3R = nn.Sequential(
			BasicTransConv2d(384-32, 32, kernel_size = 4, stride = 2, padding = 1)
		)
		
		self.deconv4R = nn.Sequential(
			BasicTransConv2d(192-16, 16, kernel_size = 4, stride = 2, padding = 1)
		)
		
		#self.output1 = BasicConv2d(48, 16, kernel_size = 3, stride = 1, padding = 1)
		#self.output2 = BasicConv2d(16, 3, kernel_size = 3, stride = 1, padding = 1)
		self.output = nn.Sequential(
			BasicConv2d(49, 16, kernel_size = 3, stride = 1, padding = 1),
			BasicConv2d(16, 3, kernel_size =3, stride = 1, padding = 1)
		)
		
		#GradientNetwork
		#self.convg = nn.Conv2d(3, 48, kernel_size = 5, stride = 1, padding = 2)
		
		self.conv6_1 = BasicConv2d(512, 1024, kernel_size = 7, stride = 1, padding = 3)
		self.conv6_2 = BasicConv2d(1024, 512, kernel_size = 1, stride = 1, padding = 0)
		
		self.deconv5_1 = BasicConv2d(512, 256, kernel_size = 3, stride = 1, padding = 1)
		#self.deconv5_2 = Conv2dUnit(256, 256, kernel_size = 4, stride = 2, padding = 1)
		self.deconv5_2 = BasicConv2d(256, 256, kernel_size = 5, stride = 1, padding = 2)		
		self.featureEnhance5 = BasicConv2d(256, 128, kernel_size = 7, stride = 1, padding = 3)
		#32*32
		
		self.deconv4_1 = BasicConv2d(768, 128, kernel_size = 3, stride = 1, padding = 1)
		self.deconv4_2 = Conv2dUnit(128, 128, kernel_size = 4, stride = 2, padding = 1)
		self.featureEnhance4 = BasicConv2d(128, 64, kernel_size = 7, stride = 1, padding = 3)
		#64*64
		self.deconv3_1 = BasicConv2d(384, 64, kernel_size = 3, stride = 1, padding = 1)
		self.deconv3_2 = Conv2dUnit(64, 64, kernel_size = 4, stride = 2, padding = 1)
		self.featureEnhance3 = BasicConv2d(64, 32, kernel_size = 7, stride = 1, padding = 3)
		#128*128
		self.deconv2_1 = BasicConv2d(192, 32, kernel_size = 3, stride = 1, padding = 1)
		self.deconv2_2 = Conv2dUnit(32, 32, kernel_size = 4, stride = 2, padding = 1)
		self.featureEnhance2 = BasicConv2d(32, 16, kernel_size = 7, stride = 1, padding = 3)
		#256*256
		
		self.deconv1 = Conv2dUnit(96, 64, kernel_size = 4, stride = 2, padding = 1)
		
		self.pred1_contour = nn.Conv2d(64, 1, kernel_size = 5, stride = 1, padding = 2)
		self.sigmoid = nn.Sigmoid()
		self.scalar = torch.FloatTensor([1.1]).cuda()
		#self.attentionpool1 = nn.MaxPool2d(2, stride = 2, padding = 0)
		#self.attentionpool2 = nn.MaxPool2d(2, stride = 2, padding = 0)
		#self.attentionpool3 = nn.MaxPool2d(2, stride = 2, padding = 0)
		#self.attentionpool4 = nn.MaxPool2d(2, stride = 2, padding = 0)
		#self.attentionpool5 = nn.MaxPool2d(2, stride = 2, padding = 0)
		#GradientNetwork
		
		#self.sigmoid2 = nn.Sigmoid()
		
	def forward(self, x):
				
		#attention = self.sigmoid2(contour) + 0.5
		
		#attention1 = self.attentionpool1(attention)
		#attention2 = self.attentionpool2(attention1)
		#attention3 = self.attentionpool3(attention2)
		#attention4 = self.attentionpool4(attention3)
		#attention5 = self.attentionpool5(attention4)
		#Gradient Network
		
		#ReflectionNetwork
		convs1R = self.convs1R(x)#64*64
		convs2R = self.convs2R(convs1R)#32*32
		convs3R = self.convs3R(convs2R)#16*16
		convs4R = self.convs4R(convs3R)#8*8
		convs5R = self.convs5R(convs4R)#4*4
		
		
		#Gradient Network y 128*128
		#convsg = self.maxout(self.convg(x))
		#convsg = torch.cat((x, convsg), 1)
		
		#convs1g = self.pool1(self.convs1_2(self.convs1_1(convsg))) #64*64
		#convs2g = self.pool2(self.convs2_2(self.convs2_1(convs1g))) #32*32
		#convs3g = self.pool3(self.convs3_3(self.convs3_2(self.convs3_1(convs2g)))) #16*16
		#convs4g = self.pool4(self.convs4_3(self.convs4_2(self.convs4_1(convs3g)))) #8*8
		#convs5g = self.pool5(self.convs5_3(self.convs5_2(self.convs5_1(convs4g)))) #4*4
		
		#convs1R = self.convs1R(x)#64*64
		#convs2R = self.convs2R(convs1R)#32*32
		#convs3R = self.convs3R(convs2R)#16*16
		#convs4R = self.convs4R(convs3R)#8*8
		#convs5R = self.convs5R(convs4R)#4*4
		convs6g = self.conv6_2(self.conv6_1(convs4R))#4*4
		
		#print(convs6g.size())
		deconv5g = self.deconv5_2(self.deconv5_1(convs6g))#8*8
		#print(deconv5g.size())
		#print(convs4R.size())
		sum1g = torch.cat((deconv5g, convs4R),1) #256+512 = 768
		deconv4g = self.deconv4_2(self.deconv4_1(sum1g))#16*16
		#sum2g = deconv4g + convs3g 
		sum2g = torch.cat((deconv4g, convs3R),1) #128+256 = 384
		deconv3g = self.deconv3_2(self.deconv3_1(sum2g))#32*32
		#sum3g = deconv3g + convs2g #64+128 = 192
		sum3g = torch.cat((deconv3g, convs2R),1)
		deconv2g = self.deconv2_2(self.deconv2_1(sum3g))#64*64
		#sum4g = deconv2g+convs1g #32+64 = 96

		sum4g = torch.cat((deconv2g, convs1R),1)
		
		deconv1g = self.deconv1(sum4g)#128*128
		pred1_contour = self.pred1_contour(deconv1g)
		
		contour = self.sigmoid(pred1_contour)
		
		#Reflection
		conv6R = self.conv6R(convs5R)
		conv6R = conv6R
		featureA = self.featureExtractionA(conv6R)
		
		deconv01R = self.deconv0R(featureA)
		deconv02R = self.deconv0R(featureA)
		deconv03R = self.deconv0R(featureA) 
		#8*8 256, 256,256, 512
		deconv5gfE = self.featureEnhance5(deconv5g) #-12
		deconv0R = torch.cat((deconv01R, deconv02R, deconv03R, convs4R, deconv5gfE), 1)

		#deconv0R = deconv0R
		
		deconv11R = self.deconv1R(deconv0R)
		deconv12R = self.deconv1R(deconv0R)
		deconv13R = self.deconv1R(deconv0R)
		#16*16 128,128,128, 256
		deconv4gfE = self.featureEnhance4(deconv4g) #-64
		deconv1R = torch.cat((deconv11R, deconv12R, deconv13R, convs3R, deconv4gfE), 1)
		deconv1R = deconv1R
		featureB = self.featureExtractionB(deconv1R)
		
		deconv21R = self.deconv2R(featureB)
		deconv22R = self.deconv2R(featureB)
		deconv23R = self.deconv2R(featureB)
		#32*32 64, 64, 64, 128
		deconv3gfE = self.featureEnhance3(deconv3g) #-32
		deconv2R = torch.cat((deconv21R, deconv22R, deconv23R, convs2R, deconv3gfE), 1)
		deconv2R = deconv2R
		deconv31R = self.deconv3R(deconv2R)
		deconv32R = self.deconv3R(deconv2R)
		deconv33R = self.deconv3R(deconv2R)
		#64*64	32 32 32 64
		deconv2gfE = self.featureEnhance2(deconv2g) #-16
		deconv3R = torch.cat((deconv31R, deconv32R, deconv33R, convs1R, deconv2gfE), 1)
		deconv3R = deconv3R
		deconv41R = self.deconv4R(deconv3R)
		deconv42R = self.deconv4R(deconv3R)
		deconv43R = self.deconv4R(deconv3R)
		#128*128
		deconv4R = torch.cat((deconv41R, deconv42R, deconv43R, contour), 1)		
		deconv4R = deconv4R
		
		outputB = self.output(deconv4R)
		outputB = outputB*self.scalar.expand_as(outputB)
		outputR = x - outputB
		#Reflection network
		
		
		
		return outputB, outputR, contour
		
	def maxout(self, x):
		for step in range(24):
			maxtmp,index = torch.max(x[:,((step+1)*2-2):(step+1)*2,:,:], dim = 1)
			if step == 0:
				F1 = maxtmp.unsqueeze(1)
			else:
				F1 = torch.cat((F1, maxtmp.unsqueeze(1)), 1)
		return F1
