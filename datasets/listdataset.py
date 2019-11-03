import torch.utils.data as data
import os
import os.path
from scipy.ndimage import imread
import numpy as np
import scipy.io as sio


def load_flo(path):
	with open(path, 'rb') as f:
		magic = np.fromfile(f, np.float32, count=1)
		assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
		h = np.fromfile(f, np.int32, count=1)[0]
		w = np.fromfile(f, np.int32, count=1)[0]
		data = np.fromfile(f, np.float32, count=2*w*h)
	# Reshape data into 3D array (columns, rows, bands)
	data2D = np.resize(data, (w, h, 2))
	return data2D


def default_loader(root, path_imgm, path_imgb, path_imgr, path_gradient):
	imgm = [os.path.join(root,path) for path in path_imgm]
	imgb = [os.path.join(root,path) for path in path_imgb]
	imgr = [os.path.join(root,path) for path in path_imgr]
	imggradient = [os.path.join(root,path) for path in path_gradient]
	
	return [imread(img).astype(np.float32) for img in imgm],[imread(img).astype(np.float32) for img in imgb], [imread(img).astype(np.float32) for img in imgr], [sio.loadmat(img)['edgeB'].astype(np.float32) for img in imggradient]
	
class ListDataset(data.Dataset):
	def __init__(self, root, path_list, transform=None, gradient_transform = None, reflection_transform = None, background_transform=None,
					co_transform=None, loader=default_loader):
	
		self.root = root
		self.path_list = path_list
		self.transform = transform
		self.gradient_transform = gradient_transform
		self.reflection_transform = reflection_transform
		self.background_transform = background_transform
		self.co_transform = co_transform
		self.loader = loader
	
	def __getitem__(self, index):
		inputs, background, reflection, gradient = self.path_list[index]

		inputs, background, reflection, gradient = self.loader(self.root, inputs, background, reflection, gradient)
		
		if self.co_transform is not None:
			inputs, background, reflection, gradient = self.co_transform(inputs, background, reflection, gradient)
			
		if self.transform is not None:
			inputs[0] = self.transform(inputs[0])
		
		if self.background_transform is not None:
			background[0] = self.background_transform(background[0])

		if self.reflection_transform is not None:
			reflection[0] = self.reflection_transform(reflection[0])

		if self.gradient_transform is not None:
			gradient[0] = self.gradient_transform(gradient[0])

			
		return inputs, background, reflection, gradient
	
	def __len__(self):
		return len(self.path_list)
