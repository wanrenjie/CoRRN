from __future__ import division
import torch
import random
import numpy as np
import numbers
import types
import scipy.ndimage as ndimage

'''Set of tranform random routines that takes both input and target as arguments,
in order to have random but coherent transformations.
inputs are PIL Image pairs and targets are ndarrays'''


class Compose(object):

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target, reflection, mask):
        for t in self.co_transforms:
            input,target,reflection, mask = t(input,target, reflection, mask)
        return input, target, reflection, mask


class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert(isinstance(array, np.ndarray))
        if len(array.shape)==3:
            array = np.transpose(array, (2, 0, 1))
            # handle numpy array
            tensor = torch.from_numpy(array)
        else:
            tensor = torch.from_numpy(array).unsqueeze(0)
        # put it from HWC to CHW format
        return tensor.float()


class Lambda(object):
    """Applies a lambda as a transform"""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, input,target):
        return self.lambd(input,target)


class CenterCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, target, reflection, mask):
        h1, w1, _ = inputs[0].shape
        h2, w2, _ = inputs[1].shape
        th, tw = self.size
        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))
        x2 = int(round((w2 - tw) / 2.))
        y2 = int(round((h2 - th) / 2.))

        inputs[0] = inputs[0][y1: y1 + th, x1: x1 + tw]
        #inputs[1] = inputs[1][y2: y2 + th, x2: x2 + tw]
        target[0] = target[0][y1: y1 + th, x1: x1 + tw]
        reflection[0] = reflection[0][y1: y1 + th, x1: x1 + tw]
        mask[0] = mask[0][y1: y1 + th, x1: x1 + tw]

        return inputs, target, reflection, mask


class Scale(object):
    """ Rescales the inputs and target arrays to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, size, order=2):
        self.size = size
        self.order = order

    def __call__(self, inputs, target, reflection, mask):
        h, w, _ = inputs[0].shape
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return inputs,target
        if w < h:
            ratio = self.size/w
        else:
            ratio = self.size/h

        inputs[0] = ndimage.interpolation.zoom(inputs[0], ratio, order=self.order)
        #inputs[1] = ndimage.interpolation.zoom(inputs[1], ratio, order=self.order)
        reflection[0] = ndimage.interpolation.zoom(inputs[0], ratio, order=self.order)
        target[0]= ndimage.interpolation.zoom(target[0], ratio, order=self.order)
        mask[0] = mask.interpolation.zoom(mask[0], ratio, order=self.order)

        return inputs, target, reflection, mask


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, target, reflection, mask):
        h, w, _ = inputs[0].shape
        th, tw = self.size
        if w == tw and h == th:
            return inputs,target

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs[0] = inputs[0][y1: y1 + th,x1: x1 + tw]
        target[0] = target[0][y1: y1 + th,x1: x1 + tw]
        reflection[0] = reflection[0][y1: y1 + th,x1: x1 + tw]
        mask[0] = mask[0][y1: y1 + th,x1: x1 + tw]


        return inputs, target, reflection, mask


class RandomHorizontalFlip(object):
	"""Randomly horizontally flips the given PIL.Image with a probability of 0.5
	"""

	def __call__(self, inputs, background, reflection, gradient):
		if random.random() < 0.5:
			inputs[0] = np.copy(np.fliplr(inputs[0]))
			
			background[0] = np.copy(np.fliplr(background[0]))

			reflection[0] = np.copy(np.fliplr(reflection[0]))
			
			gradient[0] = np.copy(np.fliplr(gradient[0]))

			
		return inputs, background, reflection, gradient


class RandomVerticalFlip(object):
	"""Randomly horizontally flips the given PIL.Image with a probability of 0.5
	"""

	def __call__(self, inputs, background, reflection, gradient):
		if random.random() < 0.5:
			inputs[0] = np.copy(np.flipud(inputs[0]))
			#inputs[1] = np.copy(np.flipud(inputs[1]))
			#inputs[3] = np.copy(np.flipud(inputs[3]))
			
			background[0] = np.copy(np.flipud(background[0]))
			#target[1] = np.copy(np.flipud(target[1]))
			#target[3] = np.copy(np.flipud(target[3]))
			reflection[0] = np.copy(np.flipud(reflection[0]))
			
			gradient[0] = np.copy(np.flipud(gradient[0]))
			#mask[1] = np.copy(np.flipud(mask[1]))
		return inputs, background, reflection, gradient


class RandomRotate(object):
	"""Random rotation of the image from -angle to angle (in degrees)
	This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
	angle: max angle of the rotation
	interpolation order: Default: 2 (bilinear)
	reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
	diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
	"""

	def __init__(self, angle, diff_angle=0, order=2, reshape=False):
		self.angle = angle
		self.reshape = reshape
		self.order = order
		self.diff_angle = diff_angle

	def __call__(self, inputs,target,reflection,mask):
		applied_angle = random.uniform(-self.angle,self.angle)
		diff = random.uniform(-self.diff_angle,self.diff_angle)
		angle1 = applied_angle - diff/2
		angle2 = applied_angle + diff/2
		angle1_rad = angle1*np.pi/180

		h, w, _ = target[0].shape

		inputs[0] = ndimage.interpolation.rotate(inputs[0], angle1, reshape=self.reshape, order=self.order)
		target[0] = ndimage.interpolation.rotate(target[0], angle1, reshape=self.reshape, order=self.order)
		reflection[0] = ndimage.interpolation.rotate(target[0], angle1, reshape=self.reshape, order=self.order)		
		mask[0] = ndimage.interpolation.rotate(mask[0], angle1, reshape=self.reshape, order=self.order)
		
		return inputs, target, reflection, mask


class RandomTranslate(object):
    def __init__(self, translation):
        if isinstance(translation, numbers.Number):
            self.translation = (int(translation), int(translation))
        else:
            self.translation = translation

    def __call__(self, inputs, target):
        h, w, _ = inputs[0].shape
        th, tw = self.translation
        tw = random.randint(-tw, tw)
        th = random.randint(-th, th)
        if tw == 0 and th == 0:
            return inputs, target
        # compute x1,x2,y1,y2 for img1 and target, and x3,x4,y3,y4 for img2
        x1,x2,x3,x4 = max(0,tw), min(w+tw,w), max(0,-tw), min(w-tw,w)
        y1,y2,y3,y4 = max(0,th), min(h+th,h), max(0,-th), min(h-th,h)

        inputs[0] = inputs[0][y1:y2,x1:x2]
        inputs[1] = inputs[1][y3:y4,x3:x4]
        target = target[y1:y2,x1:x2]
        target[:,:,0] += tw
        target[:,:,1] += th

        return inputs, target


class RandomColorWarp(object):
	def __init__(self, mean_range=0, std_range=0):
		self.mean_range = mean_range
		self.std_range = std_range

	def __call__(self, inputs, background, reflection, gradient):
		random_std = np.random.uniform(-self.std_range, self.std_range, 3)
		random_mean = np.random.uniform(-self.mean_range, self.mean_range, 3)
		random_order = np.random.permutation(3)

		inputs[0] *= (1 + random_std)
		inputs[0] += random_mean

		#inputs[1] *= (1 + random_std)
		#inputs[1] += random_mean

		inputs[0] = inputs[0][:,:,random_order]
		#inputs[1] = inputs[1][:,:,random_order]

		return inputs, background, reflection, gradient
