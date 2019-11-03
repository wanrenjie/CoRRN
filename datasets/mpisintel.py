import os.path
import glob
from .listdataset import ListDataset
import flow_transforms

def make_train_dataset(rootpath):
	images = []
	for folder in os.listdir(rootpath):
		imagefiles = [file for file in os.listdir(rootpath+'/'+folder) if file.endswith('_b.jpg')]
		for imgname in imagefiles:
			prefix = imgname[0:-6]
			nameb = prefix + '_b'
			namer = prefix + '_r'
			nameg = prefix + '_g'
			namem = prefix + '_m'
			
			imgm = os.path.join(rootpath, folder, namem+'.jpg')
			imgb = os.path.join(rootpath, folder, nameb+'.jpg')
			imgr = os.path.join(rootpath, folder, namer+'.jpg')
			imgg = os.path.join(rootpath, folder, nameg+'.mat')

			if not (os.path.isfile(imgm) and os.path.isfile(imgb) and os.path.isfile(imgr) and os.path.isfile(imgg)):
				continue	

			images.append([[imgm],[imgb],[imgr],[imgg]])
	return images	

def make_evaluation_dataset(rootpath):
	images = []
	for folder in os.listdir(rootpath):
		imagefiles = [file for file in os.listdir(rootpath+'/'+folder) if file.endswith('_b.jpg')]
		for imgname in imagefiles:
			prefix = imgname[0:-6]
			nameb = prefix + '_b'
			namer = prefix + '_r'
			namem = prefix + '_m'
			nameg = prefix + '_g'
			imgm = os.path.join(rootpath, folder, namem + '.jpg')
			imgb = os.path.join(rootpath, folder, nameb + '.jpg')
			imgr = os.path.join(rootpath, folder, namer + '.jpg')
			imgg = os.path.join(rootpath, folder, nameg+'.mat')
			
			if not (os.path.isfile(imgm) and os.path.isfile(imgb) and os.path.isfile(imgr) and os.path.isfile(imgg)):
				continue
			images.append([[imgm], [imgb], [imgr],[imgg]])
	return images
		
def mpi_sintel_both(root, transform=None, gradient_transform=None, reflection_transform = None,background_transform = None,
					co_transform=None):

	roote = '/home/rjwan/Datageneration/imagereflection'
	train_list = make_train_dataset(root)
	test_list = make_evaluation_dataset(roote)
	
	train_dataset = ListDataset(root, train_list, transform, gradient_transform, reflection_transform, background_transform, co_transform)
	test_dataset = ListDataset(roote, test_list, transform, None, background_transform)

	return train_dataset, test_dataset
