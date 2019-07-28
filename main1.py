import argparse, os
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from network1 import ReflectionNetwork
from utils import PSNR, MatrixToImage
from torchvision.utils import save_image
from SSIMLoss import SSIMLoss
from SILoss import SILoss
from MMDLoss import MMDLoss
import torchvision
from tensorboardX import SummaryWriter
import flow_transforms
import scipy.io as sio
import datetime
import torchvision.transforms as transforms
import datasets
import time

model_names = 'sasa'

dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch FlowNet Training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', metavar='DATASET', default='mpi_sintel_both',
                    choices=dataset_names,
                    help='dataset type : ' +
                    ' | '.join(dataset_names))
parser.add_argument('-s', '--split', default=80,
                    help='test-val split file')
parser.add_argument('--arch', '-a', metavar='ARCH', default='FaceGeneration',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names))
parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=30000, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')
parser.add_argument('--no-date', action='store_true',
                    help='don\'t append date timestamp to folder' )
parser.add_argument('--milestones', default=[18,60,80], nargs='*', help='epochs at which learning rate is divided by 2')

n_iter = 0

def main():
	torch.cuda.set_device(1)
	global args, best_EPE, save_path
	args = parser.parse_args()

	save_path = '{},{},{}epochs{},b{},lr{}'.format(
		args.arch,
		args.solver,
		args.epochs,
		',epochSize'+str(args.epoch_size) if args.epoch_size > 0 else '',
		args.batch_size,
		args.lr)
	if not args.no_date:
		timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
		save_path = os.path.join(timestamp,save_path)	

	save_path = os.path.join(args.dataset,save_path)
	print('=> will save everything to {}'.format(save_path))
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	###tensorboard
	train_writer = SummaryWriter(os.path.join(save_path,'train'))
	test_writer = SummaryWriter(os.path.join(save_path,'test'))
	
	training_output_writers = []
	for i in range(15):
		training_output_writers.append(SummaryWriter(os.path.join(save_path,'train',str(i))))
		
	output_writers = []
	for i in range(15):
		output_writers.append(SummaryWriter(os.path.join(save_path,'test',str(i))))
	################
	
	######## Data transformation code
	input_transform = transforms.Compose([
		flow_transforms.ArrayToTensor(), ##from numpy array to tensor
		transforms.Normalize(mean=[0,0,0], std=[255,255,255]) ##divide each channel of the image by 255
	]) ###input image transform 
	background_transform = transforms.Compose([
		flow_transforms.ArrayToTensor(),
		transforms.Normalize(mean=[0,0,0], std=[255,255,255])
	]) ####background image transform
	
	gradient_transform = transforms.Compose([
		flow_transforms.ArrayToTensor(),
		transforms.Normalize(mean=[0,0,0], std=[255,255,255])
	]) ##gradient transform
	
	reflection_transform = transforms.Compose([
		flow_transforms.ArrayToTensor(),
		transforms.Normalize(mean=[0,0,0], std=[255,255,255])
	]) ##reflection transform 
	
	co_transform = flow_transforms.Compose([
		flow_transforms.RandomVerticalFlip(), ##flip the image vertically
		flow_transforms.RandomHorizontalFlip(), ##flip the image horizontally
		#flow_transforms.RandomColorWarp(0,0)
	])
	
	#####data loading see the mpi_sintel_both file in datasets folder
	print("=> fetching img pairs in '{}'".format(args.data))
	train_set, test_set = datasets.__dict__[args.dataset](
		args.data,
		transform=input_transform,
		gradient_transform = gradient_transform,
		reflection_transform = reflection_transform,
		background_transform = background_transform,
		co_transform = co_transform
	)
	
	print('{} samples found, {} train samples, {} test samples '.format(len(train_set), len(train_set), len(test_set)))
	train_loader = torch.utils.data.DataLoader(
		train_set, batch_size=args.batch_size,
		num_workers=0, pin_memory=True, shuffle=True)
		
	val_loader = torch.utils.data.DataLoader(
		test_set, batch_size=args.batch_size,
		num_workers=0, pin_memory=True, shuffle=False)
		
	vgg = torchvision.models.vgg16_bn(pretrained = True) ##load the vgg model
	vgglist = list(vgg.features.children()) 

	model = ReflectionNetwork(vgglist) ##load the training model

	optimizer = optim.Adam(model.parameters(), lr = args.lr)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)

	model = model.cuda()

	if args.pretrained:
		data = torch.load('/home/Test25/model2/checkpoint_81.pth.tar')
		model.load_state_dict(data['state_dict'])

	####load the loss functions
	ssimLoss = SSIMLoss().cuda()
	L1Loss = nn.L1Loss().cuda()
	siLoss = SILoss().cuda()
	mmdLoss = MMDLoss().cuda()

	num_epoches = 80
	
	for epoch in range(num_epoches+1):
		scheduler.step()
		print('epoch {}'.format(epoch))
		
		train(train_loader, optimizer, ssimLoss, L1Loss, siLoss, mmdLoss, model, train_writer, training_output_writers, epoch) ##training code
		
		validate(val_loader, model, L1Loss, test_writer, output_writers, epoch)
		
		if epoch % 1 == 0:
			#save the model
			save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'optimizer' : optimizer.state_dict(),
			},epoch+1)

def train(train_loader, optimizer, ssimLoss, L1Loss, siLoss, mmdLoss, model, train_writer, training_output_writers, epoch):
	global n_iter
	loss_background = AverageMeter()
	loss_reflection = AverageMeter()
	loss_gradient = AverageMeter()
	data_time = AverageMeter()
	
	#switch to train mode
	model.train()
	end = time.time()

	for iteration, (mixture, background, reflection, gradient) in enumerate(train_loader):
		
		data_time.update(time.time() - end)
		
		input = [j.cuda() for j in mixture]
		input = Variable(input[0], requires_grad = True) ##read the input image

		background = [j.cuda() for j in background]
		background = Variable(background[0], requires_grad = False) ##read the background image

		reflection = [j.cuda() for j in reflection]
		reflection = Variable(reflection[0], requires_grad = False) ##read the reflection image

		gradient = [j.cuda() for j in gradient]
		gradient = Variable(gradient[0], requires_grad = False) ##read the gradient image

		output = model.forward(input)
		
		outputB = output[0]
		outputR = output[1]
		outputG = output[2]
			
		bl = 0.8*ssimLoss(outputB, background) + L1Loss(outputB, background)  + 0.5*mmdLoss(outputB, background) ##loss functios for background
		rl = ssimLoss(outputR, reflection) + 0.5*mmdLoss(outputR, reflection)##loss functios for reflection
		gl = siLoss(outputG, gradient)##loss functios for gradient
				
		loss_background.update(bl.data[0], input.size(0))
		loss_reflection.update(rl.data[0], input.size(0))
		loss_gradient.update(gl.data[0], input.size(0))
		
		totalLoss = bl + rl + gl ##based on my experiments different weighting coefficients on rl may generate different results, if you make the coefficients smaller, maybe the results can be better
		#totalLoss = bl+gl

		optimizer.zero_grad()
		totalLoss.backward() ##backward the loss fucntions
		optimizer.step()
		
		train_writer.add_scalar('train_loss', bl.data[0], n_iter) ###show the loss function values on tensorboard
		
		###show the intermediate results on tensorboard
		if iteration < len(training_output_writers):  # log first output of first batches
			#if epoch == 0:
			training_output_writers[iteration].add_image('TInputs', input[0].data.cpu(), epoch)
			training_output_writers[iteration].add_image('targetB', background[0].data.cpu(), epoch)
			training_output_writers[iteration].add_image('outputB', outputB[0].data.cpu(), epoch)
		n_iter += 1
		if iteration % args.print_freq == 0:
			print("Finish{}/{}epoch, {} iterations Loss-b: {}, Loss_g:{}".format(epoch, 80, iteration, loss_background.avg, loss_gradient.avg))

class AverageMeter(object):
	def __init__(self):
		self.reset()
	
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
		
	def update(self, val, n = 1):
		self.val = val
		self.sum += val*n
		self.count += n
		self.avg = self.sum / self.count

def save_checkpoint(state, epoch, filename='checkpoint.pth.tar'):
	filename = "checkpoint_{}.pth.tar".format(epoch)
	torch.save(state, os.path.join(save_path, filename))

def validate(val_loader, model, L1Loss, test_writer, output_writers, epoch):	
	#switch to evaluate mode
	batch_time = AverageMeter()
	losses = AverageMeter()
	
	model.eval()
	
	end = time.time()
	for i, (mixture, background, reflection, gradient) in enumerate(val_loader):
		input = [j.cuda() for j in mixture]
		input = Variable(input[0], requires_grad = True)

		background = [j.cuda() for j in background]
		background = Variable(background[0], requires_grad = False)

		reflection = [j.cuda() for j in reflection]
		reflection = Variable(reflection[0], requires_grad = False)

		gradient = [j.cuda() for j in gradient]
		gradient = Variable(gradient[0], requires_grad = False)
		
		output = model(input)
		
		outputB = output[0]
		
		loss = L1Loss(outputB, background)
		losses.update(loss.data[0], background.size(0))
		batch_time.update(time.time() - end)
		end = time.time()
		
		###show the estimated results on tensorboard
		test_writer.add_scalar('evaluation_loss', loss.data[0], epoch)
		if i < len(output_writers):  # log first output of first batches
			output_writers[i].add_image('TGroundTruth', input[0].data.cpu(), epoch)
			output_writers[i].add_image('ToutputB', outputB[0].data.cpu(), epoch)
			#output_writers[i].add_image('targetB', background[0].data.cpu(), epoch)
		
if __name__ == "__main__":
	main()	