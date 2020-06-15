# NGD, 2020-231C Final, Tianyang Zhao, cooperated with Jiaming Guo
# Reproducing Paper: Exact natural gradient in deep linear networks and application to the nonlinear case

import os
import copy
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models


# For Colab
class Args:

	# preamble
	device = 0
	if_cudnn = True
	seed = 1
	deterministic = True

	# path
	dataset_dir = './'
	outputs_dir = './'
	reload_dir = './'

	# training
	batch_size_train = 32
	batch_size_test = 32
	reload = False
	n_epochs = 10
	log_interval = 50

	# optim
	optim_class = 2	# 1 for conventional 1st order, 2 for ngd
	optim = 'SGD'
	optim_lr = 3e-2
	optim_betas = (0.9, 0.999)
	optim_eps = 1e-8
	optim_weight_decay = 0
	optim_momentum = 0.0

	# network
	dim_list = [784,400,200,100,50,10]
	W_init = 1e-2	# 0 will raise problem of trivial convergence in multiple layers??
	lambda_k = 1e-5
	alpha = 3e-2


class NGD_block(nn.Module):
	def __init__(self, input_dim, output_dim, W_init, lambda_k):
		super(NGD_block, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.lambda_k = lambda_k
		self.W = nn.Parameter(torch.rand(output_dim, input_dim) * W_init, requires_grad=True) # uniform [0, W_init]

	def forward(self, inputs, if_cudnn):
		self.inputs = copy.deepcopy(inputs.data)
		self.h = torch.matmul(self.W, inputs)
		if self.training:	# important
			self.h.retain_grad()
				# https://www.jianshu.com/p/ad66f2e38f2f
				# https://www.cnblogs.com/SivilTaram/p/pytorch_intermediate_variable_gradient.html
		return self.h

	def save_for_ngd(self):		# p^*
		inputs_t = torch.transpose(self.inputs,1,2)
		tmp = torch.matmul(self.inputs, inputs_t)
		self.Lambda = copy.deepcopy(torch.mean(tmp, 0).data)
			# .data important; deepcopy!!
		tmp = torch.matmul(self.h.grad, inputs_t)
		self.grads = copy.deepcopy(torch.mean(tmp, 0).data)

	def ngd_update(self, alpha, L, if_cudnn):		# p_\theta
		error = self.h.grad
		error_t = torch.transpose(error,1,2)
		tmp = torch.matmul(error, error_t)
		self.Lambda_tilde = torch.mean(tmp, 0)

		# TODO
		lambda_k = self.lambda_k

		# ngd
		eye_out = torch.eye(self.output_dim)
		eye_in = torch.eye(self.input_dim)
		if if_cudnn:
			eye_out = eye_out.type(torch.cuda.FloatTensor)
			eye_in = eye_in.type(torch.cuda.FloatTensor)

		left = torch.inverse(self.Lambda_tilde + np.sqrt(lambda_k)*eye_out)
		right = torch.inverse(self.Lambda + np.sqrt(lambda_k)*eye_in)

		delta = alpha/L * torch.matmul(torch.matmul(left, self.grads), right)

		# update
		# https://discuss.pytorch.org/t/custom-optimizer-in-pytorch/22397/3
		self.W.data -= delta	


class Nets(nn.Module):
	def __init__(self, dim_list, W_init, lambda_k, alpha):
		super(Nets, self).__init__()
		self.L = len(dim_list) - 1
		self.alpha = alpha
		self.blocks = nn.ModuleList()
			# https://saqibns.github.io/pytorch%20errors%20series/2018/11/07/optimizer-got-an-empty-parameter-list.html	
		for i in range(self.L):
			self.blocks.append(NGD_block(dim_list[i], dim_list[i+1], W_init, lambda_k))

	def forward(self, inputs, if_cudnn):
		inputs.requires_grad = True 	# important
		x = inputs.view(inputs.shape[0],784,1)

		for i in range(self.L - 1):
			x = self.blocks[i](x, if_cudnn)
			x = F.tanh(x)

		x = self.blocks[self.L-1](x, if_cudnn)
		self.outputs = x.view(x.shape[0], -1)	# do not change the order of this line and the next!
		return F.log_softmax(self.outputs)

	def save_for_ngd(self):
		for i in range(self.L):
			self.blocks[i].save_for_ngd()

	def ngd_update(self, if_cudnn):
		for i in range(self.L):
			self.blocks[i].ngd_update(alpha=self.alpha, L=self.L, if_cudnn=if_cudnn)


def main(args):
	
	torch.backends.cudnn.enabled = args.if_cudnn
	if args.deterministic:
		torch.manual_seed(args.seed)
		torch.backends.cudnn.deterministic = True

	# dataset
	train_loader = torch.utils.data.DataLoader(
						torchvision.datasets.MNIST(args.dataset_dir, train=True, download=True,
							transform=torchvision.transforms.Compose([
							torchvision.transforms.ToTensor(),
							torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
							batch_size=args.batch_size_train, shuffle=True)

	test_loader = torch.utils.data.DataLoader(
						torchvision.datasets.MNIST(args.dataset_dir, train=False, download=True,
							transform=torchvision.transforms.Compose([
							torchvision.transforms.ToTensor(),
							torchvision.transforms.Normalize((0.1307,), (0.3081,))])), 
							batch_size=args.batch_size_test, shuffle=True)

	# model
	network = Nets(	dim_list=args.dim_list, W_init=args.W_init, lambda_k=args.lambda_k,
					alpha=args.alpha)

	# Optimizer
	if args.optim == 'SGD':
		optimizer = optim.SGD(network.parameters(), lr=args.optim_lr, \
			momentum=args.optim_momentum, weight_decay=args.optim_weight_decay)
	elif args.optim == 'Adam':
		optimizer = optim.Adam(network.parameters(), lr=args.optim_lr, \
			betas=args.optim_betas, eps=args.optim_eps, weight_decay=args.optim_weight_decay)
	
	# for GPU usage
	if args.if_cudnn:
		def use_gpu():
			return torch.cuda.is_available()
		if use_gpu():
			network.cuda(args.device)

	# Reload
	if args.reload == True:
		map_location = 'cpu'
		if args.if_cudnn:
			map_location = 'gpu'
		network_state_dict = torch.load('{}model.pth'.format(args.reload_dir), map_location=map_location)
		network.load_state_dict(network_state_dict)
		optimizer_state_dict = torch.load('{}optimizer.pth'.format(args.reload_dir), map_location=map_location)
		optimizer.load_state_dict(optimizer_state_dict)

	# Placeholders
	train_losses = []
	train_counter = []
	test_losses = []
	test_counter = [i*len(train_loader.dataset) for i in range(args.n_epochs + 1)]

	# Train an epoch
	def train(epoch):
		network.train()
		for batch_idx, (data, target) in enumerate(train_loader):

			if args.if_cudnn:
				data = data.type(torch.cuda.FloatTensor)
				target = target.type(torch.cuda.LongTensor)

			optimizer.zero_grad()
			output = network(data, args.if_cudnn)

			if args.optim_class == 1:
				loss = F.nll_loss(output, target)
				loss.backward()
				optimizer.step()

			else:
				# 2nd order experiment

				# sample from p^*
				loss = F.nll_loss(output, target)
				loss.backward()
				network.save_for_ngd()

				# sample from p_\theta and ngd update
				optimizer.zero_grad()
				output = network(data, args.if_cudnn)
				fake_target = torch.multinomial(torch.exp(output), 1).view(target.shape[0])

				fake_loss = F.nll_loss(output, fake_target)
				fake_loss.backward()
				network.ngd_update(args.if_cudnn)

			if batch_idx % args.log_interval == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, batch_idx * len(data), len(train_loader.dataset),
					100. * batch_idx / len(train_loader), loss.item()))
				train_losses.append(loss.item())
				train_counter.append(
					(batch_idx*args.batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
				torch.save(network.state_dict(), '{}model.pth'.format(args.outputs_dir))    # '.../outputs/inhibition-2005/ckpt/model.pth'
				torch.save(optimizer.state_dict(), '{}optimizer.pth'.format(args.outputs_dir))

	# Test an epoch
	def test():
		network.eval()
		test_loss = 0
		correct = 0
		with torch.no_grad():
			for data, target in test_loader:

				if args.if_cudnn:
					data = data.type(torch.cuda.FloatTensor)
					target = target.type(torch.cuda.LongTensor)

				output = network(data, args.if_cudnn)
				test_loss += F.nll_loss(output, target, size_average=False).item()
				pred = output.data.max(1, keepdim=True)[1]
				correct += pred.eq(target.data.view_as(pred)).sum()
		test_loss /= len(test_loader.dataset)
		test_losses.append(test_loss)
		print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			test_loss, correct, len(test_loader.dataset),
			100. * correct / len(test_loader.dataset)))

	# Workflow
	test()
	for epoch in range(1, args.n_epochs + 1):
		train(epoch)
		test()

	# Training Curve
	fig = plt.figure()
	plt.plot(train_counter, train_losses, color='blue')
	plt.scatter(test_counter, test_losses, color='red')
	plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
	plt.xlabel('number of training examples seen')
	plt.ylabel('negative log likelihood loss')
	plt.show()


args = Args()
main(args)

