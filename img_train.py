import numpy as np
import math
import cv2
import random
import torch
from torch.autograd import Variable
import os.path
from os import path
import os
import time
import threading
from threading import Lock
import copy

dev_string = "cuda:0"
#dev_string = "cpu"



img_width = 400 # reduce this if running out of CPU RAM
num_channels = 3 # we're using RGB images
kernel_width = 7 # an odd integer bigger than or equal to 3
padding_width = round((kernel_width - 1) / 2) # an integer
num_output_components = 2 # an integer representing the number of one-hot outputs

num_epochs = 10
learning_rate = 0.001

max_train_files_per_animal_type = 100
train_data_sliding_window_length = 64 # reduce this if running out of GPU RAM

num_recursions = 10
num_child_networks = 5



class Net(torch.nn.Module):

	def __init__(self, num_channels, num_output_components):
	
		super().__init__()

		self.model = torch.nn.Sequential(

		    torch.nn.Conv2d(in_channels = num_channels, out_channels = 16, kernel_size = kernel_width, stride = 1, padding = padding_width), 
		    torch.nn.ReLU(),
		    torch.nn.MaxPool2d(kernel_size = kernel_width),
  
		    torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = kernel_width, stride = 1, padding = padding_width),
		    torch.nn.ReLU(),
		    torch.nn.MaxPool2d(kernel_size = kernel_width),
		      
		    torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = kernel_width, stride = 1, padding = padding_width),
		    torch.nn.ReLU(),
		    torch.nn.MaxPool2d(kernel_size = kernel_width),
	
		    torch.nn.Flatten(),
		    torch.nn.Linear(64, 16),
		    torch.nn.ReLU(),
		    torch.nn.Linear(16, num_output_components),

			torch.nn.Softmax(dim = 1)
		)
  
	def forward(self, x):
		return self.model(x)





class float_image:

	def __init__(self, img):
		self.img = img

class image_type:

	def __init__(self, img_type, float_img):
		self.img_type = img_type
		self.float_img = float_img






def do_train_files(all_train_files):

	file_count = 0
	path = 'training_set/cats/'
	filenames = next(os.walk(path))[2]

	for f in filenames:

		file_count = file_count + 1
		if file_count >= max_train_files_per_animal_type:
			break;

		print(path + f)
		img = cv2.imread(path + f)
		
		if (img is None) == False:

			img = img.astype(np.float32)
			res = cv2.resize(img, dsize=(img_width, img_width), interpolation=cv2.INTER_LINEAR)
			flat_file = res / 255.0
			flat_file = np.transpose(flat_file, (2, 0, 1))
			all_train_files.append(image_type(0, flat_file))

		else:
			print("image read failure")


	file_count = 0
	path = 'training_set/dogs/'
	filenames = next(os.walk(path))[2]

	for f in filenames:

		file_count = file_count + 1
		if file_count >= max_train_files_per_animal_type:
			break;

		print(path + f)
		img = cv2.imread(path + f)
		
		if (img is None) == False:

			img = img.astype(np.float32)
			res = cv2.resize(img, dsize=(img_width, img_width), interpolation=cv2.INTER_LINEAR)
			flat_file = res / 255.0
			flat_file = np.transpose(flat_file, (2, 0, 1))
			all_train_files.append(image_type(1, flat_file))

		else:
			print("image read failure")




def do_test_files(in_net, filename, epoch, random_seed, num_recursion, num_child_network):

	file_handle = open(filename, 'a')

	path = 'test_set/cats/'
	filenames = next(os.walk(path))[2]

	cat_count = 0
	total_count = 0

	for f in filenames:

		img = cv2.imread(path + f)
			
		if (img is None) == False:

			img = img.astype(np.float32)
			res = cv2.resize(img, dsize=(img_width, img_width), interpolation=cv2.INTER_LINEAR)
			flat_file = res / 255.0
			flat_file = np.transpose(flat_file, (2, 0, 1))

		else:

			#print("image read failure")
			continue

		batch = torch.zeros((1, num_channels, img_width, img_width), dtype=torch.float32)
		batch[0] = torch.from_numpy(flat_file)
		
		x = Variable(batch)
		x = x.to(torch.device(dev_string))

		prediction = in_net(x)
		prediction = prediction.to(torch.device(dev_string))

	#	print(prediction)

		if prediction[0][0] > 0.5:
			cat_count = cat_count + 1

		total_count = total_count + 1

	file_handle.write(str(num_recursion) + " " + str(num_child_network) + "\n")
	file_handle.write(str(random_seed) + "\n")
	file_handle.write(str(epoch) + "\n")
	file_handle.write(str(cat_count / total_count) + "\n")
	file_handle.write(str(total_count) + "\n")

	print(str(num_recursion) + " " + str(num_child_network))
	print(str(random_seed))
	print(str(epoch))
	print(str(cat_count / total_count))
	print(str(total_count))


	path = 'test_set/dogs/'
	filenames = next(os.walk(path))[2]

	dog_count = 0
	total_count = 0

	for f in filenames:

		img = cv2.imread(path + f)
			
		if (img is None) == False:

			img = img.astype(np.float32)
			res = cv2.resize(img, dsize=(img_width, img_width), interpolation=cv2.INTER_LINEAR)
			flat_file = res / 255.0
			flat_file = np.transpose(flat_file, (2, 0, 1))

		else:

			#print("image read failure")
			continue

		batch = torch.zeros((1, num_channels, img_width, img_width), dtype=torch.float32)
		batch[0] = torch.from_numpy(flat_file)
		
		x = Variable(batch)
		x = x.to(torch.device(dev_string))

		prediction = in_net(x)
		prediction = prediction.to(torch.device(dev_string))

	#	print(prediction)

		if prediction[0][1] > 0.5:
			dog_count = dog_count + 1

		total_count = total_count + 1

	file_handle.write(str(num_recursion) + " " + str(num_child_network) + "\n")
	file_handle.write(str(random_seed) + "\n")
	file_handle.write(str(epoch) + "\n")
	file_handle.write(str(dog_count / total_count) + "\n")
	file_handle.write(str(total_count) + "\n")
	file_handle.write("\n")

	print(str(num_recursion) + " " + str(num_child_network))
	print(str(random_seed))
	print(str(epoch))
	print(str(dog_count / total_count))
	print(str(total_count))
	print("")

	file_handle.close()



class net_loss:

	def __init__(self, in_net, in_loss):
		self.in_net = in_net
		self.in_loss = in_loss


def do_network(lock, input_net, num_channels, num_output_components, all_train_files, filename, random_seed, num_epochs, num_recursions, num_child_networks, ret_vals, index):
	
	in_net = input_net

	optimizer = torch.optim.Adam(in_net.parameters(), lr = learning_rate)
	loss_func = torch.nn.MSELoss()

	random.seed(random_seed)

	loss = 0

	in_net.to(torch.device(dev_string))

	for epoch in range(num_epochs):

		random.shuffle(all_train_files)

		curr_train_file = 0
		train_files_remaining = len(all_train_files)
		buffer_size = train_data_sliding_window_length

		while train_files_remaining > 0:

			if train_files_remaining < buffer_size:
				buffer_size = train_files_remaining
	
			buffer = all_train_files[curr_train_file : (curr_train_file + buffer_size)]

			train_files_remaining -= buffer_size
			curr_train_file += buffer_size

			batch = np.zeros((buffer_size, num_channels, img_width, img_width), dtype=np.float32)
			ground_truth = np.zeros((buffer_size, num_output_components), dtype=np.float32)

			count = 0

			for i in buffer:

				batch[count] = i.float_img
		
				if i.img_type == 0: # cat

					ground_truth[count][0] = 1
					ground_truth[count][1] = 0
			
				elif i.img_type == 1: # dog
				
					ground_truth[count][0] = 0
					ground_truth[count][1] = 1

				count = count + 1
	
			x = Variable(torch.from_numpy(batch))
			y = Variable(torch.from_numpy(ground_truth))
			x = x.to(torch.device(dev_string))
			y = y.to(torch.device(dev_string))

			prediction = in_net(x)
			prediction = prediction.to(torch.device(dev_string))

			loss = loss_func(prediction, y)
			
			with lock:
				print(num_recursions, num_child_networks, train_files_remaining, epoch, loss)

			optimizer.zero_grad()	 # clear gradients for next train
			loss.backward()		 # backpropagation, compute gradients
			optimizer.step()		# apply gradients

		if ((epoch + 1) % 10 == 0):
			with lock:
				print("Running test files")
				do_test_files(in_net, filename, epoch + 1, random_seed, num_recursions, num_child_networks)

	
	ret_vals[index] = net_loss(in_net, loss)







if False: #path.exists('weights_' + str(prng_seed) + '.pth'):
	net.load_state_dict(torch.load('weights_' + str(prng_seed) + '.pth'))
	print("Loaded file successfully")
else:
	print("Training...")

	device = torch.device(dev_string)

	all_train_files = []
	do_train_files(all_train_files)

	prng_seed = round(time.time()*1000)

	filename = str(prng_seed) + ".txt"
	
	# Wipe the file if necessary
	file_handle = open(filename, "w")
	file_handle.close()

	start = time.time()

	lock = Lock()

	seed_net = Net(num_channels, num_output_components)

	ret_vals = []
	ret_vals.append(net_loss(seed_net, 0))

	do_network(lock, seed_net, num_channels, num_output_components, all_train_files, filename, prng_seed, num_epochs, 0, 0, ret_vals, 0)

	curr_net = ret_vals[0].in_net
	curr_loss = ret_vals[0].in_loss

	torch.save(seed_net.state_dict(), 'weights_' + str(prng_seed) + '.pth')

	for y in range(num_recursions):

		threads = []
		thread_ret_vals = []

		for x in range(num_child_networks):

			thread_ret_vals.append(net_loss(Net(num_channels, num_output_components), 0))

			prng_seed = prng_seed + 1

			# use a new copy of curr_net, since we can't pass by value in Python
			t = threading.Thread(target=do_network, args=(lock, copy.deepcopy(curr_net), num_channels, num_output_components, all_train_files, filename, prng_seed, num_epochs, y, x, thread_ret_vals, x))
			threads.append(t)
			threads[x].start()

		for x in range(num_child_networks):
			
			threads[x].join()

			if thread_ret_vals[x].in_loss < curr_loss:
				curr_loss = thread_ret_vals[x].in_loss
				curr_net = thread_ret_vals[x].in_net


	end = time.time()

	print(end - start)
