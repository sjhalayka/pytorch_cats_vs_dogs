import numpy as np
import math
import cv2
import random
import torch
from torch.autograd import Variable
import os.path
from os import path
import time




dev_string = "cuda:0" # "cpu"

img_width = 400 # reduce this if running out of CPU RAM
num_channels = 3 # RGB images
kernel_width = 7 # an odd integer
padding_width = round((kernel_width - 1) / 2)
num_output_components = 2 # an integer representing the number of one-hot outputs

num_epochs = 200
learning_rate = 0.0005 # test... was 0.001 before

max_train_files_per_animal_type = 100000
train_data_sliding_window_length = 64 # reduce this if running out of GPU RAM

num_recursions = 0
num_child_networks = 0



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




def do_test_files(in_net, file_name, epoch, random_seed):

	file_handle = open(file_name, 'a')

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

	file_handle.write(str(random_seed) + "\n")
	file_handle.write(str(epoch) + "\n")
	file_handle.write(str(cat_count / total_count) + "\n")
	file_handle.write(str(total_count) + "\n")
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

	file_handle.write(str(random_seed) + "\n")
	file_handle.write(str(epoch) + "\n")
	file_handle.write(str(dog_count / total_count) + "\n")
	file_handle.write(str(total_count) + "\n\n")
	print(str(random_seed))
	print(str(epoch))
	print(str(dog_count / total_count))
	print(str(total_count))

	file_handle.close()





def do_network(in_net, num_channels, num_output_components, all_train_files, random_seed, num_epochs):

	if (in_net is None):
		in_net = Net(num_channels, num_output_components)

	net = in_net



	random.seed(random_seed)

	optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
	loss_func = torch.nn.MSELoss()

	loss = 0;

	net.to(torch.device(dev_string))

	file_handle = open("output.txt", "w")
	file_handle.close()

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

			prediction = net(x)
			prediction = prediction.to(torch.device(dev_string))

			loss = loss_func(prediction, y)
			
			print(train_files_remaining, epoch, loss)

			optimizer.zero_grad()	 # clear gradients for next train
			loss.backward()		 # backpropagation, compute gradients
			optimizer.step()		# apply gradients

		if ((epoch + 1) % 10 == 0):
			do_test_files(net, "output.txt", epoch + 1, random_seed)

	return net, loss






if False: #path.exists('weights_' + str(img_width) + '_' + str(num_epochs) + '.pth'):
	net.load_state_dict(torch.load('weights_' + str(img_width) + '_' + str(num_epochs) + '.pth'))
	print("loaded file successfully")
else:
	print("training...")

	device = torch.device(dev_string)

	all_train_files = []
	do_train_files(all_train_files)

	start = time.time()

	curr_net, curr_loss = do_network(None, num_channels, num_output_components, all_train_files, round(time.time()*1000), num_epochs)

	for y in range(num_recursions):

		for x in range(num_child_networks):

			print(y, num_recursions, x, num_child_networks)

			net, loss = do_network(curr_net, num_channels, num_output_components, all_train_files, round(time.time()*1000), num_epochs)

			if loss < curr_loss:

				curr_loss = loss
				curr_net = net


	end = time.time()

	print(end - start)

#	torch.save(net.state_dict(), 'weights_' + str(img_width) + '_' + str(num_epochs) + '.pth')
