import numpy as np
import math
import cv2
import random
import torch
from torch import flatten
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import os.path
from os import path



img_width = 64
num_channels = 3

#num_input_components = img_width*img_width*num_channels
num_output_components = 2

num_epochs = 1000
learning_rate = 0.001






class Net(torch.nn.Module):
	def __init__(self, num_channels, num_output_components, all_train_files_len):
		super().__init__()
		self.model = torch.nn.Sequential(
		    #Input = 3 x 32 x 32, Output = 32 x 32 x 32
		    torch.nn.Conv2d(in_channels = num_channels, out_channels = 32, kernel_size = 3, padding = 1), 
		    torch.nn.ReLU(),
		    #Input = 32 x 32 x 32, Output = 32 x 16 x 16
		    torch.nn.MaxPool2d(kernel_size=2),
  
		    #Input = 32 x 16 x 16, Output = 64 x 16 x 16
		    torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1),
		    torch.nn.ReLU(),
		    #Input = 64 x 16 x 16, Output = 64 x 8 x 8
		    torch.nn.MaxPool2d(kernel_size=2),
		      
		    #Input = 64 x 8 x 8, Output = 64 x 8 x 8
		    torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1),
		    torch.nn.ReLU(),
		    #Input = 64 x 8 x 8, Output = 64 x 4 x 4
		    torch.nn.MaxPool2d(kernel_size=2),
  
		    torch.nn.Flatten(),
		    torch.nn.Linear(4096, all_train_files_len),
		    torch.nn.ReLU(),
		    torch.nn.Linear(all_train_files_len, num_output_components)
		)
  
	def forward(self, x):
		return self.model(x)

"""
	def __init__(self):
		super(Net, self).__init__()
		self.hidden1 = torch.nn.Linear(num_input_components, 8192)
		self.hidden2 = torch.nn.Linear(8192, 1024) 
		self.hidden3 = torch.nn.Linear(1024, 128)
		self.predict = torch.nn.Linear(128, num_output_components)

	def forward(self, x):
		x = torch.tanh(self.hidden1(x))		
		x = torch.tanh(self.hidden2(x))
		x = torch.tanh(self.hidden3(x))
		x = self.predict(x)    # linear output
		return x
"""



class float_image:

	def __init__(self, img):
		self.img = img

class image_type:

	def __init__(self, img_type, float_img):
		self.img_type = img_type
		self.float_img = float_img




if False: #path.exists('weights_' + str(num_input_components) + '_' + str(num_epochs) + '.pth'):
	net.load_state_dict(torch.load('weights_' + str(num_input_components) + '_' + str(num_epochs) + '.pth'))
	print("loaded file successfully")
else:
	print("training...")





	all_train_files = []

	file_count = 0

	path = 'training_set/cats/'
	filenames = next(os.walk(path))[2]

	for f in filenames:

		file_count = file_count + 1
		#if file_count >= 100:
		#	break;

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
		#if file_count >= 100:
		#	break;

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



	


	net = Net(num_channels, num_output_components, len(all_train_files))
	optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
	loss_func = torch.nn.MSELoss()

	batch = np.zeros((len(all_train_files), num_channels, img_width, img_width), dtype=np.float32)
	ground_truth = np.zeros((len(all_train_files), num_output_components), dtype=np.float32)
	
	for epoch in range(num_epochs):
		
		random.shuffle(all_train_files)

		count = 0

		for i in all_train_files:

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

		prediction = net(x)	 
		loss = loss_func(prediction, y)

		print(epoch, loss)

		optimizer.zero_grad()	 # clear gradients for next train
		loss.backward()		 # backpropagation, compute gradients
		optimizer.step()		# apply gradients



	#torch.save(net.state_dict(), 'weights_' + str(num_input_components) + '_' + str(num_epochs) + '.pth')



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

		print("image read failure")
		continue

	batch = torch.zeros((1, num_channels, img_width, img_width), dtype=torch.float32)
	batch[0] = torch.from_numpy(flat_file)

	prediction = net(Variable(batch))

	if prediction[0][0] > prediction[0][1]:
		cat_count = cat_count + 1

	total_count = total_count + 1

print(cat_count / total_count)
print(total_count)





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

		print("image read failure")
		continue

	batch = torch.zeros((1, num_channels, img_width, img_width), dtype=torch.float32)
	batch[0] = torch.from_numpy(flat_file)

	prediction = net(Variable(batch))

	if prediction[0][0] < prediction[0][1]:
		dog_count = dog_count + 1

	total_count = total_count + 1

print(dog_count / total_count)
print(total_count)