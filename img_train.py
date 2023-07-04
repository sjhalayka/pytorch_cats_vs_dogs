import numpy as np
import math
import cv2
import random
import torch
from torch.autograd import Variable

import os.path
from os import path



img_width = 32
num_input_components = img_width*img_width*3
num_output_components = 1

num_epochs = 100
learning_rate = 0.00001



class Net(torch.nn.Module):

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

class float_image:

	def __init__(self, img):
		self.img = img

class image_type:

	def __init__(self, img_type, float_img):
		self.img_type = img_type
		self.float_img = float_img





net = Net()


if False: #path.exists('weights_' + str(num_input_components) + '_' + str(num_epochs) + '.pth'):
	net.load_state_dict(torch.load('weights_' + str(num_input_components) + '_' + str(num_epochs) + '.pth'))
	print("loaded file successfully")
else:
	print("training...")





	all_train_files = []



	path = 'training_set\\cats\\'
	filenames = next(os.walk(path))[2]

	for f in filenames:

		print(path + f)
		img = cv2.imread(path + f).astype(np.float32)
		res = cv2.resize(img, dsize=(img_width, img_width), interpolation=cv2.INTER_LINEAR)
		flat_file = np.asarray(res).flatten() / 255.0
		all_train_files.append(image_type(0, flat_file))

	path = 'training_set\\dogs\\'
	filenames = next(os.walk(path))[2]

	for f in filenames:

		print(path + f)
		img = cv2.imread(path + f).astype(np.float32)
		res = cv2.resize(img, dsize=(img_width, img_width), interpolation=cv2.INTER_LINEAR)
		flat_file = np.asarray(res).flatten() / 255.0
		all_train_files.append(image_type(1, flat_file))




	optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
	loss_func = torch.nn.MSELoss()



	
	batch = np.zeros((len(all_train_files), num_input_components), dtype=np.float32)
	ground_truth = np.zeros((len(all_train_files), 1), dtype=np.float32)

	random.shuffle(all_train_files)

	count = 0

	for i in all_train_files:

		batch[count] = i.float_img
		ground_truth[count] = i.img_type
		count = count + 1

	for epoch in range(num_epochs):

		x = Variable(torch.from_numpy(batch))
		y = Variable(torch.from_numpy(ground_truth))

		prediction = net(x)	 
		loss = loss_func(prediction, y)

		print(epoch, loss)

		optimizer.zero_grad()	 # clear gradients for next train
		loss.backward()		 # backpropagation, compute gradients
		optimizer.step()		# apply gradients



	#torch.save(net.state_dict(), 'weights_' + str(num_input_components) + '_' + str(num_epochs) + '.pth')



path = 'test_set\\cats\\'
filenames = next(os.walk(path))[2]

cat_count = 0
total_count = 0

for f in filenames:

#	print(path + f)
	img = cv2.imread(path + f).astype(np.float32)
	res = cv2.resize(img, dsize=(img_width, img_width), interpolation=cv2.INTER_LINEAR)
	flat_file = np.asarray(res).flatten() / 255.0

	batch = torch.from_numpy(flat_file)

	prediction = net(Variable(batch))

	if prediction < 0.5:
		cat_count = cat_count + 1

	total_count = total_count + 1
#	print(batch)
#	print(prediction)

print(cat_count / total_count)
print(total_count)



path = 'test_set\\dogs\\'
filenames = next(os.walk(path))[2]

dog_count = 0
total_count = 0

for f in filenames:

#	print(path + f)
	img = cv2.imread(path + f).astype(np.float32)
	res = cv2.resize(img, dsize=(img_width, img_width), interpolation=cv2.INTER_LINEAR)
	flat_file = np.asarray(res).	flatten() / 255.0

	batch = torch.from_numpy(flat_file)

	prediction = net(Variable(batch))

	if prediction > 0.5:
		dog_count = dog_count + 1

	total_count = total_count + 1
#	print(batch)
#	print(prediction)

print(dog_count / total_count)
print(total_count)
