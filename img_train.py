import numpy as np
import math
import cv2
import random
import torch
from torch.autograd import Variable

import os.path
from os import path



img_width = 256
num_input_components = img_width*img_width*3
num_output_components = 1

num_epochs = 100
cut_off = 0.0001
learning_rate = 0.0001



class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.hidden1 = torch.nn.Linear(num_input_components, 256)
		self.hidden2 = torch.nn.Linear(256, 128) 
		self.hidden3 = torch.nn.Linear(128, 32)
		self.predict = torch.nn.Linear(32, num_output_components)

	def forward(self, x):
		x = torch.tanh(self.hidden1(x))		
		x = torch.tanh(self.hidden2(x))
		x = torch.tanh(self.hidden3(x))
		x = self.predict(x)    # linear output
		return x



class image_type:
	def __init__(self, img_type, image):
		self.img_type = img_type
		self.image = image





net = Net()


if False: #path.exists('weights_' + str(num_input_components) + '_' + str(num_epochs) + '.pth'):
	net.load_state_dict(torch.load('weights_' + str(num_input_components) + '_' + str(num_epochs) + '.pth'))
	print("loaded file successfully")
else:
	print("training...")





	all_train_files = []



	file_count = 0
	path = 'training_set\\cats\\'
	filenames = next(os.walk(path))[2]

	for f in filenames:

		print(path + f)
		img = cv2.imread(path + f).astype(np.float32)
		res = cv2.resize(img, dsize=(img_width, img_width), interpolation=cv2.INTER_LINEAR)
		flat_file = np.asarray(res).flatten()
		all_train_files.append(image_type(0, flat_file))

		file_count = file_count + 1

		if file_count >= 100:
			break



	file_count = 0
	path = 'training_set\\dogs\\'
	filenames = next(os.walk(path))[2]

	for f in filenames:

		print(path + f)
		img = cv2.imread(path + f).astype(np.float32)
		res = cv2.resize(img, dsize=(img_width, img_width), interpolation=cv2.INTER_LINEAR)
		flat_file = np.asarray(res).flatten()
		all_train_files.append(image_type(1, flat_file))
		
		file_count = file_count + 1

		if file_count >= 100:
			break




	optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
	loss_func = torch.nn.MSELoss()



	for epoch in range(num_epochs):

		random.shuffle(all_train_files)

		for i in all_train_files:

			batch = torch.from_numpy(i.image)
			ground_truth = np.zeros(1, np.float32)
			ground_truth[0] = i.img_type

			x = Variable(batch)
			y = Variable(torch.from_numpy(ground_truth))

			prediction = net(x)	 
			loss = loss_func(prediction, y)

			print(epoch, loss)

			optimizer.zero_grad()	 # clear gradients for next train
			loss.backward()		 # backpropagation, compute gradients
			optimizer.step()		# apply gradients

			if loss.detach().numpy() < cut_off:
				break




	#torch.save(net.state_dict(), 'weights_' + str(num_input_components) + '_' + str(num_epochs) + '.pth')






path = 'test_set\\cats\\'
filenames = next(os.walk(path))[2]

cat_count = 0
total_count = 0

for f in filenames:

#	print(path + f)
	img = cv2.imread(path + f).astype(np.float32)
	res = cv2.resize(img, dsize=(img_width, img_width), interpolation=cv2.INTER_LINEAR)
	
	batch = torch.from_numpy(np.asarray(res).flatten())

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
	
	batch = torch.from_numpy(np.asarray(res).flatten())

	prediction = net(Variable(batch))

	if prediction > 0.5:
		dog_count = dog_count + 1

	total_count = total_count + 1
	#	print(batch)
	#print(prediction)

print(dog_count / total_count)
print(total_count)







"""
batch = torch.zeros(4, 2, dtype=torch.float32)

batch[0][0] = 0;
batch[0][1] = 0;
batch[1][0] = 0;
batch[1][1] = 1;
batch[2][0] = 1;
batch[2][1] = 0;
batch[3][0] = 1;
batch[3][1] = 1;

gt = ground_truth(batch.numpy())
prediction = net(batch).detach().numpy()

print(gt)
print("\n")
print(prediction)

"""
