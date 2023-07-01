import cv2
import numpy as np
import os

path = 'training_set\\cats\\'
filenames = next(os.walk(path))[2]

for f in filenames:
	print(path + f)
	img = cv2.imread(path + f)
	res = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
	cv2.imwrite(path + f, res)

path = 'training_set\\dogs\\'
filenames = next(os.walk(path))[2]

for f in filenames:
	print(path + f)
	img = cv2.imread(path + f)
	res = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
	cv2.imwrite(path + f, res)




path = 'test_set\\cats\\'
filenames = next(os.walk(path))[2]

for f in filenames:
	print(path + f)
	img = cv2.imread(path + f)
	res = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
	cv2.imwrite(path + f, res)

path = 'test_set\\dogs\\'
filenames = next(os.walk(path))[2]

for f in filenames:
	print(path + f)
	img = cv2.imread(path + f)
	res = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
	cv2.imwrite(path + f, res)


