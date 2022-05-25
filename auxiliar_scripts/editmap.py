import os

import cv2
import numpy as np


directory = os.getcwd()
directory = directory + "/maps/map4.png"

img = cv2.imread(directory, 1)
hight, width, h = img.shape
for i in range(hight):
	for j in range(width):
		if not np.all(img[i, j] == [0, 0, 0]):
			img[i, j] = [255, 255, 255]

cv2.imwrite('image2.png', img)
