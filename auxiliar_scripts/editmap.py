import cv2
import os



def equals(vec1, vec2, size):
	for i in range(size):
		if vec1[i] != vec2[i]:
			return False
	return True

directory = os.getcwd()
directory = directory + "/maps/map4.png"

img = cv2.imread(directory, 1)
hight, width, h = img.shape
for i in range(hight):
	for j in range(width):
		if not equals(img[i, j], [0, 0, 0], 3):
			img[i, j] = [255, 255, 255]

cv2.imwrite('image2.png', img)