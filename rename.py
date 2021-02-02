import os
import shutil


destination = r'/home/shahir/Documents/4-2/Pattern Lab/Project/dataset/Dataset for drawing/dataset/dataset_circle'
i = 0
directory = r'/home/shahir/Documents/4-2/Pattern Lab/Project/dataset/Dataset for drawing/dataset/dataset_circle'

for filename in os.listdir(directory):
	if filename.endswith(".png"):

		new_file_name = "circle_" + str(i) + ".png"
		os.rename(filename, new_file_name)
		      
		i += 1

