import cv2
from mtcnn import MTCNN
import Face_Detector
import os
from Rescale import Rescale
import torch.nn as nn
import torch
from torchvision import models

"""
This script takes x faces from an folder. (expected structure: folder containing folder for each persons. 
"""

mtcnn = MTCNN()
detector = Face_Detector.Detector(mtcnn)

print("Type in the path of the folder: ")
path = input()
print("Type in the save-path of the folder: ")
save_path = input()
dic = os.listdir(path)
print("Type in the expected number of extracted faces: ")
max_images = input()
print("Type in the maximum number of images per person: ")
max_images_pp = input()
num_faces = 0
scaler = Rescale()
for folder in dic:
    files = os.listdir(path + '/' + folder)
    if len(files) == 0:
        continue
    if num_faces == max_images:
        break
    if num_faces % 50 == 0:
        print('Image ' + str(num_faces))

    # reading the image
    img = cv2.imread(path + '/' + folder + '/' + files[1], 1)
    try:
        detection = detector.detect(img)
        faces = detector.extract_face(img, detection)
    except cv2.error:
        print('Error in file ' + files[0])
    for k in range(len(faces)):
        if k == max_images_pp:
            break
        if faces[k].size != 0:
            num_faces += 1
            cv2.imwrite(save_path + '/' + str(num_faces) + '.jpg', faces[k])

print('Successfully saved ' + str(num_faces))