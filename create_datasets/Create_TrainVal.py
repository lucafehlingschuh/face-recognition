import os
import random
import cv2



"""
This skript splits the three datasets into train and val datasets
"""


path_LucaFS = 'FaceRecognition/LucaFS_Dataset/'
files_LucaFS = os.listdir(path_LucaFS)
random.shuffle(files_LucaFS)

path_LucaH = 'FaceRecognition/LucaH_Dataset/'
files_LucaH = os.listdir(path_LucaH)
random.shuffle(files_LucaH)

path_Unknown = 'FaceRecognition/Unknown_Dataset/'
files_Unknown = os.listdir(path_Unknown)
random.shuffle(files_Unknown)

for i in range(int(0.8*len(files_LucaFS))):
    img = cv2.imread(path_LucaFS + files_LucaFS[i], 1)
    path = 'FaceRecognition/Train/Luca_FS/' + files_LucaFS[i]
    cv2.imwrite(path, img)

for i in range(int(0.8*len(files_LucaFS)), len(files_LucaFS)):
    img = cv2.imread(path_LucaFS + files_LucaFS[i], 1)
    path = 'FaceRecognition/Val/Luca_FS/' + files_LucaFS[i]
    cv2.imwrite(path, img)



for i in range(int(0.8 * len(files_LucaH))):
    img = cv2.imread(path_LucaH + files_LucaH[i], 1)
    path = 'FaceRecognition/Train/Luca_H/' + files_LucaH[i]
    cv2.imwrite(path, img)

for i in range(int(0.8 * len(files_LucaH)), len(files_LucaH)):
    img = cv2.imread(path_LucaH + files_LucaH[i], 1)
    path = 'FaceRecognition/Val/Luca_H/' + files_LucaH[i]
    cv2.imwrite(path, img)



for i in range(int(0.8 * len(files_Unknown))):
    img = cv2.imread(path_Unknown + files_Unknown[i], 1)
    path = 'FaceRecognition/Train/Unknown/' + files_Unknown[i]
    cv2.imwrite(path, img)

for i in range(int(0.8 * len(files_Unknown)), len(files_Unknown)):
    img = cv2.imread(path_Unknown + files_Unknown[i], 1)
    path = 'FaceRecognition/Val/Unknown/' + files_Unknown[i]
    cv2.imwrite(path, img)
