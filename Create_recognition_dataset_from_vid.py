import cv2
from mtcnn import MTCNN
import Face_Detector
from Rescale import Rescale

"""
This script extracts all  faces found in a video.
"""

mtcnn = MTCNN()
detector = Face_Detector.Detector(mtcnn)

#Import video
print("Type in the path of the video: (as mp4 format)")
path = input()
print("Type in the the save location for the data set (folder): ")
save_path = input()
video = cv2.VideoCapture('path')


num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print('Found ' + str(num_frames) + ' frames in the video.')
num_faces = 0
scaler = Rescale()

for i in range(num_frames):
    #if num_faces == 60:
     #   break

    #if ((i % 2) != 0):
     #   continue
    if i % 100 == 0:
        print('Frame ' + str(i))
    retval, frame = video.read()

    if not retval:
        continue

    # Predict
    detection = detector.detect(frame)
    faces = detector.extract_face(frame, detection)
    for k in range(len(faces)):
        if not k==0:
            break
        num_faces += 1
        save_path = save_path + '\image_' + str(num_faces) + '.jpg'
        cv2.imwrite(save_path, faces[k])

print('video1 finished')
print('Successfully saved ' + str(num_faces))