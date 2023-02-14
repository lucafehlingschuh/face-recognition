from mtcnn import MTCNN
import cv2
from model.Face_Detector import Detector
from torchvision import models
import torch.nn as nn
import torch
from collections import OrderedDict
from tensorflow.keras.models import load_model

"""
This script reads a video and executes the predictions. The selected video is saved under the given path
"""

if __name__ == "__main__":
    # Load all the needed models
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_features, 100)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(100, 3)),
        ('output', nn.Softmax(dim=1))]))
    # Load the trained weights. Here we load the model trained on the small dataset
    model.load_state_dict(torch.load('model\model-face-recognition-dictSmallDataset.pth', map_location=torch.device('cpu')))
    age_classifier = load_model('age_model.h5')
    # Change to eval mode (would be training otherwise)
    model.eval()
    # Create the mtcnn
    mtcnn = MTCNN()
    # Create the detector
    detector = Detector(mtcnn, model, age_classifier)

    img_array = []
    run = True
    while run:
        # Read an image and check the size of it.
        print("Type in the path of the video: ")
        path = input()
        if path == "q":
            run = False
            continue
        video = cv2.VideoCapture(path)
        if video is None:
            print("Video could not be found! Check name and path!")
            continue
        print("Type in the save-path of the marked video (save as .avi video): ")
        save_path = input()
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print('Number of frames: ' + str(num_frames))
        print('Current frame: ')

        for i in range(num_frames):
            print(str(i))
            retval, frame = video.read()
            if not retval:
                continue

            detection = detector.detect(frame)

            faces = detector.extract_face(frame, detection)

            names = detector.predict_name(faces)

            ages = detector.predict_age(faces)

            # draw on frame
            marked_frame = detector.draw_box(frame, detection, names, ages)

            if i == 0:
                height, width, layers = marked_frame.shape
                size = (width, height)
            img_array.append(marked_frame)

        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        run = False
