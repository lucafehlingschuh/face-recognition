import cv2
from mtcnn import MTCNN
from model.Face_Detector import Detector
from torchvision import models
import torch.nn as nn
import torch
from collections import OrderedDict
from tensorflow.keras.models import load_model


"""
This script predicts on the live webcam.
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
    model.load_state_dict(
        torch.load('model\model-face-recognition-dictSmallDataset.pth', map_location=torch.device('cpu')))
    age_classifier = load_model('age_model.h5')
    # Change to eval mode (would be training otherwise)
    model.eval()
    # Create the mtcnn
    mtcnn = MTCNN()
    # Create the detector
    detector = Detector(mtcnn, model, age_classifier)

    # 0 as index to use the wecam of the laptop
    cap = cv2.VideoCapture(0)
    one = True
    while True:
        ret, frame = cap.read()
        try:
            # detect face box, probability and landmarks
            faces = detector.detect(frame)
            extracted_faces = detector.extract_face(frame, faces)
            names = detector.predict_name(extracted_faces)
            ages = detector.predict_age(extracted_faces)
            # draw on frame
            frame_draw = detector.draw_box(frame, faces, names, ages)

        except:
           pass

        # Show the frame
        cv2.imshow('Face Detection', frame_draw)

        # End the stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
