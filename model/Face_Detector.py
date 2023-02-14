import cv2
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
from Rescale import Rescale

class Detector():
    """
    Provides functionality to detect faces with an mtcnn detector and functionality to mark the faces and extract them.
    Default enlargement factor for boxes is 10%. Changable using the set_enlarge_factor() function.
    """

    def __init__(self, mtcnn, classifier=None, age_classifier=None):
        """
        Constructor
        :param mtcnn: An MTCNN Detector
        """
        self.mtcnn = mtcnn
        self.classifier = classifier
        self.scaler = Rescale()
        self.age_classifier = age_classifier

    def detect(self, image):
        """
        Detect all faces in the given image
        :param image: Image as an ndarray
        :return: Dictionary with the information about the location, keypoints and probabilities of the faces found in the image.
        """
        # Detect faces and return dict with the information
        return self.mtcnn.detect_faces(image)


    def draw_box(self, image, faces, names, ages): # Type of faces: dict
        """
        Draws a rectangle on every face found and marks the keypoints.
        :param image: Image as an ndarray
        :param faces: Dictionary with the information about the location, keypoints and probabilities of the faces found in the image.
        :param names: The string name information of the detected faces
        :param ages: The string age information of the detected faces
        :return: Image as an ndarray with the added information about the faces found
        """
        try:
            for i in range(len(faces)):
                # Get the position of the detected face (type of box: list)
                box = faces[i]['box']
                # Get the keypoints of the detected face (type of keypoints: dict)
                keypoints = faces[i]['keypoints']

                confidence = faces[i]['confidence']

                # Draw red rectangle over the detected face (ractangle param: image, x&y range, BGR-color tuple,
                # thickness of the lines)
                cv2.rectangle(image,(box[0], box[1]),(box[0] + box[2], box[1] + box[3]),(0, 0, 250),2)
                # Names
                cv2.putText(image, names[i],
                            (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 3)
                # Ages
                cv2.putText(image, ages[i],
                            (box[0], box[1] + box[3] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 3)
                # Draw red circles over the keypoints
                cv2.circle(image, (keypoints['left_eye']), 2, (0, 0, 250), 2)
                cv2.circle(image, (keypoints['right_eye']), 2, (0, 0, 250), 2)
                cv2.circle(image, (keypoints['nose']), 2, (0, 0, 250), 2)
                cv2.circle(image, (keypoints['mouth_left']), 2, (0, 0, 250), 2)
                cv2.circle(image, (keypoints['mouth_right']), 2, (0, 0, 250), 2)
        except:
            x = int(image.shape[1] * 0.25)
            y = int(image.shape[0] * 0.5)
            cv2.putText(image, 'No face detected!', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 250), 1)
        return image


    def extract_face(self, image, faces):
        """
        Extracts all detected faces from the image rescaled to square format
        :param image: Image as an ndarray
        :param faces: Dictionary with the information about the location, keypoints and probabilities of the faces
        found in the image.
        :return: List containing all the faces found in square format
        """
        extracted_faces = []
        if not len(faces) == 0:
            for i in range(len(faces)):
                x1, y1, width, height = faces[i]['box']
                if width == height:
                    x2, y2 = x1 + width, y1 + height
                    extracted_faces.append(image[y1:y2, x1:x2])
                elif width < height:
                    scale = height - width
                    width = height
                    x1 = x1 - int(scale/2)
                    if x1 < 0:
                        x1 = 0
                    x2, y2 = x1 + width, y1 + height
                    if x2 > len(image[0]):
                        x1 = x1 - (x2 - len(image[0]))
                        x2 = len(image[0])
                    extracted_faces.append(image[y1:y2, x1:x2])
                else:
                    scale = width - height
                    height = width
                    y1 = y1 - int(scale/2)
                    if y1 < 0:
                        y1 = 0
                    x2, y2 = x1 + width, y1 + height
                    if y2 > len(image):
                        y1 = y1 - (y2 - len(image))
                        y2 = len(image)
                    extracted_faces.append(image[y1:y2, x1:x2])
        return extracted_faces


    def predict_name(self, extracted_faces):
        """
        Predict the names
        :param extracted_faces: List containing all the faces found in square format
        :return: List containing the names
        """
        names = []
        for i in range(len(extracted_faces)):
            image = Image.fromarray(extracted_faces[i])
            # Perform the needed image preparations
            preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            preprocessed_img = preprocess(image)
            batch = torch.unsqueeze(preprocessed_img, 0)
            with torch.no_grad():
                outputs = self.classifier(batch)
                loss, pred = torch.max(outputs, 1)
            prediction = np.array(pred[0])
            if prediction == 0:
                names.append('Luca Fehling-Schuh')
            elif prediction == 1:
                names.append('Luca Hartmann')
            else:
                names.append('Unknow person')
        return names


    def predict_age(self, extracted_faces):
        """
        Predict the ages
        :param extracted_faces: List containing all the faces found in square format
        :return: List containing the ages
        """
        ages = []
        age_intervals = [(1, 3), (4, 13), (14, 23), (24, 30), (31, 45), (45, 60), (61, 116)]
        for i in range(len(extracted_faces)):
            scaled_image = self.scaler.rescale(image=extracted_faces[i])
            img = scaled_image / 255
            pred_cathegorical = self.age_classifier.predict(np.expand_dims(img, axis=0))
            prediction = np.argmax(pred_cathegorical)
            pred_interval = age_intervals[prediction]
            ages.append('Predicted age: ' + str(pred_interval[0]) + '-' + str(pred_interval[1]) +
                 ' years\n with {:.2f}'.format(pred_cathegorical[0,prediction]*100) + '% certainty')
        return ages

