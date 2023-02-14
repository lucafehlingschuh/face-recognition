This readme describes the functions of the different scripts/notebooks used for the face detection and recognition part of the ItW project by Luca Fehling-Schuh.


------------------------------------------------------------------------------------------------------------------------------------
Datasets used:

- for classes LucaH and LucaFS: self-created dataset (see Create_recognition_dataset_from_vid.py)

- for unknown class: Labeled Faces in the Wild Dataset
http://vis-www.cs.umass.edu/lfw/

------------------------------------------------------------------------------------------------------------------------------------


Create_recognition_dataset_from_images.py 
 This script was used to create the dataset for the Unknown-person dataset out of the lfw_funneled-dataset

Create_recognition_dataset_from_vid.py
 This script was used to create the dataset for the Luca_FS and Luca_H dataset out of an video (couple of videos per person for multiple lighting conditions)

Create_TrainVal.py
 This skript splits the three datasets into train and val datasets

TrainRecogModel.ipynb
 This notebook trains and optimizes the face recognition model

model-face-recognition-dictSmallDataset.pth
 This is the final model with the best validation and generalization scores. Used to predict the videos in the ItW-presentation

model-face-recognition-dictFinal3(advModel_batch64).pth
 This is the model which was trained on the big dataset (statistics can be found in the ItW-presentation)

Face_Detector.py
 Skript contains the face_detector and performs all the predictions (face recognition + age detection)

Rescale.py
 Used to rescale extracted faces to a given size

Run_webcam.py
 This script enables the usage of the predictors with the webcam

Video_predict.py
 This script reads a video and executes the predictions. Used to produce the videos which can be found at the end of the ItW-presentation




