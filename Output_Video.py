import cv2
import numpy as np
import time
import imutils
import os
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from skimage import transform
from tensorflow import expand_dims

#load the mask detector model
mask_net = load_model('face_mask_detector.model')

#load the face detector model
prototxtPath = r'C:\Users\Daniel Adama\Desktop\PROGRAMMING\1.3PythonDatascience\datascience\FaceDetection\CovidFaceMask\face_detector\deploy.prototxt'
weightsPath = r'C:\Users\Daniel Adama\Desktop\PROGRAMMING\1.3PythonDatascience\datascience\FaceDetection\CovidFaceMask\face_detector\res10_300x300_ssd_iter_140000.caffemodel'
face_net = cv2.dnn.readNet(prototxtPath, weightsPath)


#initialize the webcam
webcam = VideoStream(src=0).start()
time.sleep(2.0)

font = cv2.FONT_HERSHEY_SIMPLEX

def detectAndpredictMask(frame, face_net, mask_net):
    #grab the dimensions of a frame and then construct a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    
    #pass the blob through the face network to get the face detections
    face_net.setInput(blob)
    face_detections = face_net.forward()
    
    #get the faces, locations and predictions from the face mask model
    faces = []
    locations = []
    predictions = []
    
    #loop over the face detections
    for i in range(0, face_detections.shape[2]):
        #extract the confidence associated with the face detection
        confidence = face_detections[0, 0, i, 2]
        
        #filter out weak detections to ensure the confidence is greater than the minimum
        if confidence > 0.4:
            #compute the (x, y) coordinates of the bounding box for the object
            box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            #making sure the bounding boxes fall within the dimensions of the frame
            (x, y) = (max(0,x), max(0,y))
            (x1, y1) = (min(w-1,x1), min(h-1,y1))
            #slice the frame, convert to RGB and preprocess
            face = frame[y:y1, x:x1]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = imutils.resize(face, 224, 224)
            face = img_to_array(face)
            face = preprocess_input(face)
            face = transform.resize(face, (224, 224, 3))#make sure the image is resized as the model to prevent error
            face = expand_dims(face, axis=0)#expand to fit the frame
            #attach to the empty lists
            faces.append(face)
            locations.append((x,y,x1,y1))
    #only make predictions if atleast one face is detected
    if len(faces) > 0:
            #for faster processing we will make both predictions(0 & 1) at the same time
            predictions = mask_net.predict(faces) #making predictions on the on the image
            
    #return a 2-tuple of the face locations and their corresponding locations
    return (locations, predictions)
            
while True:
    
    #capture the frame from the videostream
    frame = webcam.read()
    
    #resize the image for faster processing
    frame = imutils.resize(frame, 550, 550)
    
    #detect faces in the frame and determine whether they wearing a face mask or not
    (locations, predictions) = detectAndpredictMask(frame, face_net, mask_net)
    
    #loop over the detected face
    for (box, pred) in zip(locations, predictions):
        #unpack the bounding box and predictions
        (x,y, x1,y1) = box
        (mask, withoutmask) = pred
        #determine the class label, color and text
        label = 'Mask' if mask > withoutmask else 'No mask'
        color = (0, 0, 255) if label == 'Mask' else (255, 255, 255)
        
        #include the probability in the label
        label = '{}: {:.2f}%'.format(label, max(mask, withoutmask)*100)
        
    #display the label and bounding box rectangle
    cv2.putText(frame, label, (x,y-10), font, 0.5, color, 2)
    cv2.rectangle(frame, (x,y), (x1,y1), color, 2)
    
    cv2.imshow('LIVE!', frame)
    key = cv2.waitKey(5)
    if key == 27: #exit when Esc key is pressed
        break
        
#stop handle to the webcam
cv2.destoryAllWindows()
webcam.stop()