import os
import cv2
import numpy as np

model_architecture = os.path.join('.//models', 'deploy.prototxt')
model_weights = os.path.join('.//models', 'res10_300x300_ssd_iter_140000.caffemodel')
detector_model = cv2.dnn.readNetFromCaffe(model_architecture, model_weights)
confidence_level = 0.95

def get_face_boundings(img):
    ''' Returns the face boundings from image. '''
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
    (h, w) = img.shape[:2]
    detector_model.setInput(imageBlob)
    detections = detector_model.forward()
    boundings=[]

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # filter out weak detections
        if confidence > confidence_level:
            
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            boundings.append((startX, startY, endX, endY))

    return boundings


def get_boxes_from_boundings(img, boundings):
    ''' Put rectangle in a image from the bounding boxes in boundings. '''

    for (x1, y1, x2, y2) in boundings:
        cv2.rectangle(img,(x1, y1), (x2, y2), (255, 255, 123), 1)
    
    return img
