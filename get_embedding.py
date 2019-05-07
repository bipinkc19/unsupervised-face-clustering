import os
import cv2

encoder = cv2.dnn.readNetFromTorch((os.path.join('models', 'VGG_FACE.t7')))


def get_face_encodings(img):
    ''' Returns the encoding from the image and face location. '''
    faceBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, (224, 224), (0, 0, 0), swapRB=True, crop=False)
    encoder.setInput(faceBlob)
    encodings = encoder.forward()[0]

    return encodings
