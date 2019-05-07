import os
import cv2
import shutil
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from face_recognition.face_recognition_cli import image_files_in_folder

from get_embedding import get_face_encodings
from get_face import get_face_boundings, get_boxes_from_boundings


X = []
Y = []

# Path to the image dataset and number of required clusters.
# PATH_TO_DATASET = './dslr_data/'
# NUM_OF_CLUSTERS = 26
PATH_TO_DATASET = input("enter the path like folder-name: ")
NUM_OF_CLUSTERS = int(input("enter the number of people in the jumble of photos : "))

def padded_image(im):
    ''' Padd image of it isn't 300 * 300 size as for face detection of this model atleast 300 * 300 image size is required. '''
    desired_size=300
    old_size = im.shape[:2]

    if old_size[1]>120:
        desired_size = 600
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_im

# Loop through each person in the training set
def main():
    ''' Read the image from directory and cluster them using K-means algorithm, then store the images of different cluster to different folders. '''
    for image_path in os.listdir(PATH_TO_DATASET):
        img = cv2.imread(os.path.join(PATH_TO_DATASET, image_path))

        if img.shape[0] < 300 or img.shape[1] < 300:
            img = padded_image(img)
        boundings = get_face_boundings(img)

        if len(boundings) == 0:
            continue

        for bounding in boundings:       
            cropped_face = img[bounding[1]:bounding[3], bounding[0]:bounding[2]]
            if cropped_face.shape[0] == 0 or cropped_face.shape[1] == 0:
                continue
            embeddings = get_face_encodings(cropped_face)

        X.append(embeddings)
        Y.append(os.path.join(PATH_TO_DATASET, image_path))

    kmeans = KMeans(n_clusters=NUM_OF_CLUSTERS, random_state=0).fit(X)
    results = kmeans.labels_

    for label in np.unique(results):
        os.mkdir(str(label))

    print('--')
    for (label, path) in zip(results, Y):
        shutil.copy2(path, os.path.join('./', str(label), str(datetime.now()) + '.jpg'))

if __name__ == '__main__':
    main()
