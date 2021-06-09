import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Step 1 : Setting up a path and get all files from the given path
path = 'known/'
Images = []

# Getting all the files in the path to be added in img_list as list items
img_list = os.listdir(path)

# reading all images in the array from the source folder
for img in img_list:

    # reading all images 1 by one as CurImg
    curImg = cv2.imread(path + '/'+img)

    # Adding those read images into our Images list
    Images.append(curImg)

# Step 2 : function that results in encoded values of all known and unknown faces when it is called
def Encodings(images):
    encodings = []
    for img in images:
        encode = face_recognition.face_encodings(img)[0]
        encodings.append(encode)
    return encodings


# setting all encoded values in a list called known_faces
known_faces = Encodings(Images)

# Getting Realtime Video using webcam
video_ = cv2.VideoCapture(0)

# This loop will match our known faces with every face in the frame
while True:
    _, image = video_.read()

    # decreasing image resolution by 1/2
    imageSm = cv2.resize(image, (int(image.shape[1]*0.5),int(image.shape[0]*0.5)))

    # Getting encoded values of a frame through webcam
    face_loc = face_recognition.face_locations(imageSm)
    face_encode = face_recognition.face_encodings(imageSm, face_loc)

    # matching faces for every frame through webcam
    for encoding, location in zip(face_encode, face_loc):

        # Returns true or false when the img is compared
        comp = face_recognition.compare_faces(known_faces, encoding)

        # Calculating Distance
        dist_ = face_recognition.face_distance(known_faces, encoding)

        # returns index of least distance
        matchIndex = np.argmin(dist_)

        # if distance of known and test image is smaller then person is known
        if comp[matchIndex]:
            y1, x2, y2, x1 = location
            y1, x2, y2, x1 = y1*2, x2*2, y2*2, x1*2
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # if distance of known and test image is greater then person is unknown
        else:
            y1, x2, y2, x1 = location
            y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # save file wih title current date time
            title = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            cv2.imwrite('unknown/'+title+'.jpeg', image)

    # Displaying the Webcan
    cv2.imshow('webcam', image)
    cv2.waitKey(1)
