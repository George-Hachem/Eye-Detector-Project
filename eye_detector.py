import cv2
from random import randrange

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

webcam = cv2.VideoCapture(0)
while True:
    is_frame_read, the_frame = webcam.read()

    black_n_white_frame = cv2.cvtColor(the_frame, cv2.COLOR_BGR2GRAY)

    eye_coordinates = eye_cascade.detectMultiScale(black_n_white_frame)
    for i in range(len(eye_coordinates)):

        (pos_x,pos_y,width,height) = eye_coordinates[i]
        cv2.rectangle(the_frame,(pos_x, pos_y), (width+pos_x,height+pos_y),
                      (randrange(256),randrange(256),randrange(256)),10)

    cv2.imshow('Eye Tracker', the_frame)
    cv2.waitKey(1)
