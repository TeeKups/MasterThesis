import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import cv2
import imutils
import numpy
import pathlib
import logging
import time

logger = logging.getLogger('__main__')

cascade_front = cv2.CascadeClassifier(str(pathlib.Path('capturelib', 'haarcascade_frontalface_alt.xml')))
cascade_side = cv2.CascadeClassifier(str(pathlib.Path('capturelib', 'haarcascade_profileface.xml')))

def prep(img):
    img = imutils.resize(img, width=640)
    flipped = img # cv2.flip(img, 1)
    gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
    return gray, img


def detect_front(image, gray):
    faces = cascade_front.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in faces:
        #cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        roi_color = cv2.GaussianBlur(roi_color, (53,53), 30)
        image[y:y+roi_color.shape[0], x:x+roi_color.shape[1]] = roi_color

    return image
    
def detect_left(image, gray):
    faces = cascade_side.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in faces:
        #cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        roi_color = cv2.GaussianBlur(roi_color, (53,53), 30)
        image[y:y+roi_color.shape[0], x:x+roi_color.shape[1]] = roi_color

    return image

def detect_right(image, gray):
    gray = cv2.flip(gray, 1)
    image = cv2.flip(image, 1)
    faces = cascade_side.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in faces:
        #cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        roi_color = cv2.GaussianBlur(roi_color, (53,53), 30)
        image[y:y+roi_color.shape[0], x:x+roi_color.shape[1]] = roi_color

    return cv2.flip(image, 1)

def blur_faces(image):
    start = time.perf_counter()
    gray, image = prep(image)
    image = detect_front(image, gray)
    image = detect_left(image, gray)
    image = detect_right(image, gray)
    logger.debug(f'blurring a single image took {(time.perf_counter() - start):.3f} s')
    return image

 
if __name__ == '__main__':
    cam = cv2.VideoCapture(2)

    while True:
        ret_val, image = cam.read()
        image = blur_faces(image)
        cv2.imshow('Image', image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    cam.release

