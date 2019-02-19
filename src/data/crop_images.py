import cv2 as cv
import os
import sys
import numpy as np
from tqdm import tqdm
IMAGE = '../../data/messidor/test/2/20060412_61790_0200_PP.tif'
paths = ['../../data/messidor/test/', '../../data/messidor/train/']


def show_image(image=None, processed=None):
    if image is not None:
        cv.namedWindow('Original Image', cv.WINDOW_NORMAL)
        cv.imshow('Original Image', image)
    if processed is not None:
        cv.namedWindow('Processed', cv.WINDOW_NORMAL)
        cv.imshow('Processed', processed)
    cv.waitKey(200)
    cv.destroyAllWindows()


def crop_image(image_path):
    image = cv.imread(image_path, 1)
    shape = image.shape
    if shape == (960, 1440, 3):
        cropped = image[50:-50, 250:-250, :]
    else:
        cropped = image[50:-50, 400:-400, :]
    cv.imwrite(image_path, cropped)


def process_folder(PATH):
    for image in tqdm(os.listdir(PATH)):
        image_path = os.path.join(PATH, image)
        crop_image(image_path)
        # print(image_path)


def start_preprocessing(PATH):
    # Call the process_folder function on each folder
    for folder in os.listdir(PATH):
        folder_path = os.path.join(PATH, folder)
        print(folder_path)
        process_folder(folder_path)


for path in paths:
    start_preprocessing(path)
# crop_image(IMAGE)
