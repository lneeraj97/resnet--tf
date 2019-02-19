import cv2 as cv
import os
import sys
import numpy as np
from tqdm import tqdm

DESTINATION = 'processed'
paths = ['../../data/augmented/4ary/test/', '../../data/augmented/4ary/train/']


def show_image(image=None, processed=None):
    if image is not None:
        cv.namedWindow('Original Image', cv.WINDOW_NORMAL)
        cv.imshow('Original Image', image)
    if processed is not None:
        cv.namedWindow('Processed', cv.WINDOW_NORMAL)
        cv.imshow('Processed', processed)
    cv.waitKey(200)
    cv.destroyAllWindows()


def preprocess_image(image_path):
    color_image = cv.imread(image_path, 1)
    bw_image = cv.imread(image_path, 0)
    ret, mask = cv.threshold(bw_image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    image = color_image[:, :, 1]
    image = cv.medianBlur(image, 5)
    clahe = cv.createCLAHE(clipLimit=10.0, tileGridSize=(5, 5))
    image = clahe.apply(image)
    final_image = cv.bitwise_and(image, image, mask=mask)
    save_path = image_path.replace('augmented', DESTINATION)
    cv.imwrite(save_path, image)
    # show_image(None, final_image)


def process_folder(PATH):
    for image in tqdm(os.listdir(PATH)):
        image_path = os.path.join(PATH, image)
        preprocess_image(image_path)
        # print(image_path)


def start_preprocessing(PATH):
    # Call the process_folder function on each folder
    for folder in os.listdir(PATH):
        folder_path = os.path.join(PATH, folder)
        print(folder_path)
        process_folder(folder_path)


for path in paths:
    start_preprocessing(path)
