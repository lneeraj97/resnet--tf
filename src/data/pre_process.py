import cv2 as cv
import os
import sys
import numpy as np
from tqdm import tqdm

TRAIN = '../data/2ary/train'
TEST = '../data/2ary/test'
PROCESSED = 'processed'
# dr3 = 'data/interim/validate/3/20051019_38557_0100_PP_180.0.tif'
# dr0 = 'data/interim/validate/0/20051020_43808_0100_PP_vhflip.tif'


def show_image(image, processed):
    cv.namedWindow('Original Image', cv.WINDOW_NORMAL)
    cv.imshow('Original Image', image)
    cv.namedWindow('Processed', cv.WINDOW_NORMAL)
    cv.imshow('Processed', processed)
    cv.waitKey(4000)
    cv.destroyAllWindows()


def preprocess_image(image_path):
    # Open the image
    original_image = cv.imread(image_path, 1)
    # Resize image to 256x256
    original_image = cv.resize(original_image, (512, 512))
    # Extract green channel from the image
    # NOTE: OPENCV USES BGR COLOR ORDER
    image = original_image[:, :, 1]
    # Apply median blur to remove salt and pepper noise
    image = cv.medianBlur(image, 3)

    # Apply CLAHE thresholding
    clahe = cv.createCLAHE(clipLimit=10.0, tileGridSize=(4, 4))
    image = clahe.apply(image)
    # Get save path
    # save_path = image_path.replace('interim', PROCESSED)
    cv.imwrite(image_path, image)
    # Show the image
    # show_image(original_image, image)


def process_folder(PATH):
    for image in tqdm(os.listdir(PATH)):
        image_path = os.path.join(PATH, image)
        preprocess_image(image_path)


def start_preprocessing(PATH):
    # Call the process_folder function on each folder
    for folder in os.listdir(PATH):
        folder_path = os.path.join(PATH, folder)
        print(folder_path)
        process_folder(folder_path)


start_preprocessing(TRAIN)
start_preprocessing(TEST)
# preprocess_image(dr3)
# preprocess_image(dr0)
