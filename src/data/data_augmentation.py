import cv2 as cv
import os
import imutils
import numpy as np
from tqdm import tqdm

paths = ['../../data/4ary/test/']
#  '../../data/4ary/train/']


def get_save_path(image_path, modifier):
    name, extension = image_path.split('.tif')
    extension = '.tif'
    save_path = name + '_' + modifier + extension
    return save_path


def show_image(image, flipped):
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.namedWindow('flipped', cv.WINDOW_NORMAL)
    cv.imshow('image', image)
    cv.resizeWindow('flipped', 384, 384)
    cv.imshow('flipped', flipped)
    cv.waitKey(10000)
    cv.destroyAllWindows()


def rotate_image(image, image_path, number_of_rotations):
    if number_of_rotations == 0:
        return None

    for angle in np.linspace(0, 360, number_of_rotations):
        if angle == 0 or angle == 360:
            continue
        # image = cv.imread(image_path)
        rotated = imutils.rotate_bound(image, angle)
        save_path = get_save_path(image_path, str(angle))
        cv.imwrite(save_path, rotated)
        # print(save_path)


def horizontal_flip(image, image_path):
    # image = cv.imread(image_path)
    flipped = image[:, ::-1, :]
    save_path = get_save_path(image_path, 'flip_h)')
    #show_image(image, flipped)
    cv.imwrite(save_path, flipped)


def vertical_flip(image, image_path):
    # image = cv.imread(image_path)
    flipped = image[::-1, :, :]
    save_path = get_save_path(image_path, 'flip_v')
    #show_image(image, flipped)
    cv.imwrite(save_path, flipped)


def vertical_horizontal_flip(image, image_path):
    # image = cv.imread(image_path)
    flipped = image[::-1, ::-1, :]
    save_path = get_save_path(image_path, 'flip_vh')
    # show_image(image, flipped)
    cv.imwrite(save_path, flipped)


def data_augmentation(image_path, number_of_rotations):
    image = cv.imread(image_path)
    rotate_image(image, image_path, number_of_rotations)
    if number_of_rotations != 0:
        horizontal_flip(image, image_path)
        vertical_flip(image, image_path)
    vertical_horizontal_flip(image, image_path)


def process_folder(PATH):
    pid = PATH.split('/')[-1]
    rotations = {'0': 0, '1': 6, '2': 3, '3': 3}
    number_of_rotations = rotations.get(pid)
    for image in tqdm(os.listdir(PATH)):
        image_path = os.path.join(PATH, image)
        data_augmentation(image_path, number_of_rotations)
        # print(image_path, number_of_rotations)


def augment_dataset(PATH):
    for folder in os.listdir(PATH):
        # process_folder(os.path.join(PATH, folder))
        print(os.path.join(PATH, folder).split('/')[-1])


for path in paths:
    augment_dataset(path)
