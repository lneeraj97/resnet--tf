# Sort the data from various folders in messidor dataset to 4 folders based on the label

import pandas as pd
import os
import shutil
from tqdm import tqdm

RAW_PATH = './data/raw'
PATH = './data'
DEST_PATH = './data/interim'


def sort_images(folder_path):
    # Read the excel sheet for image names and their labels
    file_path = folder_path+'index.xls'
    df = pd.read_excel(file_path, sheet_name=0, usecols=[0, 2])
    # Renaming the columns
    df.columns = ['image', 'label']
    # Iterating through every image listed in the excel sheet and sorting it based on the label
    for index, series in df.iterrows():
        # iterrows returns the row number and a pd.series containing the columns
        image_name, label = series
        image_path = folder_path + image_name
        dest_path = DEST_PATH + '/' + str(label) + '/'
        shutil.copy2(image_path, dest_path)


for item in tqdm(os.listdir(RAW_PATH)):
    # Iteraring through every folder and then sorting the images in that folder
    folder_path = RAW_PATH + '/' + item + '/'
    sort_images(folder_path)
print("All the images have been sorted successfully")
