'''
This module is used for managing the object detection modules by the developers.
It can be deleted in and does not play any role while running the django server.
'''

import pandas as pd
import pathlib
import os

from collections import defaultdict

BASE_DIR = pathlib.Path(__file__).resolve().parent
COCO_LABELS_PATH = os.path.join(BASE_DIR,'labels.csv')

def update_coco_labels(coco_class_names , default_value = "--nan--") : 
    '''
    Function to return a defaultdict containing all the coco dataset labels we are interested in.
    The return variable should be assigned manually to COCO_CLASS_NAMES variable in the objectDetectionModel.py.

    param 1 : List of coco labels to include from the labels.csv. Please refer the file 'labels.csv'. External values are not permitted
    '''

    ccn = defaultdict(lambda : default_value)
    df = pd.read_csv(COCO_LABELS_PATH, sep=";") 
    
    for id in df.index :
        if df.iloc[id]['OBJECT (2017 REL.)'] in coco_class_names :
            ccn[id + 1] = df.iloc[id]['OBJECT (2017 REL.)']

    return ccn


if __name__ == "__main__" :

    print(update_coco_labels(coco_class_names = [
        "person", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "handbag",
        "fork", "knife"
    ]))