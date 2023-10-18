


import os
import pathlib


BASE_DIR = pathlib.Path(__file__).resolve().parent


FILE_NAME = r'garage5.png'


SAVE_DIR_PATH = os.path.join(BASE_DIR ,'images','temp', FILE_NAME.split('.')[0])
FILE_PATH = os.path.join(BASE_DIR,'images' , FILE_NAME)