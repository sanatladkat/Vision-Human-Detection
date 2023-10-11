import tensorflow_hub as hub
import numpy
import tensorflow as tf
import pandas as pd
from pathlib import Path
import os



# * modify paths here
APP_PATH = Path(__file__).resolve().parent
MODEL_PATH = os.path.join(APP_PATH, "models","efficientdet_lite2_detection_1")
LABEL_PATH = os.path.join(APP_PATH, "labels.csv")



class Model() :
    def __init__(self) -> None:
        

        self.model = tf.saved_model.load(MODEL_PATH)

        self.labels = pd.read_csv(LABEL_PATH,sep=';',index_col='ID')
        self.labels = self.labels['OBJECT (2017 REL.)']

    def image_to_array(self ,img) :

        #Is optional but i recommend (float convertion and convert img to tensor image)
        rgb_tensor = tf.convert_to_tensor(img, dtype=tf.uint8)

        #Add dims to rgb_tensor
        rgb_tensor = tf.expand_dims(rgb_tensor , 0)

        return rgb_tensor
    

if __name__ == "__main__" : 
    print(MODEL_PATH)


