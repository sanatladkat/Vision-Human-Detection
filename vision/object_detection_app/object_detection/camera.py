
import cv2
import threading
import urllib.request
import numpy as np
import tensorflow as tf
from .cnn_model import Model
m1 = Model()

class VideoCamera(object):

    def __init__(self) :
        self.width , self.height = 512,512

        self.video = cv2.VideoCapture(0)
        self.grabbed , self.frame = self.video.read()
        threading.Thread(target=self.update , args = ()).start()


    def __del__(self) :
        self.video.release()

    def get_frame(self) :
        img = self.frame

        # * for testing only
        img = cv2.resize(img, (self.width , self.height ))


        # apply image processing here
        feed_img = self.image_processing(img)

        # convert image to tensors
        img_tensor = m1.image_to_array(feed_img)

        boxes, scores, classes, num_detections = m1.model(img_tensor)

        pred_labels = classes.numpy().astype('int')[0]

        pred_labels = [m1.labels[i] for i in pred_labels]
        pred_boxes = boxes.numpy()[0].astype('int')
        pred_scores = scores.numpy()[0]
        #loop throughout the faces detected and place a box around it
        
        for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
            if score < 0.5:
                continue
                
            score_txt = f'{100 * round(score,0)}'
            img_boxes = cv2.rectangle(img,(xmin, ymax),(xmax, ymin),(0,255,0),1)      
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_boxes,label,(xmin, ymax-10), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(img_boxes,score_txt,(xmax, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)

            display_frame = img_boxes
            print(1)
        else :
            display_frame = img
            print(0)

        _, jpeg = cv2.imencode('.jpg' ,display_frame)


        return jpeg.tobytes()

    def update(self) :
        while True :
            (self.grabbed , self.frame) = self.video.read()

    def image_processing(self, img) : 

        # gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

        # resize = cv2.resize(gray, (640,480), interpolation=cv2.INTER_LINEAR)
        # frame_flip = cv2.flip(resize , 1)

        # return frame_flip
    
        #Resize to respect the input_shape
        inp = cv2.resize(img, (self.width , self.height ))

        #Convert img to RGB
        rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

        return rgb


class IpCamera :

    def __init__(self) :
        self.url = "http://192.168.1.2:8080/shot.jpg"

    def __del__(self) :
        cv2.destroyAllWindows()

    def get_frame(self) :

        imgResp = urllib.request.urlopen(self.url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)

        img = cv2.imdecode(imgNp , -1)

        gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

        resize = cv2.resize(gray, (640,480), interpolation=cv2.INTER_LINEAR)
        frame_flip = cv2.flip(resize , 1)
        ret, jpeg = cv2.imencode(".jpg" , frame_flip)

        return jpeg.tobytes()

