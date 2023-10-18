import cv2
import os

import numpy as np  
  
cap = cv2.VideoCapture(0)  
  

def image_processing(frame) :

    # gamma = 0.5
    # frame = cv2.pow(frame / 255.0, gamma)
    # frame = (frame * 255).astype('uint8')

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # equalized_image = cv2.equalizeHist(frame)
    frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply CLAHE to the image
    frame = clahe.apply(frame)


    return frame

while(True):  
    # Capture image frame-by-frame  
    ret, frame = cap.read()  
  
    # Our operations on the frame come here  
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  

    frame = image_processing(frame)
  
    # Display the resulting frame  
    cv2.imshow('frame',frame)  
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  
  
# When everything done, release the capture  
cap.release()  
cv2.destroyAllWindows()  