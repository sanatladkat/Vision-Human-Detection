
import cv2
import numpy as np
import os
import pathlib
import image_variable as imgvar

BASE_DIR = imgvar.BASE_DIR
FILE_NAME = imgvar.FILE_NAME
SAVE_DIR_PATH = imgvar.SAVE_DIR_PATH
FILE_PATH = imgvar.FILE_PATH


print(SAVE_DIR_PATH)


image = cv2.imread(FILE_PATH)
cv2.imshow("Original Image",image)

# Brightness
BETA = 100
increased_brightness_image = cv2.convertScaleAbs(image, beta=BETA)
cv2.imshow("Brightness",increased_brightness_image)
cv2.imwrite(SAVE_DIR_PATH+'brightness-'+ str(BETA)+'.jpg',increased_brightness_image )

# Contrast Adjustment
alpha = 7.5  # Increase this value to increase contrast
contrast_adjusted_image = cv2.convertScaleAbs(image, alpha=alpha)
cv2.imshow("Contrast Adjustment",contrast_adjusted_image)
cv2.imwrite(SAVE_DIR_PATH+'con-adj-'+ str(alpha)+'.jpg',contrast_adjusted_image )


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equalized_image = cv2.equalizeHist(gray_image)

cv2.imshow("Equalized Histogram",equalized_image)
cv2.imwrite(SAVE_DIR_PATH + 'eq-hist'+'.jpg',equalized_image )



H = 10
HColor = 10
denoised_image = cv2.fastNlMeansDenoisingColored(image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

cv2.imshow("Denoised",denoised_image)
cv2.imwrite(SAVE_DIR_PATH+f'denoised-{H}-{HColor}'+'.jpg',denoised_image)



gamma = 0.5  # Increase this value to make the image brighter
gamma_corrected_image = cv2.pow(image / 255.0, gamma)
gamma_corrected_image = (gamma_corrected_image * 255).astype('uint8')

cv2.imshow("Gamma Correction",gamma_corrected_image)
cv2.imwrite(SAVE_DIR_PATH+f'gamma-{gamma}'+'.jpg',gamma_corrected_image)



CL = 2.0
tileGridSize =(8,8)
gray_image = cv2.imread(FILE_PATH, 0)  # Load as grayscale
clahe = cv2.createCLAHE(clipLimit=CL, tileGridSize=tileGridSize)

# Apply CLAHE to the image
clahe_image = clahe.apply(gray_image)

cv2.imshow("CLAHE",clahe_image)
cv2.imwrite(SAVE_DIR_PATH+f'clahe-cl{CL}-{tileGridSize[0]}x{tileGridSize[1]}'+'.jpg',denoised_image)
cv2.waitKey(0) 
  
# closing all open windows 
cv2.destroyAllWindows() 
