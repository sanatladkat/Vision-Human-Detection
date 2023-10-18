
import cv2
import numpy as np
import os
import image_variable as imgvar



# BASE_DIR = os.getcwd()
# FILE_NAME = r'garage5.png'
# FILE_PATH = os.path.join(BASE_DIR,'image_processing_test1','images' , FILE_NAME)


print(imgvar.FILE_PATH)


image = cv2.imread(imgvar.FILE_PATH)
cv2.imshow("Original Image",image)
# cv2.imwrite()


# gamma = 0.5  # Increase this value to make the image brighter
# gamma_corrected_image = cv2.pow(image / 255.0, gamma)
# gamma_corrected_image = (gamma_corrected_image * 255).astype('uint8')


gammalist = np.arange(start=0.3 , stop=0.6 , step=0.05)

for gamma in gammalist : 
    gamma = round(gamma,2)
    gamma_corrected_image = cv2.pow(image / 255.0, gamma)
    gamma_corrected_image = (gamma_corrected_image * 255).astype('uint8')
    cv2.imshow("Gamma Correction " + str(gamma),gamma_corrected_image)
    cv2.imwrite(imgvar.SAVE_DIR_PATH+f'gamma-{gamma}'+'.jpg',gamma_corrected_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
