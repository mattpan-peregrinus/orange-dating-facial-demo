# Loading the image and converting it to RGB format

import cv2 

def load_image(image_path):
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, rgb
    