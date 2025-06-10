# Check blurriness and brightness of the image

import cv2 

def check_blurriness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < 100:  # Threshold for blurriness
        print("The image is blurry.")
        return "Blurry"
    else:
        print("The image is not blurry.")
        return "Sharp"

def check_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2].mean()
    if brightness < 50:  # Threshold for darkness
        print("The image is too dark.")
        return "Too Dark"
    elif brightness > 200:  # Threshold for brightness
        print("The image is too bright.")
        return "Too Bright"
    else:
        print("The image has good brightness.")
        return "Good"