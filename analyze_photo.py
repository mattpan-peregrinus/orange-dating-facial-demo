import cv2
import mediapipe as mp

image_path = "image_path.jpg"
image = cv2.imread(image_path)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def how_many_faces(image):
    face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    results = face_detection.process(rgb)
    if results.detections:
        print(f"Number of faces detected: {len(results.detections)}")
    else:
        print("No faces detected.")

def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < 100:  # Threshold for blurriness
        print("The image is blurry.")
    else:
        print("The image is not blurry.")

def check_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2].mean()
    if brightness < 50:  # Threshold for darkness
        print("The image is too dark.")
    elif brightness > 200:  # Threshold for brightness
        print("The image is too bright.")
    else:
        print("The image has good brightness.")
        
    
if __name__ == "__main__":
    how_many_faces(image)
    is_blurry(image)
    check_brightness(image)
    
    

