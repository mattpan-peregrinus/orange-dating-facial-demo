# Check how many faces are in the image

import mediapipe as mp

def num_of_faces(image):
    face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    results = face_detection.process(image)
    if results.detections:
        print(f"Number of faces detected: {len(results.detections)}. We only want one face.")
    else:
        print("No faces detected.")