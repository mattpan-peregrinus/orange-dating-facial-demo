# Check whether the face is well aligned with the camera

import cv2
import mediapipe as mp

def analyze_face_angle(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            print("No faces detected for angle analysis.")
            return "No face detected"
        
        landmarks = results.multi_face_landmarks[0].landmark
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        dx = right_eye.x - left_eye.x
        dy = right_eye.y - left_eye.y
        angle = cv2.fastAtan2(dy, dx)
        print(f"Estimated face yaw angle: {angle:.2f} degrees")
        deviation_from_horizontal = min(abs(angle), abs(angle - 180), abs(angle - 360))

        if deviation_from_horizontal > 15: 
            print("Face is turned. Try facing the camera more directly.")
            return "Face turned"
        else:
            print("Face is well aligned.")
            return "Well aligned"