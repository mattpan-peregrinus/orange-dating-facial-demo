import os
import cv2
import numpy as np
import mediapipe as mp
from utils.face_detect import num_of_faces
from utils.quality import check_blurriness, check_brightness
from utils.orientation import analyze_face_angle
from utils.image_loader import load_image
import json
from datetime import datetime

class ParameterTuner:
    def __init__(self, test_images_dir):
        self.test_images_dir = test_images_dir
        self.results = {}
        
    def test_blurriness_parameters(self, image, blur_thresholds):
        """Test different blurriness thresholds"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        results = {}
        for threshold in blur_thresholds:
            status = "Sharp" if variance >= threshold else "Blurry"
            results[threshold] = {
                "variance": variance,
                "status": status
            }
        return results
    
    def test_brightness_parameters(self, image, dark_thresholds, bright_thresholds):
        """Test different brightness thresholds"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brightness = hsv[:, :, 2].mean()
        
        results = {}
        for dark_thresh, bright_thresh in zip(dark_thresholds, bright_thresholds):
            key = f"dark_{dark_thresh}_bright_{bright_thresh}"
            if brightness < dark_thresh:
                status = "Too Dark"
            elif brightness > bright_thresh:
                status = "Too Bright"
            else:
                status = "Good"
            results[key] = {
                "brightness": brightness,
                "status": status
            }
        return results
    
    def test_orientation_parameters(self, image, angle_thresholds):
        """Test different face orientation thresholds"""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_face_mesh = mp.solutions.face_mesh
        
        results = {}
        with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
            face_results = face_mesh.process(rgb)
            
            if not face_results.multi_face_landmarks:
                for threshold in angle_thresholds:
                    results[threshold] = {
                        "status": "No face detected",
                        "angle": None
                    }
                return results
            
            landmarks = face_results.multi_face_landmarks[0].landmark
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            
            dx = right_eye.x - left_eye.x
            dy = right_eye.y - left_eye.y
            angle = cv2.fastAtan2(dy, dx)
            deviation = min(abs(angle), abs(angle - 180), abs(angle - 360))
            
            for threshold in angle_thresholds:
                status = "Well aligned" if deviation <= threshold else "Face turned"
                results[threshold] = {
                    "status": status,
                    "angle": angle,
                    "deviation": deviation
                }
        return results
    
    def run_tests(self, blur_thresholds=None, dark_thresholds=None, 
                 bright_thresholds=None, angle_thresholds=None):
        """Run tests with different parameter combinations"""
        if blur_thresholds is None:
            blur_thresholds = [50, 75, 100, 125, 150]
        if dark_thresholds is None:
            dark_thresholds = [30, 40, 50, 60, 70]
        if bright_thresholds is None:
            bright_thresholds = [180, 190, 200, 210, 220]
        if angle_thresholds is None:
            angle_thresholds = [10, 15, 20, 25, 30]
            
        for filename in os.listdir(self.test_images_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            image_path = os.path.join(self.test_images_dir, filename)
            image, rgb = load_image(image_path)
            
            self.results[filename] = {
                "blurriness": self.test_blurriness_parameters(image, blur_thresholds),
                "brightness": self.test_brightness_parameters(image, dark_thresholds, bright_thresholds),
                "orientation": self.test_orientation_parameters(image, angle_thresholds)
            }
    
    def save_results(self, output_file=None):
        """Save test results to a JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"parameter_tuning_results_{timestamp}.json"
            
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {output_file}")
        
    def print_summary(self):
        """Print a summary of the test results"""
        print("\nParameter Tuning Summary:")
        print("=" * 50)
        
        # Analyze blurriness results
        print("\nBlurriness Analysis:")
        blur_counts = {}
        for image_results in self.results.values():
            for threshold, result in image_results["blurriness"].items():
                if threshold not in blur_counts:
                    blur_counts[threshold] = {"Sharp": 0, "Blurry": 0}
                blur_counts[threshold][result["status"]] += 1
        
        for threshold, counts in blur_counts.items():
            print(f"\nThreshold {threshold}:")
            print(f"  Sharp: {counts['Sharp']} images")
            print(f"  Blurry: {counts['Blurry']} images")
        
        # Analyze brightness results
        print("\nBrightness Analysis:")
        brightness_counts = {}
        for image_results in self.results.values():
            for params, result in image_results["brightness"].items():
                if params not in brightness_counts:
                    brightness_counts[params] = {"Too Dark": 0, "Good": 0, "Too Bright": 0}
                brightness_counts[params][result["status"]] += 1
        
        for params, counts in brightness_counts.items():
            print(f"\nParameters {params}:")
            print(f"  Too Dark: {counts['Too Dark']} images")
            print(f"  Good: {counts['Good']} images")
            print(f"  Too Bright: {counts['Too Bright']} images")
        
        # Analyze orientation results
        print("\nOrientation Analysis:")
        orientation_counts = {}
        for image_results in self.results.values():
            for threshold, result in image_results["orientation"].items():
                if threshold not in orientation_counts:
                    orientation_counts[threshold] = {"Well aligned": 0, "Face turned": 0, "No face detected": 0}
                orientation_counts[threshold][result["status"]] += 1
        
        for threshold, counts in orientation_counts.items():
            print(f"\nThreshold {threshold}:")
            print(f"  Well aligned: {counts['Well aligned']} images")
            print(f"  Face turned: {counts['Face turned']} images")
            print(f"  No face detected: {counts['No face detected']} images")

def main():
    test_images_dir = "test_images"
    if not os.path.exists(test_images_dir):
        os.makedirs(test_images_dir)
        print(f"Created directory {test_images_dir}. Please add some test images there.")
        return
    
    # Initialize the parameter tuner
    tuner = ParameterTuner(test_images_dir)
    
    # Define parameter ranges to test 
    blur_thresholds = [75, 100, 125, 150, 175] 
    dark_thresholds = [40, 50, 60, 70, 80]     
    bright_thresholds = [160, 170, 180, 190, 200]  
    angle_thresholds = [5, 10, 15, 20, 25]     
    
    # Run the tests
    print("Running parameter tuning tests...")
    tuner.run_tests(
        blur_thresholds=blur_thresholds,
        dark_thresholds=dark_thresholds,
        bright_thresholds=bright_thresholds,
        angle_thresholds=angle_thresholds
    )
    
    # Save and print results
    tuner.save_results()
    tuner.print_summary()

if __name__ == "__main__":
    main() 