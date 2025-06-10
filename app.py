from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from PIL import Image
import os

from utils.face_detect import num_of_faces
from utils.quality import check_blurriness, check_brightness
from utils.orientation import analyze_face_angle

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

def analyze_image(image):

    results = {
        'face_count': None,
        'blurriness': None,
        'brightness': None,
        'orientation': None,
        'overall_score': 0,
        'recommendations': []
    }
    
    try:
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Face count check
        face_count = num_of_faces(rgb_image)
        results['face_count'] = face_count
        if face_count == 1:
            results['overall_score'] += 25
        elif face_count == 0:
            results['recommendations'].append("No face detected. Make sure your face is clearly visible.")
        else:
            results['recommendations'].append(f"{face_count} faces detected. Use a photo with just yourself.")
        
        # Blurriness check 
        blur_status = check_blurriness(cv_image)
        results['blurriness'] = blur_status
        if blur_status == "Sharp":
            results['overall_score'] += 25
        else:
            results['recommendations'].append("Image is blurry. Try holding your phone steadier or using better lighting.")
        
        # Brightness check 
        brightness_status = check_brightness(cv_image)
        results['brightness'] = brightness_status
        if brightness_status == 'Good':
            results['overall_score'] += 25
        elif brightness_status == 'Too Dark':
            results['recommendations'].append("Image is too dark. Try taking the photo in better lighting.")
        else:
            results['recommendations'].append("Image is too bright. Avoid direct harsh lighting or flash.")
        
        # Orientation check 
        if face_count == 1:
            orientation_status = analyze_face_angle(cv_image)
            results['orientation'] = orientation_status
            if orientation_status == 'Well aligned':
                results['overall_score'] += 25
            else:
                results['recommendations'].append("Face is turned away. Try facing the camera more directly.")
        
        # Add feedback
        if results['overall_score'] >= 75:
            results['recommendations'].insert(0, "Great photo! This would make an excellent profile picture.")
        elif results['overall_score'] >= 50:
            results['recommendations'].insert(0, "Good photo with some room for improvement.")
        else:
            results['recommendations'].insert(0, "This photo needs some work before uploading.")
            
    except Exception as e:
        results['error'] = str(e)
    
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'})
    
    try:
        image = Image.open(file.stream)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        results = analyze_image(image)
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'})

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    app.run(debug=True, host='0.0.0.0', port=8000)