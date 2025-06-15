# Profile Picture Analyzer 
Web application demo that analyzes profile pictures and provides actionable insights.

## Features
- **Face Detection**: Ensures exactly one person is in the photo.
- **Blur Analysis**: Detects image sharpness.
- **Lighting Assessment**: Evaluates brightness levels for optimal visibility.
- **Face Orientation**: Checks if the person is facing the camera directly.
- **Actionable Recommendations**: Specific tips to improve photo quality.

## Tech Stack
**Backend**: Flask, OpenCV, MediaPipe

**Frontend**: Vanilla JS, HTML5, CSS3

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/mattpan-peregrinus/orange-dating-facial-demo.git
2. Install required packages:
   ```bash
   pip install -r requirements.txt
3. Run CLI tool
   ```bash
   python main.py
4. Or run application
   ```bash
   python app.py
   ```

   By navigating to ```http://localhost:8000```

## Parameter Tuning
The `parameter_tuning.py` script helps optimize detection thresholds:

1. Add test images to `test_images/` directory
2. Run:
   ```bash
   python parameter_tuning.py
   ```
3. Script tests multiple thresholds:
   - Blurriness: Laplacian variance [75-175]
   - Brightness: HSV values [40-80] dark, [160-200] bright
   - Face angle: Deviation from horizontal [5-25] degrees
4. Results saved to JSON with timestamp
5. Console output shows distribution of results per threshold

## Future Features
- Smile detection
- Background quality
- Photo composition (ex: rule of thirds)
  




   
   
