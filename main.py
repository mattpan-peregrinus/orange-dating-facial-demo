# Run to test utils functions

from utils.image_loader import load_image
from utils.face_detect import num_of_faces
from utils.quality import check_blurriness, check_brightness
from utils.orientation import analyze_face_angle

if __name__ == "__main__":
    image_path = 'path/to/your/image.jpg'  
    image, rgb = load_image(image_path)
    num_of_faces(rgb)
    check_blurriness(image)
    check_brightness(image)
    analyze_face_angle(image)