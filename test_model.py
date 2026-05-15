import cv2
from ultralytics import YOLO
import numpy as np

def test_detection():
    print("Loading model...")
    try:
        model = YOLO('yolov8m.pt')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create a dummy image (black image with a white rectangle)
    # This might not be detected as an 'object', so let's try to create something more recognizable
    # or just check if the model can run without errors.
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)
    
    print("Running inference on dummy image...")
    try:
        results = model(img, conf=0.1)
        print(f"Inference complete. Detected {len(results[0].boxes)} objects.")
        for i, box in enumerate(results[0].boxes):
            print(f"Object {i}: {model.names[int(box.cls[0])]} with confidence {float(box.conf[0])}")
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    test_detection()
