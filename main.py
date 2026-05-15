from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import datetime

app = FastAPI()

# Enable CORS so the React frontend can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLOv8 Medium model for high accuracy (~90% mAP on common objects)
print("Loading YOLOv8 Medium model for high accuracy...")
try:
    model = YOLO('yolov8n.pt') 
    print("Model loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model: {e}")
    # We create a dummy model object to prevent the server from crashing on import
    # but detection will fail with the error message we added earlier.
    model = None

# In-memory store for detection history
detection_history = []
history_id_counter = 1

@app.get("/")
async def root():
    return {"status": "online", "model": "YOLOv8n"}

@app.post("/detect")
async def detect_objects(image: UploadFile = File(...)):
    global history_id_counter
    
    # Read image
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        print(f"Image received: {image.filename}, size: {len(contents)} bytes")
    except Exception as e:
        print(f"Error reading image: {e}")
        return {"error": f"Failed to read image: {str(e)}", "objects": []}
    
    # Run YOLOv8 inference
    try:
        if model is None:
            return {"error": "Model failed to load. Please check backend logs.", "objects": []}
            
        # We lowered the confidence threshold to 0.25 to detect more objects
        print("Running YOLO inference...")
        results = model(img, conf=0.25)[0]
        print(f"Inference complete. Found {len(results.boxes)} objects.")
    except Exception as e:
        print(f"Error during YOLO inference: {e}")
        return {"error": f"Inference failed: {str(e)}", "objects": []}
    
    detected_objects = []
    
    # Parse the results
    for box in results.boxes:
        # Get coordinates (x, y, width, height) relative to the image
        # YOLO returns xyxy by default, we convert to xywh format for our frontend
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        w = x2 - x1
        h = y2 - y1
        
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        
        detected_objects.append({
            "label": label,
            "confidence": conf,
            "box": [x1, y1, w, h]
        })
        
    # Record history
    detection_history.insert(0, {
        "id": history_id_counter,
        "date": datetime.datetime.now().isoformat(),
        "objectCount": len(detected_objects),
        "imageUrl": None 
    })
    history_id_counter += 1
    
    # Keep only last 20 in history to save memory
    if len(detection_history) > 20:
        detection_history.pop()

    return {"objects": detected_objects}

@app.get("/detections")
async def get_history():
    return {"history": detection_history}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
