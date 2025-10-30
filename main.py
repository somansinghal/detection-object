
from ultralytics import YOLO
import cv2
import numpy as np

# --- Configuration ---
MODEL_PATH = "yolov8n.pt" 

# --- 1. TARGET CLASSES (Objects) ---
TARGET_CLASSES = [
    "person", "chair", "dining table", "tv", "laptop", "mouse", "remote", 
    "keyboard", "cell phone", "book", "clock", "cat", "dog", "bird", 
    "banana", "apple", "orange", "broccoli", "carrot", "bottle", 
    "cup", "bowl", "fork", "knife", "spoon", 
]

# --- 2. COLOR MAPPING (for Bounding Box and Text) ---
COLOR_MAP = {
    "person": (0, 255, 0), "chair": (0, 0, 255), "dining table": (0, 255, 255),
    "tv": (100, 100, 100), "laptop": (150, 150, 150), "mouse": (255, 10, 10),
    "remote": (200, 0, 0), "keyboard": (128, 128, 128), "cell phone": (255, 0, 0),
    "book": (10, 10, 100), "clock": (255, 0, 255), "cat": (255, 165, 0),
    "dog": (128, 0, 128), "bird": (255, 255, 0), "banana": (0, 255, 255),
    "apple": (0, 0, 128), "orange": (0, 165, 255), "broccoli": (0, 100, 0),
    "carrot": (0, 140, 255), "bottle": (255, 192, 203), "cup": (165, 42, 42),
    "bowl": (0, 128, 0), "fork": (192, 192, 192), "knife": (128, 0, 0),
    "spoon": (0, 0, 100),
}

# --- 3. CUSTOM COLOR DEFINITIONS (15 Colors in BGR) ---
# Used for the secondary color detection
COLORS_BGR = {
    "Red": (0, 0, 255),           "Green": (0, 255, 0),         "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),      "Cyan": (255, 255, 0),        "Magenta": (255, 0, 255),
    "White": (255, 255, 255),     "Black": (0, 0, 0),           "Gray": (128, 128, 128),
    "Orange": (0, 165, 255),      "Pink": (255, 192, 203),      "Brown": (42, 42, 165),
    "Violet": (148, 0, 211),      "Lime": (50, 205, 50),        "Teal": (128, 128, 0),
}


# --- CUSTOM DETECTION FUNCTIONS ---

def detect_color(roi, colors_bgr):
    """Detects the dominant color in the Region of Interest (ROI)."""
    if roi.size == 0:
        return "Unknown Color"
        
    # Calculate the average BGR value of the ROI
    avg_bgr = np.mean(roi, axis=(0, 1)).astype(int)
    
    # Calculate the distance to each defined color
    min_distance = float('inf')
    closest_color = "Unknown Color"
    
    for name, bgr_value in colors_bgr.items():
        # Euclidean distance in BGR space
        distance = np.sqrt(np.sum((np.array(bgr_value) - avg_bgr) ** 2))
        
        if distance < min_distance:
            min_distance = distance
            closest_color = name
            
    # Simple threshold: if the closest color is too far (e.g., for complex patterns/shades)
    if min_distance > 100: # Tune this threshold based on lighting and needs
        return "Mixed/Patterned"
        
    return closest_color


#Initialization
try:
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam. Check camera connection or index.")
except Exception as e:
    print(f"Error during initialization: {e}")
    exit()

#Main Detection Loop
print(f"Starting detection with {MODEL_PATH}. Press ESC to exit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    #Perform detection
    results = model(frame, stream=True, verbose=False) 

    #]Process results and draw bounding boxes
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if label in TARGET_CLASSES:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Ensure box coordinates are valid and within frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                # Check for a valid ROI size
                if x2 > x1 and y2 > y1:
                    # Extract Region of Interest (ROI)
                    roi = frame[y1:y2, x1:x2]
                    
                    # --- Custom Detection ---
                    detected_color = detect_color(roi, COLORS_BGR)
                    
                    # Combine YOLO class, Color, and Shape
                    main_color = COLOR_MAP.get(label, (255, 0, 255))
                    text_label = f"{label} ({detected_color}) {conf:.2f}"
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), main_color, 2)
                    
                    # Draw label background and text
                    (w, h), _ = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), main_color, -1)
                    cv2.putText(frame, text_label, (x1, y1 - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2) 

    # 3. Show frame
    cv2.imshow("Object, Color, and Shape Detection Stream", frame)

    # 4. Exit on ESC key press
    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()