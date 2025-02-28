import time
import cv2
import mediapipe as mp
import numpy as np
import base64
import threading
import os
from datetime import datetime
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tracker import *

# Manually set Google API Key (Replace with your actual key)
GOOGLE_API_KEY = "AIzaSyAHR9ITHQbFrKo4DSOK56H-P5eUFKXfsfQ"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Load the model
model_path = "efficientdet_lite0.tflite"
video_path = "tvid.mp4"  # Change this to your video file

# Initialize Tracker
tracker = Tracker()

# Get current date for folder naming
current_date = datetime.now().strftime("%Y-%m-%d")
crop_folder = f"crop_{current_date}"
os.makedirs(crop_folder, exist_ok=True)

# Create response file
response_file = f"responses_{current_date}.txt"

# Open video file
cap = cv2.VideoCapture(video_path)

# Define MediaPipe options
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    max_results=5,
    running_mode=VisionRunningMode.IMAGE,
    score_threshold=0.5
)

# Dictionary to store processed IDs to avoid duplicates
processed_ids = set()

def encode_image_to_base64(image):
    _, img_buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(img_buffer).decode("utf-8")

def analyze_image_with_gemini(image_path, obj_id):
    try:
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")

        message = HumanMessage(
            content=[
                {"type": "text", "text": f"Analyze this image and extract only the following details:\n\n"
                 "|Vehicle Type(Name of Vehicle) | Vehicle Color | Vehicle Company |\n"
                 "|--------------|--------------|---------------|"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, "description": "Detected vehicle"}
            ]
        )

        response = gemini_model.invoke([message])
        result = response.content.strip()
        
        # Save result to file with ID and date
        with open(response_file, "a") as file:
            file.write(f"Date: {current_date}\nID: {obj_id}\nImage: {image_path}\nResponse: {result}\n\n")
        
        return result
    except Exception as e:
        print(f"Error invoking Gemini model: {e}")
        return "Error processing image."

def process_and_save_object(frame, bbox, obj_id):
    x1, y1, x2, y2 = bbox

    # Check if object ID has already been processed
    if obj_id in processed_ids:
        return

    # Save new object ID
    processed_ids.add(obj_id)

    # Crop and save image
    cropped_object = frame[y1:y2, x1:x2]
    img_path = os.path.join(crop_folder, f"object_{obj_id}.jpg")
    cv2.imwrite(img_path, cropped_object)

    # Process image with Gemini in a separate thread
    threading.Thread(target=analyze_image_with_gemini, args=(img_path, obj_id)).start()

count = 0
cy1 = 238
offset = 10

# Initialize object detector
with ObjectDetector.create_from_options(options) as detector:
    while cap.isOpened():
        success, frame = cap.read()
        count += 1

        # Skip frames to improve performance
        if count % 3 != 0:
            continue

        if not success:
            break  # Stop if the video ends
        
        # Resize frame for consistency
        frame = cv2.resize(frame, (640, 480))  
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Run object detection
        start_time = time.time()
        detection_result = detector.detect(mp_image)
        print(f"Frame Processed in {time.time() - start_time:.2f} sec")
        
        detected_objects = []
        for data in detection_result.detections:
            bbox = data.bounding_box
            category = data.categories[0]

            # Bounding box coordinates
            x1, y1, x2, y2 = bbox.origin_x, bbox.origin_y, bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            detected_objects.append([x1, y1, x2, y2])

        bbox_idx = tracker.update(detected_objects)
        
        for bbox, category in zip(bbox_idx, detection_result.detections):
            x3, y3, x4, y4, obj_id = bbox
            class_name = category.categories[0].category_name

            cx = (x3 + x4) // 2
            cy = (y3 + y4) // 2

            if cy1 - offset < cy < cy1 + offset:
                # Process and save object only if not already saved
                process_and_save_object(frame, (x3, y3, x4, y4), obj_id)

                # Draw center point
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                # Draw the rectangle
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 3)
                # Put the object ID on top-left corner
                cv2.putText(frame, str(obj_id), (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.line(frame, (0, 238), (628, 238), (0, 0, 255), 1)

        # Display the frame with detections
        cv2.imshow("Object Detection (Video)", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
