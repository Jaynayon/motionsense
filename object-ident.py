import cv2
import numpy as np
import time
import tkinter as tk
from PIL import Image, ImageTk

#thres = 0.45 # Threshold to detect object

# Load object detection model
classNames = []
with open("coco.names", "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, thres, nms, classNames):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    objectInfo = []

    max_confidence = 0
    best_object = None

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if confidence > max_confidence:
                max_confidence = confidence
                best_object = (box, className, confidence)

    # Append the best object found to objectInfo
    if best_object is not None:
        objectInfo.append(best_object)

    return objectInfo

# Function to update the UI with movement count
def update_movement_count(count):
    label_movement_count.config(text=f"Movement Count: {count}")

# Function to update the UI with the current frame and handle motion detection
def update_frame():
    global movement_count, motion_detected, min_movement_area  # Declare global variables

    ret, frame = cap.read()

    if ret:
        frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)  # Resize for larger display

        fg_mask = bg_subtractor.apply(frame)
        fg_mask = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)[1]

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_time = time.time()

        # Check for the most recent significant movement (largest contour area)
        max_area = 0
        max_contour = None
        motion_detected = False  # Set motion detected flag

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_movement_area and area > max_area:
                max_area = area
                max_contour = contour

        if max_contour is not None:
            x, y, w, h = cv2.boundingRect(max_contour)
            current_detection = (x, y, w, h, current_time + 2)  # Set expiration time for this detection
            movement_count += 1  # Increment movement counter
            motion_detected = True  # Set motion detected flag

            # Draw object information
            for (box, className, confidence) in getObjects(frame[y:y+h, x:x+w], thres=0.45, nms=0.2, classNames=classNames):
                text = f"{className.upper()}: {round(confidence * 100, 2)}%"
                text_color = (255, 255, 255)  # White color for text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                font_thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = x + (w - text_size[0]) // 2
                text_y = y + text_size[1] + 5
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

            # Play alarm sound
            #winsound.PlaySound("alarm.wav", winsound.SND_ASYNC)

        # Draw active detection (if exists) on the frame
        if max_contour is not None and current_time < current_detection[4]:
            x, y, w, h, expiration_time = current_detection
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Update the UI with the movement count
        update_movement_count(movement_count)

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the frame to PIL Image
        pil_image = Image.fromarray(frame_rgb)

        # Convert PIL Image to ImageTk format
        imgtk = ImageTk.PhotoImage(image=pil_image)

        # Update the video feed label with the new image
        label_video.imgtk = imgtk
        label_video.config(image=imgtk)

        # Check if motion is detected
        if motion_detected:
            # Show motion detected alert
            label_motion_alert.config(text="Motion Detected", fg="red")
        else:
            # Hide motion detected alert if no motion
            label_motion_alert.config(text="")

    # Schedule the next update after 10 milliseconds
    label_video.after(10, update_frame)

# Function to update the minimum movement area
def update_min_movement_area(value):
    global min_movement_area
    min_movement_area = int(value)

# Create a Tkinter window
root = tk.Tk()
root.title("MotionSense")
root.geometry("800x600")  # Set window size to 800x600 pixels

# Create a label to display the video feed
label_video = tk.Label(root)
label_video.pack()

# Create a label to display the movement count
label_movement_count = tk.Label(root, text="Movement Count: 0", font=("Helvetica", 16))
label_movement_count.pack()

# Create a label to display motion detected alert
label_motion_alert = tk.Label(root, text="", font=("Helvetica", 16))
label_motion_alert.pack()

# Create a scale widget to adjust minimum movement area
label_sensitivity = tk.Label(root, text="Motion Sensitivity:")
label_sensitivity.pack()
scale_sensitivity = tk.Scale(root, from_=100, to=5000, orient=tk.HORIZONTAL, command=update_min_movement_area)
scale_sensitivity.set(2000)  # Initial value
scale_sensitivity.pack()

# Open the webcam
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set webcam width
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set webcam height

        bg_subtractor = cv2.createBackgroundSubtractorMOG2()

        # Parameters for movement detection
        min_movement_area = 2000  # Minimum area (in pixels) to consider as significant movement

        current_detection = None  # Store the most recent active detection
        movement_count = 0  # Initialize movement counter
        motion_detected = False  # Flag to indicate if motion is detected

        # Start updating the UI
        update_frame()

    # Run the Tkinter event loop
    root.mainloop()

    # Release the webcam and close OpenCV windows when the Tkinter window is closed
    cap.release()
    cv2.destroyAllWindows()