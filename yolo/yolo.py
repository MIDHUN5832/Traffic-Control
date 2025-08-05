import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from ultralytics import YOLO


def contours_detector(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blank = np.zeros(img.shape[:2], dtype='uint8')
    blurred_image = cv.GaussianBlur(gray, (5, 5), 0)

    edges = cv.Canny(blurred_image, 50, 150)

    contours, hierarchy = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    count = len(contours)
    area = [cv.contourArea(contour) for contour in contours]

    vehicle_contours = [cnt for cnt in contours if 600 < cv.contourArea(cnt) < 1400]
    vehicle_count = len(vehicle_contours)

    cv.drawContours(blank, contours, -1, (255), thickness=2)
    blank = cv.dilate(blank, (3,3), iterations=1)

    return blank, vehicle_count, count

def crop_image(image, top, left, right):
  polygon = np.array([[0, image.shape[0]], [left[0], left[1]], [top[0], top[1]], [right[0], right[1]], [image.shape[1],image.shape[0]]],  np.int32)
  polygon = polygon.reshape((-1, 1, 2))

  mask = np.zeros(image.shape[:2], dtype=np.uint8)
  cv.fillPoly(mask, [polygon], 255)

  masked_image = cv.bitwise_and(image, image, mask=mask)

  x, y, w, h = cv.boundingRect(polygon)
  cropped_image = masked_image[y:y+h, x:x+w]

  return cropped_image


video_path = 'videos/road.mp4'
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv.CAP_PROP_FPS)
frame_interval = int(fps * 0.1)  

vehicle_counts = []
timestamps = []

model = YOLO('yolov8n.pt') 

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)  


    if frame_count % frame_interval == 0:
      # Yolo
      yolo_vehicle_count = 0
      results = model(frame)
    
      for result in results:
          for box in result.boxes:
              cls_id = int(box.cls)  # Convert tensor to integer
              label = result.names[cls_id]
              if label in ['car', 'truck', 'bus', 'motorbike']:
                  yolo_vehicle_count += 1

      current_time = datetime.now().strftime('%H:%M:%S')
      timestamps.append(current_time)

      top = (538, 891)
      right = (1079, 1491)
      left = (0, 1727)

      cropped_image = crop_image(frame, top, left, right)
      blank, vehicle_count, count = contours_detector(cropped_image)
      blank = cv.resize(blank, (500, 700))
      frame = cv.resize(frame, (500, 700))

      cv.putText(blank, f'Vehicle Count: {int((count-400)/20)}', (10,50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
      vehicle_counts.append(vehicle_count)

      yolo = cv.resize(cropped_image, (500, 700))

      # Display vehicle count
      cv.putText(yolo, f'Vehicle Count: {yolo_vehicle_count}', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

      # Display the resulting frame
      cv.imshow('yolo', yolo)
      cv.imshow("main", frame)
      cv.imshow("edge", blank)

    frame_count += 1

    # Break the loop on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break  

# Release the video capture object and close all OpenCV windows
cap.release()
cv.destroyAllWindows()

plt.figure(figsize=(10, 5))
plt.plot(timestamps[::10], vehicle_counts[::10], marker='o')
plt.xlabel('Time')
plt.ylabel('Vehicle Count')
plt.title('Vehicle Count Over Time')
plt.show()