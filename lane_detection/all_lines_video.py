import cv2
import numpy as np

# Load video
video_path = 'videos/road.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if video ends
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  

    frame = cv2.resize(frame, (800, 600))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    left_lane_lines = []
    # right_lane_lines = []

    # Draw detected lines and classify them
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Calculate slope

            if -0.7 < slope < -0.4:  # Left lane (negative slope)
                left_lane_lines.append((x1, y1, x2, y2))
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for left lane
            # elif slope > 0.3:  # Right lane (positive slope)
            #     right_lane_lines.append((x1, y1, x2, y2))
            #     cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for right lane

    # Display the processed frames
    cv2.imshow('Edges', edges)
    cv2.imshow('Lane Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
