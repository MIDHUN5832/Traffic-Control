import cv2 as cv
import numpy as np

def contours_detector(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blank = np.zeros(img.shape[:2], dtype='uint8')
    blurred_image = cv.GaussianBlur(gray, (5, 5), 0)

    edges = cv.Canny(blurred_image, 50, 150)

    contours, hierarchy = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    count = len(contours)

    cv.drawContours(blank, contours, -1, (255), thickness=2)
    blank = cv.dilate(blank, (3,3), iterations=1)

    return blank, count

def crop_image(image, top, left, right):
    polygon = np.array([[0, image.shape[0]], [left[0], left[1]], [top[0], top[1]], [right[0], right[1]], [image.shape[1], image.shape[0]]], np.int32)
    polygon = polygon.reshape((-1, 1, 2))

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv.fillPoly(mask, [polygon], 255)

    masked_image = cv.bitwise_and(image, image, mask=mask)

    x, y, w, h = cv.boundingRect(polygon)
    cropped_image = masked_image[y:y+h, x:x+w]

    return cropped_image


def background_contour(video_path, top, left, right): 
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    vehicle_counts = []
    last_10_counts = [] 
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        frame = cv.resize(frame, (800, 600))

        if frame_count % 10 == 0:  

            # Crop the region of interest (ROI)
            cropped_image = crop_image(frame, top, left, right)
            blank, count = contours_detector(cropped_image)

            # Store the vehicle count
            last_10_counts.append(count)
            if len(last_10_counts) > 10:
                last_10_counts.pop(0)  

            # Compute moving average
            avg_vehicle_count = int(np.mean(last_10_counts))
            vehicle_counts.append(avg_vehicle_count)

        frame_count += 1

    back_contours =  np.min(vehicle_counts) - 20
    return back_contours

        
