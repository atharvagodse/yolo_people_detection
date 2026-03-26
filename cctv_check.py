import cv2

cap = cv2.VideoCapture("cctv.mp4")

if cap.isOpened():
    print("Video loaded successfully")
else:
    print("Error loading video")

cap.release()
