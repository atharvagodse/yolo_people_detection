import cv2
import math
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("cctv.mp4")

CROWD_THRESHOLD = 4       # minimum people in a crowd
DISTANCE_THRESHOLD = 80   # pixels between people to consider in same cluster

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5, classes=[0])
    centers = []

    # Get center points of all detected people
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            centers.append((cx, cy))
            # Draw person box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Group people into clusters
    clusters = []
    visited = [False]*len(centers)

    for i in range(len(centers)):
        if visited[i]:
            continue
        cluster = [centers[i]]
        visited[i] = True
        for j in range(len(centers)):
            if visited[j]:
                continue
            dist = math.dist(centers[i], centers[j])
            if dist < DISTANCE_THRESHOLD:
                cluster.append(centers[j])
                visited[j] = True
        if len(cluster) >= CROWD_THRESHOLD:
            clusters.append(cluster)

    # Draw bounding boxes around clusters
    for cluster in clusters:
        cluster_np = np.array(cluster)
        x_min = np.min(cluster_np[:,0]) - 20
        y_min = np.min(cluster_np[:,1]) - 20
        x_max = np.max(cluster_np[:,0]) + 20
        y_max = np.max(cluster_np[:,1]) + 20
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
        cv2.putText(frame, "CROWD", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Crowd Area Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
