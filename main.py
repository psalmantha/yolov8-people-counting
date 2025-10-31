import cv2
import os
import datetime
import math
import numpy as _np
from ultralytics import YOLO
from pytube import YouTube

VIDEO_URL = 'https://www.youtube.com/watch?v=PVUjP_I3c4Q'
INPUT_VIDEO = 'people_entering.mov'
OUTPUT_VIDEO = 'counting_output.mp4'
LINE_Y = 360
MAX_DURATION_SECONDS = 120
DETECT_EVERY = 1
HEADLESS = True

if not os.path.exists(INPUT_VIDEO):
    print("Downloading video from YouTube...")
    yt = YouTube(VIDEO_URL)
    stream = yt.streams.filter(file_extension='mp4', res='720p').first()
    stream.download(filename=INPUT_VIDEO)
    print("Video downloaded successfully!")

print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')
print("Model loaded successfully!")

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_limit = fps * MAX_DURATION_SECONDS

out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

count = 0
ids_crossed = set()
frame_count = 0
prev_detections = []

# simple centroid tracker
class SimpleCentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=90):
        self.next_object_id = 0
        self.objects = dict()  # id -> centroid (x,y)
        self.disappeared = dict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        oid = self.next_object_id
        self.objects[oid] = centroid
        self.disappeared[oid] = 0
        self.next_object_id += 1

    def deregister(self, oid):
        if oid in self.objects:
            del self.objects[oid]
        if oid in self.disappeared:
            del self.disappeared[oid]

    def update(self, input_centroids):
        # input_centroids: list of (x,y)
        if len(input_centroids) == 0:
            # mark all as disappeared
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects.copy()

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
            return self.objects.copy()

        # compute distance matrix
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[i] for i in object_ids]
        D = _np.linalg.norm(_np.array(object_centroids)[:, None, :] - _np.array(input_centroids)[None, :, :], axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        assigned_rows = set()
        assigned_cols = set()

        for r, c in zip(rows, cols):
            if r in assigned_rows or c in assigned_cols:
                continue
            if D[r, c] > self.max_distance:
                continue
            oid = object_ids[r]
            self.objects[oid] = input_centroids[c]
            self.disappeared[oid] = 0
            assigned_rows.add(r)
            assigned_cols.add(c)

        # increase disappeared for unassigned
        for i, oid in enumerate(object_ids):
            if i not in assigned_rows:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

        # register unassigned input centroids
        for j, c in enumerate(input_centroids):
            if j not in assigned_cols:
                self.register(c)

        return self.objects.copy()


tracker = SimpleCentroidTracker(max_disappeared=20, max_distance=60)
id_boxes = dict()       # id -> last known box coords (x1,y1,x2,y2)
prev_centroids = dict() # id -> (x,y) previous frame centroid

print("Video is processing...")

while True:
    ret, frame = cap.read()
    if not ret or frame_count >= frame_limit:
        break

    detections = []

    # run detection only every DETECT_EVERY frames
    if frame_count % DETECT_EVERY == 0:
        results = model(frame, stream=True)
        centroids = []
        boxes_for_centroids = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if model.names[cls] == 'person':
                    xy = box.xyxy[0]
                    try:
                        x1, y1, x2, y2 = [int(v) for v in xy]
                    except Exception:
                        x1, y1, x2, y2 = int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    centroids.append((cx, cy))
                    boxes_for_centroids.append((cx, cy, x1, y1, x2, y2))

        # update tracker with detected centroids and get stable IDs
        objects = tracker.update(centroids)

        # map tracker ids to boxes
        id_boxes.clear()
        # objects is dict id->(x,y) -- need to match to boxes_for_centroids by nearest
        for oid, centroid in objects.items():
            # find closest box
            best_box = None
            best_d = None
            for (cx, cy, x1, y1, x2, y2) in boxes_for_centroids:
                d = math.hypot(centroid[0] - cx, centroid[1] - cy)
                if best_d is None or d < best_d:
                    best_d = d
                    best_box = (cx, cy, x1, y1, x2, y2)
            if best_box is not None:
                bx = (best_box[2], best_box[3], best_box[4], best_box[5])
                id_boxes[oid] = bx

        # build detections list from tracker objects and id_boxes
        for oid, centroid in objects.items():
            cx, cy = int(centroid[0]), int(centroid[1])
            if oid in id_boxes:
                x1, y1, x2, y2 = id_boxes[oid]
            else:
                x1 = y1 = x2 = y2 = 0
            detections.append((cx, cy, oid, x1, y1, x2, y2))
        prev_detections = detections.copy()
    else:
        # reuse last detections for drawing and counting when skipping detection
        detections = prev_detections.copy()

    # counting line
    cv2.line(frame, (0, LINE_Y), (width, LINE_Y), (0, 0, 255), 2)

    # count logic using tracker IDs and previous centroids to detect crossing from above->below
    for (cx, cy, oid, x1, y1, x2, y2) in detections:
        # draw box and centroid
        if x1 != 0 or y1 != 0 or x2 != 0 or y2 != 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

        prev = prev_centroids.get(oid, None)
        if prev is not None:
            prev_y = prev[1]
            # detect downward crossing (above -> below)
            if prev_y < LINE_Y and cy >= LINE_Y and oid not in ids_crossed:
                ids_crossed.add(oid)
                count += 1
        prev_centroids[oid] = (cx, cy)

    cv2.putText(frame, f'Count: {count}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    elapsed_seconds = int(frame_count / fps)
    timestamp = str(datetime.timedelta(seconds=elapsed_seconds))
    cv2.putText(frame, f'Time: {timestamp}', (width - 220, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

    out.write(frame)
    cv2.imshow('People Counting', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\nProcessing complete.")
print(f"Final count: {count}")
print(f"Output video saved as: {OUTPUT_VIDEO}")
