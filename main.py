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
LINE_Y = 950
MAX_DURATION_SECONDS = 60
DETECT_EVERY = 1
HEADLESS = False

MIN_DELTA_Y = 0

REQUIRE_CONSECUTIVE = 1
WARMUP_FRAMES = 10   
RECENT_COUNT_MAX_DIST = 80

RECENT_COUNT_TTL_FRAMES = 150
RECENT_COUNT_MAX_DIST = 120

PERMISSIVE_FIRST_SEEN = True
PERMISSIVE_STABLE_FRAMES = 3

PROTECT_FRAMES = 120
PROTECT_DISTANCE = 220

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

fps = int(cap.get(cv2.CAP_PROP_FPS)) # 
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_limit = fps * MAX_DURATION_SECONDS

print(f"Video FPS={fps}, frame_limit={frame_limit}, resolution=({width}x{height})")

out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

count = 0
ids_crossed = set()
frame_count = 0
prev_detections = []

# simple centroid tracker
class SimpleCentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=90):
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


tracker = SimpleCentroidTracker(max_disappeared=30, max_distance=90)
id_boxes = dict()       # id -> last known box coords (x1,y1,x2,y2)
prev_centroids = dict() # id -> (x,y) previous frame centroid
prev_bottoms = dict()   # id -> previous bbox bottom (y2)
below_counts = dict()   # id -> consecutive frames below line (for REQUIRE_CONSECUTIVE)
recent_counts = []  # list of (frame_index, x, y) for recent counted events
first_seen_frame = dict()
first_seen_below_counts = dict()
protected_areas = []  # list of (count_frame, x, y, protect_until_frame)

def is_recently_counted(cx, bottom, current_frame):
    """Return True if a recent count or protected area is near (cx,bottom)."""
    for (fidx, rx, ry) in recent_counts:
        if current_frame - fidx <= RECENT_COUNT_TTL_FRAMES and math.hypot(rx - cx, ry - bottom) <= RECENT_COUNT_MAX_DIST:
            return True
    for (cf, rx, ry, until) in protected_areas:
        if current_frame <= until and math.hypot(rx - cx, ry - bottom) <= PROTECT_DISTANCE:
            return True
    return False

print("Video is processing...")

while True:
    ret, frame = cap.read()
    if not ret or frame_count >= frame_limit:
        break

    recent_counts = [rc for rc in recent_counts if frame_count - rc[0] <= RECENT_COUNT_TTL_FRAMES]
    protected_areas = [p for p in protected_areas if frame_count <= p[3]]

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

    # count logic using bbox-bottom (y2) with jitter filtering and consecutive-frame check
    for (cx, cy, oid, x1, y1, x2, y2) in detections:
        event = ''
        if x1 != 0 or y1 != 0 or x2 != 0 or y2 != 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

        # fall back to centroid y if box not available
        bottom = y2 if (y2 is not None and y2 != 0) else cy
        prev_bottom = prev_bottoms.get(oid, None)

        if prev_bottom is not None:
            # require previous bottom to be above line and current bottom to be below (and a minimum delta)
            if prev_bottom < LINE_Y and bottom >= LINE_Y and (bottom - prev_bottom) >= MIN_DELTA_Y and oid not in ids_crossed:
                # count only after the object has been below the line for REQUIRE_CONSECUTIVE detections
                below_counts[oid] = below_counts.get(oid, 0) + 1
                if below_counts[oid] >= REQUIRE_CONSECUTIVE:
                    # suppress duplicate counts near recent counted positions
                    skip = False
                    if is_recently_counted(cx, bottom, frame_count):
                        skip = True
                        print(f"Skipping duplicate count for id={oid} at frame={frame_count} (near recent count/protected area)")
                    else:
                        skip = False
                    if not skip:
                        ids_crossed.add(oid)
                        count += 1
                        recent_counts.append((frame_count, cx, bottom))
                        # add a protected area to avoid recounts for nearby/new tracker ids
                        protected_areas.append((frame_count, cx, bottom, frame_count + PROTECT_FRAMES))
                        event = 'counted'
                        print(f"Counted id={oid} at frame={frame_count}, bottom={bottom}, prev_bottom={prev_bottom}, total={count}")
            else:
                # reset consecutive below counter if condition not met
                below_counts[oid] = 0
        else:
            first_seen_frame.setdefault(oid, frame_count)
            if bottom >= LINE_Y:
                first_seen_below_counts[oid] = first_seen_below_counts.get(oid, 0) + 1
            else:
                first_seen_below_counts[oid] = 0

            prev_bottoms[oid] = bottom
            prev_centroids[oid] = (cx, cy)
            below_counts[oid] = 0

            # permissive count for stable, first-seen-below
            if PERMISSIVE_FIRST_SEEN and first_seen_below_counts.get(oid, 0) >= PERMISSIVE_STABLE_FRAMES and oid not in ids_crossed and frame_count - first_seen_frame.get(oid, frame_count) + 1 >= PERMISSIVE_STABLE_FRAMES:
                if is_recently_counted(cx, bottom, frame_count):
                    skip = True
                    print(f"Skipping permissive count for first-seen id={oid} at frame={frame_count} (near recent count/protected area)")
                else:
                    skip = False
                if not skip:
                    ids_crossed.add(oid)
                    count += 1
                    recent_counts.append((frame_count, cx, bottom))
                    protected_areas.append((frame_count, cx, bottom, frame_count + PROTECT_FRAMES))
                    print(f"Permissive-counted first-seen id={oid} at frame={frame_count}, bottom={bottom}, total={count}")
                    event = 'counted'

        # store bottoms and centroids for next frame
        prev_bottoms[oid] = bottom
        prev_centroids[oid] = (cx, cy)

        if PERMISSIVE_FIRST_SEEN and oid not in ids_crossed:
            if first_seen_below_counts.get(oid, 0) >= PERMISSIVE_STABLE_FRAMES and (frame_count - first_seen_frame.get(oid, frame_count) + 1) >= PERMISSIVE_STABLE_FRAMES:
                if is_recently_counted(cx, bottom, frame_count):
                    skip = True
                    print(f"Skipping permissive count for id={oid} at frame={frame_count} (near recent count/protected area)")
                else:
                    skip = False
                if not skip:
                    ids_crossed.add(oid)
                    count += 1
                    recent_counts.append((frame_count, cx, bottom))
                    protected_areas.append((frame_count, cx, bottom, frame_count + PROTECT_FRAMES))
                    event = 'counted'
                    print(f"Permissive-counted id={oid} at frame={frame_count}, bottom={bottom}, total={count}")

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
