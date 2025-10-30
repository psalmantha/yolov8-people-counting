import cv2
import os
import datetime
from ultralytics import YOLO
from pytube import YouTube

VIDEO_URL = 'https://www.youtube.com/watch?v=PVUjP_I3c4Q'
INPUT_VIDEO = 'people_entering.mov'
OUTPUT_VIDEO = 'counting_output.mp4'
LINE_Y = 350
MAX_DURATION_SECONDS = 60

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

print("Video is processing...")

while True:
    ret, frame = cap.read()
    if not ret or frame_count >= frame_limit:
        break

    results = model(frame, stream=True)
    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] == 'person':
                x1, y1, x2, y2 = box.xyxy[0]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                id = box.id if hasattr(box, 'id') else None
                detections.append((cx, cy, id))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # for counting line
    cv2.line(frame, (0, LINE_Y), (width, LINE_Y), (0, 0, 255), 2)

    # count logic
    for (cx, cy, id) in detections:
        if abs(cy - LINE_Y) < 10:
            if id not in ids_crossed:
                ids_crossed.add(id)
                count += 1
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

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
