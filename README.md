# People Counting with YOLOv8 and a Simple Centroid Tracker

This repository implements a people-counting pipeline using YOLOv8 for person detection and a lightweight centroid tracker to produce stable object IDs. The script annotates video frames and counts people crossing a horizontal line (doorway/threshold).

## Summary

- Detection: YOLOv8 is used to detect person bounding boxes in frames.
- Tracking: A small `SimpleCentroidTracker` matches centroids across frames to maintain stable IDs.
- Counting cue: We prefer the bottom of the bounding box (`y2`) as the physical cue for feet/ground contact. If unavailable, we fall back to the centroid's y coordinate.
- Crossing rule: An object is counted when it transitions from above the counting line to below it (above->below). A permissive-first-seen option will count objects that first appear already below the line after they remain stable for a few frames.
- Duplicate suppression: A short-term suppression (recent_counts) was applied and a longer "protected area" after a count to avoid re-counting the same person when tracker IDs change or when a bbox briefly re-crosses the line.

## Files

- `main.py` — Main script. Tunable parameters are at the top of the file.
- `yolov8n.pt` — YOLOv8 model file (gets generate upon running main.py).

## How it works

1. Open the input video with OpenCV and read video metadata (FPS, resolution) or optionally download the sample video from YouTube if not present.
2. For each frame (or every `DETECT_EVERY` frame):
   - Run YOLO inference and collect person boxes and centroids.
   - Update the centroid tracker to obtain stable `object_id -> centroid` mappings.
   - Map tracker IDs back to their nearest bounding box for `y2` (bottom) values.
3. For each tracked object, compare the previous bottom (or centroid) to the current bottom to detect an above→below transition.
4. If the transition and stability checks pass and suppression rules do not block the event, increment the count and create a protected zone to prevent duplicates.
5. Optionally, if an object first appears already below the line and remains present for `PERMISSIVE_STABLE_FRAMES`, count it once (useful for short people or those who walk fast).

## Results and observations

- One of the factors affecting accuracy was the vertical position of the counting line (`LINE_Y`). Make sure `LINE_Y` corresponds to the physical threshold / feet location in your video frames.
- For shorter people (or low-resolution detections) the bounding-box bottom can be noisy; `PERMISSIVE_FIRST_SEEN` and `PERMISSIVE_STABLE_FRAMES` help recover missed counts at the cost of potential false positives.
- Duplicate counts most often arise from tracker ID reassignments or boxes flickering around the line; `recent_counts` and `protected_areas` were implemented to counter this.

## Tuning recommendations

1. Set the counting line `LINE_Y` to the approximate vertical pixel row where people's feet cross the doorway.
2. Start with `DETECT_EVERY = 1` when tuning to see all detections. Once stable, increase to 2–3 to save time.
3. If people are counted twice when they hesitate to enter, increase `PROTECT_FRAMES` (longer protection window) or `PROTECT_DISTANCE` (wider protection radius).
4. If short/fast people are missed, set `PERMISSIVE_FIRST_SEEN = True` and `PERMISSIVE_STABLE_FRAMES = 1..3`. Lower values increase sensitivity but can increase false positives.
5. Adjust `MIN_DELTA_Y` and `REQUIRE_CONSECUTIVE` to filter out jitter or camera shake.

## How to run

Ensure Python packages are installed (OpenCV, numpy, ultralytics, pytube). Example:

```powershell
pip install -r requirements.txt
python main.py
```

Run with `HEADLESS = True` (in `main.py`) to avoid opening the display window (useful for batch runs).

## Troubleshooting

- If the script immediately exits with `frame_limit = 0` or FPS=0, check that `cap.get(cv2.CAP_PROP_FPS)` returns a valid FPS for your video.
- If many detections are missing, try running with `DETECT_EVERY = 1` and `HEADLESS = False` to visually inspect detections.
- If you get duplicate counts, increase `PROTECT_FRAMES` and/or `PROTECT_DISTANCE`.