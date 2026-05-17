# Cars Multicamera Matching

Estimates the top face of the dice and counts its dots.

![Alt text](car_multicamera_short.gif)  
Click [here](https://youtu.be/telgQXJkKVk) to see a longer and larger version on YouTube:



# Algorithm

1. **Detection and tracking**
   - Each camera frame on both cameras is masked so YOLO only sees the relevant road area (in debug mode, click M to see the masks)
   - A custom YOLO car detector runs on both cameras.
   - BotSort keeps stable per-camera track IDs for detected vehicles.

2. **Source-camera gallery creation**
   - The source camera is treated as the camera where vehicles should appear first.
   - While a source-camera track is visible, the system saves cropped vehicle images.
   - Crops are marked as **strong** when the vehicle is large enough, not overlapping another car, and moving in the expected direction. Other usable crops are kept as weak fallback crops.
   - If a source track crosses one of the configured discard lines (yellow lines), it is removed because it is moving toward an irrelevant exit.
   - When a source track disappears, the saved crops are converted into vehicle ReID embeddings and averaged into one gallery embedding for that source track.
   - Source records expire after `source_record_ttl_seconds` so old vehicles are not matched indefinitely.

3. **Query-camera candidate filtering**
   - Query-camera tracks are ignored if they appear in an area marked as not coming from the source camera (in debug mode, click O to see the mask)
   - For each remaining query track, the system stores ReID embeddings from its visible crops over time.
   - A query track becomes matchable after it crosses the configured query entry line.

4. **Cross-camera matching**
   - After a query track crosses the red entry line, its stored embeddings are averaged.
   - The averaged query embedding is compared with the source gallery using cosine similarity.
   - Matches are only considered if enough travel time has passed between the source track leaving and the query track crossing the entry line.
   - Weak source crops are penalized, and matches below the embedding-score threshold are treated as no match.


# Debug Mode
- description of what the debug mode does - 

# Try on Different Cube or Different environment
Currently things are setup for that specific green dice. If you want to run it on a different dice or a different lighting
setup, the setup_color_ranges.py needs to be run to pick different color ranges.


# Difficulties
The outputs from openCv functions are extremely messy. The main difficulty is probably that cv2.approxPolyDP()  returns
many points that are not located at actual corners. And with the corners of the dice already being rounded, many corners
aren't even detected.  
However since the dots are located more inside of the face instead of near the border, the actual dot counting is 
relatively forgiving.


# Known Issues  
- The detector currently relies on HSV color segmentation, so it is sensitive to lighting. To compensate this, the 
setup file setup_color_get_range.py can update the colors.
- Because of the messy top face detection, in certain frames he classifies the count to be one or two points higher or lower. 
This often gets fixed automatically by the algorithm requiring a number to show at least a few frames to be accepted. This 
works well in the current video because the camera is moving and therefore it's always fixing itself. However if this 
were a static camera, certain miss-classifications could show more than in a moving camera.
- Currently the setup gets confused quickly if there are objects in the scene with similar colors as the dice
  
# Ideas to improve
More time could be spent on the top face detection, add more edge cases, 
to minimize false classifications on dot counting. 

## Install

**1. Install Python packages**
```powershell
pip install -r requirements.txt
```

**3. Download Video File**  
green_cube.mp4


## Run

```powershell
python run.py
```

## Controls
- `d`: toggle debug mode
- `q`: quit
- `Space`: pause/resume
- Right arrow while paused: step one frame
- Mouse click a car: isolate that track in the clicked camera; click empty space to clear

Debug-only controls:

- `0`-`9`: number of match candidates to display
- `M`: toggle inference-ignore mask overlay
- `O`: toggle query not-from-source mask overlay
- `,` / `.`: page through source crop gallery


## Use of AI

AI-assisted development tools (primarily Codex) were used throughout the project to accelerate implementation, 
refactoring, and boilerplate generation.

Core algorithmic design, system integration, and debugging were performed mostly manually. 
Some utility modules, geometry helpers, and visualization code were heavily AI-assisted and subsequently reviewed/modified.
