# Dice Pose Estimation and Dot Counting

Estimates the top face of the dice and counts its dots.


# Algorithm

1. **Dice body segmentation**
   - The calibrated `dice_body_contour` range is used to mask the dice and extract its largest contour.
   - `cv2.approxPolyDP()` converts that contour into polygon points.

2. **Contour repair**
   - Rounded corners and missing edges often produce incomplete polygon outlines.
   - Parallel edge groups and Hough lines are used to insert missing points and recover a 6-point cube outline.

3. **Top-face estimation**
   - Ray intersections and Hough line evidence choose the most plausible top face from the 6-point outline.
   - The selected top-face corners are warped into a square top-down view with a homography in order to make the dots as circular as possible for the `cv2.HoughCircles()` algorithm.

4. **Dot counting**
   - The warped top face is masked with `dice_face_color` and inverted so the dark dots become foreground.
   - `cv2.HoughCircles()` counts the visible dots.

5. **Temporal stabilization**
   - Frame-to-frame contour similarity estimates whether the dice is stable.
   - A count must repeat for several frames before it is displayed.
   

# Debug Mode
Debug mode shows the intermediate geometry and masks used by the pipeline:
- current frame number, tracking state, and contour similarity score
- detected dice contour and approximated/repaired corner points
- selected top-face outline and intersection point
- separate windows for the cropped edge/Hough-line view, warped top face, and blurred dot mask


# How to use it on a dice with different colors
Currently things are setup for that specific green dice. If you want to run it on a different dice or a different lighting
setup, run
``` batch
python setup_calibrate_colors.py
```
This opens the following UI:  
![Alt text](calibrate_colors.jpg)  
It makes you choose 2 colors. Both colors are main color of the dice (green in this case), but in the upper part 
(*dice_body_contour*), focus on getting a nice contour, he'll need that to estimate the geometry.  
In the lower part (*dice_face_color*) adjust the sliders so the dots on the top face have strong black contours. This
is needed for the actual dot counting

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
