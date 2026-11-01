# Fire_Water_AR

A basic webcam project that detects up to two hands and draws animated fire or
water around each hand. The live camera is shown normally until you raise a
finger. Then the background turns black while MediaPipe keeps you visible in
the frame, with a hand outline, effects, and status text.

## Requirements

- Windows
- Python 3.11
- A webcam
- Internet access for the first setup, to install packages and download the
  MediaPipe hand landmark model

The project uses OpenCV, MediaPipe, and NumPy. Its 640 x 480 camera setting and
procedural 2D effects are intended to run comfortably on a basic computer with
6 GB of memory.

## Setup and run

1. Double-click `setup.bat`. It creates a local `.venv`, installs the pinned
   dependencies, and downloads the hand landmark model to `models`.
2. Double-click `run.bat` to start the webcam application.
3. Press **Q** in the application window to quit.

`run.bat` also invokes setup if the environment or model is missing.

## Gestures

- **0 fingers:** keep the last selected effect; a fist holds fire or water in
  place around the moving hand
- **1 finger:** animated orange and yellow fire
- **2 or more fingers:** animated blue and cyan water

Each detected hand is handled independently. Raise one finger on one hand and
two or more fingers on the other to see both effects at once. Show a different
nonzero finger count to switch that hand's effect. The app shows the detected
finger count, current effect, and FPS.

## Troubleshooting

- If setup cannot find Python, install Python 3.11 and enable the Python
  launcher, then run `setup.bat` again.
- If the webcam cannot be opened, connect it and close other applications using
  it before running `run.bat`.
- If the model download fails, check your internet connection and run
  `setup.bat` again.
