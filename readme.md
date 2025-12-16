# Webcam AI Filter (WIP)

A small Python app that captures a live webcam stream and runs **real-time face detection** using a **pre-trained MediaPipe model**.  
This repository is being built step-by-step toward a simple face filter overlay and optional background segmentation.

## Features (so far)
- Live webcam capture (OpenCV)
- Real-time face detection (MediaPipe Face Detection, pre-trained)
- Draws a bounding box around the detected face
- PNG overlay filter (transparent PNG blended onto the webcam feed)
- Modular project structure (`src/` package)

## Assets
Place a transparent PNG overlay here:

- `assets/glasses.png`

The PNG must contain an **alpha channel** (transparency). The overlay is resized relative to the detected face box and alpha-blended onto the frame.

## Setup (Conda)

### 1) Create and activate the environment
From the project root:

```bash
conda env create -f environment.yml
conda activate webcam-ai-filter
conda activate webcam-ai-filter
```

### 2) Run the app
```bash
python -m src.main
```

## Controls
- `q` — quit the app

## Notes
- OpenCV frames are BGR by default; MediaPipe expects RGB (conversion is done in code).

## Next Steps
- Add overlay filter (PNG with alpha blending) based on direction of view
- Add simple face tracking / smoothing
- Add background segmentation and replacement
