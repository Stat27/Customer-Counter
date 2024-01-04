# Customer Counter

Customer Counter is a Python application using OpenCV for tracking and counting customers entering a store. It utilizes background subtraction and contour detection to identify customers and a line crossing logic to count entries.

## Features

- **Video Processing**: Analyzes video feed from a security camera.
- **Customer Tracking**: Identifies and tracks customers using contour detection.
- **Entry Counting**: Counts customers as they cross a predefined line in the camera's view.
- **Dynamic Line Setting**: Allows users to set the counting line directly on the video feed.

## Requirements

- Python 3.x
- OpenCV library
- NumPy

## Installation

Ensure you have Python installed on your system. Then, install the required packages using pip:

```bash
pip install numpy opencv-python
