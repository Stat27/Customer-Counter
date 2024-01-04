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
pip installnumpy opencv-python
```
Start the Application: Run the main script to start the customer counter.

## Usage

1. In the python CustomerCounter.py
2. Select the Counting Line: Click on two points in the video feed to set the line where customer entries will be counted.
3. View the Count: The application window displays the number of customers entering.
4. Exit: Press 'q' to quit the application.

## How It Works

- **Background Subtraction: Separates moving customers from the static background using MOG2 algorithm.
- **Contour Detection and Merging: Identifies and merges the outlines of customers.
- **Line Intersection Logic: Determines when a customer's path intersects with the predefined line, indicating an entry.
