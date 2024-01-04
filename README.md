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

- **Background Subtraction**: Separates moving customers from the static background using MOG2 algorithm.
- **Contour Detection and Merging**: Identifies and merges the outlines of customers.
- **Line Intersection Logic**: Determines when a customer's path intersects with the predefined line, indicating an entry.

## Customization
1. Modify the threshold_distance in CustomerCounter.py to adjust contour merging sensitivity.
2. Adjust history and varThreshold in the background subtractor for different environmental conditions.
3. Adjust the threshold in merge_close_contour function, it's set default to be 100, higher means more aggressive in merging contours
    
## Camera Setup
Position the camera to have a clear view of the area where customer entries need to be counted, A vertical setup would be ideal
Ensure proper lighting conditions for optimal detection and tracking.
The camera should be stable and fixed in position to avoid false detections.
Here is an example:
<img width="546" alt="image" src="https://github.com/Stat27/Customer-Counter/assets/90141495/9b2b1b82-a027-4edb-b034-b246be6a3a85">
