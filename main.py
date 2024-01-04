import cv2
import numpy as np

class TrackedObject:
    def __init__(self, contour):
        self.contour = contour
        self.path = []

class CustomerCounter:
    def __init__(self):
        self.cap = cv2.VideoCapture(r"C:\Users\sxu02\github\Customer-Counter\test_video.mp4")
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=100, detectShadows=False)

        self.enter_count = 0
        self.exit_count = 0
        self.ROI = None

    def SelectROI(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to retrieve frame. Exiting...")
            self.cap.release()
            cv2.destroyAllWindows()
            return

        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select ROI", 600, 400)
        self.ROI = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")

    def track_objects(self, tracked_objects, contours):
        updated_tracked_objects = []

        for obj in tracked_objects:
            min_distance = float("inf")
            closest_contour = None

            for contour in contours:
                # Calculate the centroid of the contour
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Calculate the distance between the object's path and the centroid
                distance = cv2.pointPolygonTest(obj.contour, (cX, cY), True)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_contour = contour

            # If a close contour is found, update the object's path and reset it
            if closest_contour is not None and min_distance < threshold_distance:
                obj.contour = closest_contour
                obj.path.append((cX, cY))
                updated_tracked_objects.append(obj)

        return updated_tracked_objects

    def check_enter(self, tracked_objects):
        for obj in tracked_objects:
            if len(obj.path) >= 2:
                start, end = obj.path[0], obj.path[-1]
                if (
                    (start[0] < self.ROI[0] and end[0] > self.ROI[0]) or  # Enters from the left
                    (start[0] > self.ROI[0] and end[0] < self.ROI[0]) or  # Enters from the right
                    (start[1] < self.ROI[1] and end[1] > self.ROI[1]) or  # Enters from the top
                    (start[1] > self.ROI[1] and end[1] < self.ROI[1])    # Enters from the bottom
                ):
                    self.enter_count += 1

    def run(self):
        self.SelectROI()

        tracked_objects = []  # Initialize tracked_objects here
        counted_objects = set()  # Initialize counted_objects here
        ret, prev_frame = self.cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to retrieve frame. Exiting...")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Background Subtraction
            fgMask = self.backSub.apply(frame)
            kernel = np.ones((3, 3), np.uint8)
            fgMask = cv2.erode(fgMask, kernel, iterations=1)
            fgMask = cv2.dilate(fgMask, kernel, iterations=2)

            # Find contours on the fgMask for further processing
            contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours based on area
            filtered_contours = []
            for contour in contours:
                if cv2.contourArea(contour) >= 500:
                    filtered_contours.append(contour)

            for contour in filtered_contours:
                # Check if the contour is within the ROI
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)
                if (
                    center[0] > self.ROI[0] and center[0] < self.ROI[0] + self.ROI[2] and
                    center[1] > self.ROI[1] and center[1] < self.ROI[1] + self.ROI[3]
                ):
                    continue  # Ignore contours within the ROI

                # x, y, w, h = cv2.boundingRect(contour)
                cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.track_movement(contour, tracked_objects)

            self.check_enter(tracked_objects, counted_objects)

            cv2.rectangle(frame, (int(self.ROI[0]), int(self.ROI[1])),
                          (int(self.ROI[0] + self.ROI[2]), int(self.ROI[1] + self.ROI[3])),
                          (255, 0, 0), 2)

            cv2.putText(frame, f'Enter: {self.enter_count}; Exit: {self.exit_count}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Frame', frame)
            cv2.imshow('Foreground Mask', fgMask)

            prev_gray = gray.copy()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# Instantiate and run
counter = CustomerCounter()
counter.run()
