import cv2
import numpy as np

class CustomerCounter:
    def __init__(self):
        self.cap = cv2.VideoCapture(r"C:\Users\sxu02\github\Customer-Counter\test_video.mp4")
        self.backSub = cv2.createBackgroundSubtractorKNN(history=200, dist2Threshold=700.0, detectShadows=False)

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

    def track_movement(self, contour, tracked_objects):
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)

        for obj in tracked_objects:
            if obj['id'] == id(contour):
                obj['path'].append(center)
                break
        else:
            tracked_objects.append({'id': id(contour), 'path': [center]})

    def check_enter(self, tracked_objects):
        for obj in tracked_objects:
            if len(obj['path']) >= 2:
                start, end = obj['path'][0], obj['path'][-1]
                if start[0] > self.ROI[0] + self.ROI[2] and end[0] < self.ROI[0]:
                    self.enter_count += 1
                    tracked_objects.remove(obj)

    def run(self):
        self.SelectROI()

        tracked_objects = []  # Initialize tracked_objects here
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
            kernel = np.ones((3,3), np.uint8)
            fgMask = cv2.erode(fgMask, kernel, iterations=1)
            fgMask = cv2.dilate(fgMask, kernel, iterations=2)


            # Optical Flow calculation
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Use the flow to detect and track movement
            # ...

            # Find contours on the fgMask for further processing
            contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Filter based on area, etc.
                if cv2.contourArea(contour) < 500:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.track_movement(contour, tracked_objects)

            self.check_enter(tracked_objects)

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
