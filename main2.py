import cv2
import numpy as np

class TrackedObject:
    _id_counter = 0  # Class variable to keep track of the last used ID

    def __init__(self, centroid):
        self.centroid = centroid
        self.path = [centroid]
        self.id = TrackedObject._id_counter
        TrackedObject._id_counter += 1  # Increment ID for next object
    
    def update(self, new_centroid):
        self.centroid = new_centroid
        self.path.append(new_centroid)

def merge_close_contours(contours, threshold=50):
    merged_contours = []
    processed = set()  # Set to keep track of processed contours

    for i, current_contour in enumerate(contours):
        if i in processed:
            continue

        x, y, w, h = cv2.boundingRect(current_contour)
        current_rect = (x, y, x + w, y + h)
        to_merge = []

        for j, other_contour in enumerate(contours):
            if j in processed:
                continue

            ox, oy, ow, oh = cv2.boundingRect(other_contour)
            other_rect = (ox, oy, ox + ow, oy + oh)

            if rects_are_close(current_rect, other_rect, threshold):
                to_merge.append(other_contour)
                processed.add(j)

        if to_merge:
            merged_contours.append(cv2.convexHull(np.vstack(to_merge)))

    return merged_contours


def rects_are_close(rect1, rect2, threshold):
    x1, y1, x2, y2 = rect1
    ox1, oy1, ox2, oy2 = rect2

    return (abs(x1 - ox1) < threshold and abs(y1 - oy1) < threshold) or \
           (abs(x2 - ox2) < threshold and abs(y2 - oy2) < threshold)


class CustomerCounter:
    def __init__(self):
        self.cap = cv2.VideoCapture(r"C:\Users\sxu02\github\Customer-Counter\test_video.mp4")
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=120, detectShadows=False)

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

        # Function to handle mouse click events
        def click_event(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                params.append((x, y))

        # Display the frame and capture two points
        cv2.namedWindow("Select Line", cv2.WINDOW_NORMAL)
        points = []
        cv2.setMouseCallback("Select Line", click_event, points)

        while True:
            cv2.imshow("Select Line", frame)
            if len(points) == 2:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Draw the line for visual confirmation
        if len(points) == 2:
            cv2.line(frame, points[0], points[1], (0, 255, 0), 2)
            cv2.imshow("Select Line", frame)
            cv2.waitKey(500)  # Display the line for 500 milliseconds

        # Store the line points
        self.line = points

        # Close the window
        cv2.destroyWindow("Select Line")



    def get_centroid(self, contour):
        if contour is None or not isinstance(contour, np.ndarray) or len(contour) == 0:
            return None
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    
    def track_objects(self, tracked_objects, merged_contours):
        new_centroids = [self.get_centroid(contour) for contour in merged_contours]
        new_centroids = [c for c in new_centroids if c is not None]

        updated_tracked_objects = []

        # Update existing tracked objects
        for obj in tracked_objects:
            closest_centroid, min_distance = None, float("inf")
            for c in new_centroids:
                distance = np.linalg.norm(np.array(obj.centroid) - np.array(c))
                if distance < min_distance:
                    min_distance = distance
                    closest_centroid = c

            if closest_centroid and min_distance < self.threshold_distance:
                obj.update(closest_centroid)
                updated_tracked_objects.append(obj)
                new_centroids.remove(closest_centroid)

        # Add new tracked objects for any remaining centroids
        for c in new_centroids:
            updated_tracked_objects.append(TrackedObject(c))

        return updated_tracked_objects


    def line_intersection(self, line1, line2):
        # Unpack points
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2

        # Calculate determinants
        det_line1 = (x1 * y2 - y1 * x2)
        det_line2 = (x3 * y4 - y3 * x4)
        x1mx2, y1my2 = x1 - x2, y1 - y2
        x3mx4, y3my4 = x3 - x4, y3 - y4

        # Calculate the denominator
        div = (x1mx2 * y3my4 - y1my2 * x3mx4)

        if div == 0:
            return False  # Lines don't intersect

        # Calculate the intersecting point
        px = (det_line1 * x3mx4 - x1mx2 * det_line2) / div
        py = (det_line1 * y3my4 - y1my2 * det_line2) / div

        # Check if the intersection point is on both line segments
        if (min(x1, x2) <= px <= max(x1, x2) and
            min(y1, y2) <= py <= max(y1, y2) and
            min(x3, x4) <= px <= max(x3, x4) and
            min(y3, y4) <= py <= max(y3, y4)):
            return True

        return False

    def check_enter(self, tracked_objects):
        for obj in tracked_objects:
            if len(obj.path) < 2:
                continue

            # Check each segment of the path
            for i in range(1, len(obj.path)):
                if self.line_intersection((obj.path[i - 1], obj.path[i]), (self.line[0], self.line[1])):
                    self.enter_count += 1
                    break  # Avoid double-counting


    def run(self):
        self.SelectROI()
        tracked_objects = []  # Initialize tracked_objects here
        ret, prev_frame = self.cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        self.threshold_distance = 50  # Adjust this threshold as needed
        frame_writer = cv2.VideoWriter('frame_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))
        mask_writer = cv2.VideoWriter('mask_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))

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

            # Find and merge contours
            contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            merged_contours = merge_close_contours([c for c in contours if cv2.contourArea(c) >= 500])

            # Track objects with merged contours
            tracked_objects = self.track_objects(tracked_objects, merged_contours)

            # Check for entrances
            self.check_enter(tracked_objects)

            # Drawing contours and the line on the frame
            for contour in merged_contours:
                cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
            if len(self.line) == 2:
                cv2.line(frame, self.line[0], self.line[1], (255, 0, 0), 2)  # Draw the line

            cv2.putText(frame, f'Enter: {self.enter_count}; Exit: {self.exit_count}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Frame', frame)
            cv2.imshow('Foreground Mask', fgMask)

            # Writing frames to video
            frame_writer.write(frame)
            mask_writer.write(fgMask)

            prev_gray = gray.copy()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        frame_writer.release()
        mask_writer.release()
        cv2.destroyAllWindows()




# Instantiate and run
counter = CustomerCounter()
counter.run()

