import cv2
import numpy as np

def merge_close_contours(contours, threshold=100):
    merged_contours = []
    # Set to keep track of processed contours, so we don't have a duplicate contour
    processed = set()  

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