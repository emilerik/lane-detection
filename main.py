import numpy as np
import cv2
import matplotlib
from pip._internal.utils import logging


# https://stackoverflow.com/questions/53977777/how-can-i-only-keep-text-with-specific-color-from-image-via-opencv-and
# -python
def detect_edges(frame):
    # Detects edges in image

    # First, convert image from BGR to HSV to separate the colors more clearly
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # cv2.imshow("hsv", hsv)

    # Then, create a filter based on hue ("what color" 0-180), saturation ("how much color" 0-255),
    # and value ("how bright" 0-255)
    dark_black = np.array([0, 0, 0])
    light_black = np.array([180, 255, 70])

    # Then, mask the image to keep only the pixels that went through the filter
    mask = cv2.inRange(hsv, dark_black, light_black)
    # cv2.imshow("mask", mask)

    # Use Canny algorithm to detect edges
    edges = cv2.Canny(mask, 0, 0)
    # cv2.imshow("canny", edges)

    return edges


def detect_line_segments(edges):
    rho = 1  # pixel resolution is 1 pixel
    angle = np.pi / 180  # angular resolution, which is 1 degree
    min_threshold = 5  # the minimum number of intersections to form a line

    #for full res 4023x1956
    #line_segments = cv2.HoughLinesP(edges, rho, angle, min_threshold,
    #                                np.array([]), minLineLength=250, maxLineGap=30)

    # for 500x243
    #line_segments = cv2.HoughLinesP(edges, rho, angle, min_threshold,
    #                                np.array([]), minLineLength=10, maxLineGap=10)

    # for 200x97 and down
    line_segments = cv2.HoughLinesP(edges, rho, angle, min_threshold,
                                    np.array([]), minLineLength=10, maxLineGap=10)
    print("Line segments: ", line_segments)
    print("Stop line segments")

    return line_segments


def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 10)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]


def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        #logging.info('No line_segment segments detected')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1 / 3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary  # right lane line segment should be on left 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    #logging.debug('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]

    return lane_lines


def detect_lane(frame):
    edges = detect_edges(frame)
    # cropped_edges = region_of_interest(edges)
    line_segments = detect_line_segments(edges)
    lane_lines = average_slope_intercept(frame, line_segments)
    # lane_lines = True

    return lane_lines


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=1):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


frame_in = cv2.imread('/Users/Emil/Documents/Programmering/Apps/BK123/lane-dec-2/20200331_083757_lowres_50.jpg')

edges = detect_edges(frame_in)
lane_lines = detect_lane(frame_in)
cv2.imshow("edges", edges)

lane_lines_image = display_lines(frame_in, lane_lines)
cv2.imshow("lane lines", lane_lines_image)

print("shape", frame_in.shape)

cv2.waitKey(0)
