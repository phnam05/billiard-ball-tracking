import cv2
import numpy as np
import Indexer  # Helper functions for getting indexes

from PoolTable import ToHSV, GetClothColor, GetContours, MaskTableBed, TransformToOverhead

cap = cv2.VideoCapture('kpc-break.mp4')  #Choose the video from the list of assets provided
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
result = cv2.VideoWriter('fedor_shot_result.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
areas = []
collide_radius = None
contour_list = []
sense = 10
overhead = True
detect_oj = False
debug_mode = False
# Parameters for detecting cueball
lower_bound = np.array([30, 20, 200]) - sense  # Lower bound for HSV color
upper_bound = np.array([55, 45, 233]) + sense  # Upper bound for HSV color
oj_lower_bound = np.array([100, 210, 100]) - sense-5  # Lower bound for HSV color
oj_upper_bound = np.array([110, 220, 120]) + sense+5  # Upper bound for HSV color
# oj_lower_bound = np.array([160, 50, 230])  -sense# Lower bound for HSV color PINK
# oj_upper_bound = np.array([180, 100, 240])  +sense# Upper bound for HSV color PINK
past_center = None
oj_past_center = None
collide = False
collide_coords = None
oj_collide_coords = None
oj_collide_radius = None

# Initialize variables for tracking
prev_center = []
oj_prev_center = []
# Main loop
while True:

    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(0)
    frame = cv2.resize(frame, (854, 480))
    result.write(frame)
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask to detect cue ball & object ball
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    oj_mask = cv2.inRange(hsv, oj_lower_bound, oj_upper_bound)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    oj_contours, _ = cv2.findContours(oj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If contours are found, find the largest one (assuming it's the cue ball)
    if contours:
        cueball_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(cueball_contour)
        center = (int(x), int(y))
        if len(prev_center) == 0:
            prev_center.append(center)

        elif len(prev_center) >= 1 and np.linalg.norm(np.array(center) - np.array(prev_center[-1])) < 100:
            prev_center.append(center)

        # Draw circle around cueball
        cv2.circle(frame, center, int(radius), (0, 0, 255), 2)
        past_center = center

    #Same procedure for object ball
    if oj_contours and detect_oj:
        oj_contour = max(oj_contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(oj_contour)

        #Collision check to draw contact point between CB and OB
        if not collide:
            collide_radius = int(radius)
        oj_center = (int(x), int(y))

        if len(oj_prev_center) == 0:
            oj_prev_center.append(oj_center)

        elif len(oj_prev_center) >= 1 and np.linalg.norm(np.array(oj_center) - np.array(oj_past_center)) < 20:

            oj_prev_center.append(oj_center)

        cv2.circle(frame, oj_center, int(radius), (0, 0, 255), 2)
        if oj_past_center is not None and np.linalg.norm(np.array(oj_center) - np.array(oj_past_center)) > 20:
            oj_past_center = oj_prev_center[-1]
        else:
            oj_past_center = oj_center

    if collide:
        oj_collide_coords = oj_past_center
        oj_collide_radius = collide_radius
        collide = False
    # Detect collision moment
    if past_center is not None and oj_past_center is not None:
        cueball_x, cueball_y = past_center
        objectball_x, objectball_y = oj_past_center
        distance = np.linalg.norm(np.array(past_center) - np.array(oj_past_center))
        if distance < 20:  # Adjust this threshold as needed
            # Draw marker at the collision point
            collide_coords = (int((cueball_x + objectball_x) / 2), int((cueball_y + objectball_y) / 2))
            collide = True

    #Draw the moments of contact
    if collide_coords is not None and oj_collide_coords is not None:
        cv2.circle(frame, collide_coords, 5, (250, 250, 250),
                   2)
        cv2.circle(frame, oj_collide_coords, 5, (255, 0, 0), 2)

    #Draw trajectories
    for i in range(1, len(prev_center)):
        if prev_center[i - 1] is None or prev_center[i] is None:
            continue
        cv2.line(frame, prev_center[i - 1], prev_center[i], (200, 255, 255), 2)

    for i in range(1, len(oj_prev_center)):
        if oj_prev_center[i - 1] is None or oj_prev_center[i] is None:
            continue
        cv2.line(frame, oj_prev_center[i - 1], oj_prev_center[i], (200, 0, 0), 2)


    #If specified, transform the current camera to an overhead one
    if overhead:
        #Get table cloth color and mask out the bed to create the warp perspective
        lower_color, upper_color = GetClothColor(hsv)
        contours = GetContours(hsv, lower_color, upper_color, 15)
        TableContour = MaskTableBed(contours)
        contour_list.append(TableContour)
        areas.append(cv2.contourArea(TableContour))
        largest_contour = Indexer.get_index_of_max(areas)
        TableContour = contour_list[largest_contour[0]]

        warp = TransformToOverhead(frame, TableContour)
        #Display the frame
        cv2.imshow('Overhead', warp)

    if debug_mode:
        _frame = frame.copy()
        cv2.drawContours(_frame,contours,-1,(0,255,0),3)
        cv2.imshow('Contours', _frame)
        cv2.imshow('mask',mask)

    # frame = cv2.resize(frame, (1280, 720))
    cv2.imshow('Billiard Ball Tracker', frame)

    # Check for key press to exit
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c'):
        # Redraw the frame without the lines by re-reading the original frame
        _, frame = cap.read()
        prev_center = []
        oj_prev_center = []
        cv2.imshow('Billiard Ball Tracker', frame)
        collide_coords = None
        # Clear the previous centers list


# Release video capture and close all windows
cap.release()
result.release()
cv2.destroyAllWindows()
