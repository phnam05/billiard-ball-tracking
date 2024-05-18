# -*- coding: utf-8 -*-
"""
Helper functions for transforming the standard camera view to an overhead cam
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import Indexer




def LoadImage(filename):
    """
    Loads an image file
    """
    #img is loaded in bgr colorspace
    return cv2.imread(filename)


def ToHSV(img):
    """
    Convert an image from BGR to HSV colorspace
    """
    return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)


def GetClothColor(hsv, search_width=30):
    """
    Find the most common HSV values in the image.
    In a well lit image, this will be the cloth
    """

    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    h_max = Indexer.get_index_of_max(hist)[0]

    hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])

    s_max = Indexer.get_index_of_max(hist)[0]

    hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    v_max = Indexer.get_index_of_max(hist)[0]

    # define range of blue color in HSV
    lower_color = np.array([h_max - search_width, s_max - search_width, v_max - search_width])
    upper_color = np.array([h_max + search_width, s_max + search_width, v_max + search_width])

    return lower_color, upper_color


def MaskTableBed(contours):
    """
    Mask out the table bed, assuming that it will be the biggest contour.
    """

    #The largest area should be the table bed    

    areas = []
    for c in contours:
        areas.append(cv2.contourArea(c))

    #return the contour that delineates the table bed
    largest_contour = Indexer.get_index_of_max(areas)
    #(contours[largest_contour[0]])
    return contours[largest_contour[0]]


def distbetween(x1, y1, x2, y2):
    """
    Compute the distance between points (x1,y1) and (x2,y2)
    """

    return np.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))


def Get_UL_Coord(contour, pad=10):
    """
    Get the upper left coordinate of the contour.
    """
    dists = []
    for c in contour:
        dists.append(distbetween(c[0][0], c[0][1], 0, 0))

    return (
        contour[Indexer.get_index_of_min(dists)[0]][0][0] - pad,
        contour[Indexer.get_index_of_min(dists)[0]][0][1] - pad)


def Get_UR_Coord(contour, imgXmax, pad=10):
    """
    Get the upper right coordinate of the contour.
    """
    dists = []
    for c in contour:
        dists.append(distbetween(c[0][0], c[0][1], imgXmax, 0))

    return (
        contour[Indexer.get_index_of_min(dists)[0]][0][0] + pad,
        contour[Indexer.get_index_of_min(dists)[0]][0][1] - pad)


def Get_LL_Coord(contour, imgYmax, pad=10):
    """
    Get the lower left coordinate of the contour.
    """
    dists = []
    for c in contour:
        dists.append(distbetween(c[0][0], c[0][1], 0, imgYmax))

    return (
        contour[Indexer.get_index_of_min(dists)[0]][0][0] - pad,
        contour[Indexer.get_index_of_min(dists)[0]][0][1] + pad)


def Get_LR_Coord(contour, imgXmax, imgYmax, pad=10):
    """
    Get the lower right coordinate of the contour.
    """
    dists = []
    for c in contour:
        dists.append(distbetween(c[0][0], c[0][1], imgXmax, imgYmax))

    return (
        contour[Indexer.get_index_of_min(dists)[0]][0][0] + pad,
        contour[Indexer.get_index_of_min(dists)[0]][0][1] + pad)


def TransformToOverhead(img, contour):
    """
    Get the corner coordinates of the table bed by finding the minumum
    distance to the corners of the image for each point in the contour.
    
    Transform code is built upon code from: http://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/ 
    """

    #get dimensions of image
    height, width, channels = img.shape

    #find the 4 corners of the table bed
    UL = Get_UL_Coord(contour)
    UR = Get_UR_Coord(contour, width)
    LL = Get_LL_Coord(contour, height)
    LR = Get_LR_Coord(contour, width, height)

    #store the coordinates in a numpy array    
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = [UL[0], UL[1]]
    rect[1] = [UR[0], UR[1]]
    rect[2] = [LR[0], LR[1]]
    rect[3] = [LL[0], LL[1]]

    #get the width at the bottom and top of the image
    widthA = distbetween(LL[0], LL[1], LR[0], LR[1])
    widthB = distbetween(UL[0], UL[1], UR[0], UR[1])

    #choose the maximum width 
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = (maxWidth * 2)  #pool tables are twice as long as they are wide

    # construct our destination points which will be used to
    # map the image to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return warp


def GetContours(hsv, lower_color, upper_color, filter_radius):
    """
    Returns the contours generated from the given color range
    """

    # Threshold the HSV image to get only cloth colors
    mask = cv2.inRange(hsv, lower_color, upper_color)

    #use a median filter to get rid of speckle noise
    median = cv2.medianBlur(mask, filter_radius)

    #get the contours of the filtered mask
    #this modifies median in place!
    contours, hierarchy = cv2.findContours(median, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours



