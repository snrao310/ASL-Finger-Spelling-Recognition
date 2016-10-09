# The main code

import cv2
import numpy
import math

#todo: Ashish fill this
def identifyGesture(handImage):
    return "Gesture Detected"

def nothing(x):
    pass


# Create a window to display the camera feed
cv2.namedWindow('Camera Output')
cv2.namedWindow('Hand')

# TrackBars for fixing skin color of the person
cv2.createTrackbar('B for min','Camera Output',0,255,nothing)
cv2.createTrackbar('G for min','Camera Output',0,255,nothing)
cv2.createTrackbar('R for min','Camera Output',0,255,nothing)
cv2.createTrackbar('B for max','Camera Output',0,255,nothing)
cv2.createTrackbar('G for max','Camera Output',0,255,nothing)
cv2.createTrackbar('R for max','Camera Output',0,255,nothing)

# Default skin color values in natural lighting
# cv2.setTrackbarPos('min1','Camera Output',52)
# cv2.setTrackbarPos('min2','Camera Output',128)
# cv2.setTrackbarPos('min3','Camera Output',0)
# cv2.setTrackbarPos('max1','Camera Output',255)
# cv2.setTrackbarPos('max2','Camera Output',140)
# cv2.setTrackbarPos('max3','Camera Output',146)

# Default skin color values in indoor lighting
cv2.setTrackbarPos('B for min','Camera Output',0)
cv2.setTrackbarPos('G for min','Camera Output',130)
cv2.setTrackbarPos('R for min','Camera Output',103)
cv2.setTrackbarPos('B for max','Camera Output',255)
cv2.setTrackbarPos('G for max','Camera Output',182)
cv2.setTrackbarPos('R for max','Camera Output',130)

# Get pointer to video frames from primary device
videoFrame = cv2.VideoCapture(0)

# Process the video frames
keyPressed = -1 # -1 indicates no key pressed. Can press any key to exit

# cascade xml file for detecting palm. Haar classifier
palm_cascade = cv2.CascadeClassifier('palm.xml')

# previous values of cropped variable
x_crop_prev, y_crop_prev, w_crop_prev, h_crop_prev= 0, 0, 0, 0

# previous cropped frame if we need to compare histograms of previous image with this to see the change.
# Not used but may need later.
_, prevHandImage=videoFrame.read()

# previous frame contour of hand. Used to compare with new contour to find if gesture has changed.
prevcnt = numpy.array([], dtype=numpy.int32)

# gesture static increments when gesture doesn't change till it reaches 10 (frames) and then resets to 0.
# gesture detected is set to 10 when gesture static reaches 10."Gesture Detected is displayed for next
# 10 frames till gestureDetected decrements to 0.
gestureStatic=0
gestureDetected=0

while keyPressed < 0: # any key pressed has a value >= 0

    # Getting min and max colors for skin
    min_YCrCb = numpy.array([cv2.getTrackbarPos('B for min','Camera Output'),
                             cv2.getTrackbarPos('G for min','Camera Output'),
                             cv2.getTrackbarPos('R for min','Camera Output')],numpy.uint8)
    max_YCrCb = numpy.array([cv2.getTrackbarPos('B for max','Camera Output'),
                             cv2.getTrackbarPos('G for max','Camera Output'),
                             cv2.getTrackbarPos('R for max','Camera Output')],numpy.uint8)

    # Grab video frame, Decode it and return next video frame
    readSucsess, sourceImage = videoFrame.read()

    # Convert image to YCrCb
    imageYCrCb = cv2.cvtColor(sourceImage,cv2.COLOR_BGR2YCR_CB)

    # Find region with skin tone in YCrCb image
    skinRegion = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)

    # Do contour detection on skin region
    _,contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sorting contours by area. Largest area first and drawing the largest contour.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cv2.drawContours(sourceImage, contours, 0, (0, 255, 0), 1)

    # get largest contour and compare with largest contour from previous frame.
    # set previous contour to this one after comparison.
    cnt = contours[0]
    ret = cv2.matchShapes(cnt, prevcnt, 2, 0.0)
    prevcnt = contours[0]

    # if comparison returns a high value (shapes are different), start gestureStatic over. Else increment it.
    if(ret>0.70):
        gestureStatic=0
    else:
        gestureStatic+=1

    # if gesture is static for 10 frames, set gestureDetected to 10 and display "gesture detected"
    # on screen for 10 frames.
    if gestureStatic==10:
        gestureDetected=10;
        print("Gesture Detected")
        letterDetected=identifyGesture(handImage) #todo: Ashish fill this function to return actual character

    if gestureDetected>0:
        cv2.putText(sourceImage, letterDetected, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
        gestureDetected-=1

    # crop coordinates for hand.
    x_crop, y_crop, w_crop, h_crop = cv2.boundingRect(cnt)

    # place a rectange around the hand.
    cv2.rectangle(sourceImage, (x_crop, y_crop), (x_crop + w_crop, y_crop + h_crop), (0, 255, 0), 2)

    # if the crop area has changed drastically form previous frame, update it.
    if(abs(x_crop-x_crop_prev)>50 or abs(y_crop-y_crop_prev)>50 or
               abs(w_crop-w_crop_prev)>50 or abs(h_crop-h_crop_prev)>50):
        x_crop_prev=x_crop
        y_crop_prev=y_crop
        h_crop_prev=h_crop
        w_crop_prev=w_crop

    # create crop image
    handImage = sourceImage[max(0, y_crop_prev - 50):y_crop_prev + h_crop_prev + 50, max(0, x_crop_prev - 50):x_crop_prev + w_crop_prev + 50]

    # Comparing histograms of this image and previous image to check if the gesture has changed.
    # Not accurate. So switched to contour comparisons.
    # hist1 = cv2.calcHist(handImage, [0, 1, 2], None, [8, 8, 8],
    #                     [0, 256, 0, 256, 0, 256])
    # hist1 = cv2.normalize(hist1,hist1).flatten()
    # hist2 = cv2.calcHist(prevHandImage, [0, 1, 2], None, [8, 8, 8],
    #                     [0, 256, 0, 256, 0, 256])
    # hist2 = cv2.normalize(hist2,hist2).flatten()
    # d = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    # # if d<0.9:
    # print(d)
    # prevHandImage = handImage

    # haar cascade classifier to detect palm and gestures. Not very accurate though.
    # Needs more training to become accurate.
    gray = cv2.cvtColor(handImage, cv2.COLOR_BGR2HSV)
    palm = palm_cascade.detectMultiScale(gray)
    for (x, y, w, h) in palm:
        cv2.rectangle(sourceImage, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # roi_gray = gray[y:y + h, x:x + w]
        roi_color = sourceImage[y:y + h, x:x + w]

    # to show convex hull in the image
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    # counting defects in convex hull. To find center of palm. Center is average of defect points.
    count_defects = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        if count_defects==0:
            center_of_palm=far
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
        if angle <= 90:
            count_defects += 1
            if count_defects<5:
                # cv2.circle(sourceImage, far, 5, [0, 0, 255], -1)
                center_of_palm = (far[0] + center_of_palm[0]) / 2, (far[1] + center_of_palm[1]) / 2
        cv2.line(sourceImage, start, end, [0, 255, 0], 2)
    # cv2.circle(sourceImage, avr, 10, [255, 255, 255], -1)

    # Display the source image and cropped image
    cv2.imshow('Camera Output',sourceImage)
    cv2.imshow('Hand', handImage)

    # Check for user input to close program
    keyPressed = cv2.waitKey(30) # wait 30 miliseconds in each iteration of while loop

# Close window and camera after exiting the while loop
cv2.destroyWindow('Camera Output')
videoFrame.release()
