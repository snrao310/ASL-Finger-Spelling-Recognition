# Required moduls
import cv2
import numpy
import math

def nothing(x):
    pass

# Constants for finding range of skin color in YCrCb

max_YCrCb = numpy.array([255,173,127],numpy.uint8)
min_YCrCb = numpy.array([0,136,77],numpy.uint8)

# Create a window to display the camera feed
cv2.namedWindow('Camera Output')
cv2.namedWindow('Hand')
cv2.createTrackbar('min1','Camera Output',0,255,nothing)
cv2.setTrackbarPos('min1','Camera Output',52)
cv2.setTrackbarPos('min1','Camera Output',0)
cv2.createTrackbar('min2','Camera Output',0,255,nothing)
cv2.setTrackbarPos('min2','Camera Output',128)
cv2.setTrackbarPos('min2','Camera Output',130)
cv2.createTrackbar('min3','Camera Output',0,255,nothing)
cv2.setTrackbarPos('min3','Camera Output',0)
cv2.setTrackbarPos('min3','Camera Output',103)
cv2.createTrackbar('max1','Camera Output',0,255,nothing)
cv2.setTrackbarPos('max1','Camera Output',255)
cv2.setTrackbarPos('max1','Camera Output',255)
cv2.createTrackbar('max2','Camera Output',0,255,nothing)
cv2.setTrackbarPos('max2','Camera Output',140)
cv2.setTrackbarPos('max2','Camera Output',182)
cv2.createTrackbar('max3','Camera Output',0,255,nothing)
cv2.setTrackbarPos('max3','Camera Output',146)
cv2.setTrackbarPos('max3','Camera Output',130)


# Get pointer to video frames from primary device
videoFrame = cv2.VideoCapture(0)

# Process the video frames
keyPressed = -1 # -1 indicates no key pressed

palm_cascade = cv2.CascadeClassifier('palm.xml')
x1=0
y1=0
w1=0
h1=0

_, prevHandImage=videoFrame.read()
prevcnt=numpy.array([], dtype=numpy.int32)
ab=0

while(keyPressed < 0): # any key pressed has a value >= 0
    min_YCrCb = numpy.array([cv2.getTrackbarPos('min1','Camera Output'),cv2.getTrackbarPos('min2','Camera Output'),cv2.getTrackbarPos('min3','Camera Output')],numpy.uint8)
    max_YCrCb = numpy.array([cv2.getTrackbarPos('max1','Camera Output'),cv2.getTrackbarPos('max2','Camera Output'),cv2.getTrackbarPos('max3','Camera Output')],numpy.uint8)


    # Grab video frame, decode it and return next video frame
    readSucsess, sourceImage = videoFrame.read()

    # Convert image to YCrCb
    imageYCrCb = cv2.cvtColor(sourceImage,cv2.COLOR_BGR2YCR_CB)

    # Find region with skin tone in YCrCb image
    skinRegion = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)

    # Do contour detection on skin region
    _,contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cv2.drawContours(sourceImage, contours, 0, (0, 255, 0), 1)
    cnt = contours[0]
    ret = cv2.matchShapes(cnt, prevcnt, 2, 0.0)
    if(ret>0.72):
        print (ret)
    prevcnt=contours[0]

    x, y, w, h = cv2.boundingRect(cnt)

    cv2.rectangle(sourceImage, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if(abs(x-x1)>50 or abs(y-y1)>50 or abs(w-w1)>50 or abs(h-h1)>50):
        x1=x
        y1=y
        h1=h
        w1=w
    handImage = sourceImage[max(0,y1-50):y1+h1+50, max(0,x1-50):x1+w1+50]
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


    gray = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2HSV)

    palm = palm_cascade.detectMultiScale(sourceImage)
    for (x, y, w, h) in palm:
        cv2.rectangle(sourceImage, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # roi_gray = gray[y:y + h, x:x + w]
        # roi_color = sourceImage[y:y + h, x:x + w]


    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    count_defects = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        if count_defects==0:
            avr=far
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
        if angle <= 90:
            count_defects += 1
            if count_defects<5:
                # cv2.circle(sourceImage, far, 5, [0, 0, 255], -1)
                avr = (far[0] + avr[0]) / 2, (far[1] + avr[1]) / 2
        cv2.line(sourceImage, start, end, [0, 255, 0], 2)

    # cv2.circle(sourceImage, avr, 10, [255, 255, 255], -1)

    # Draw the contour on the source image
    # for i, c in enumerate(contours):
    #     area = cv2.contourArea(c)
    #     if area > 1000:
    #         cv2.drawContours(sourceImage, contours, i, (0, 255, 0), 3)

    # Display the source image
    cv2.imshow('Camera Output',sourceImage)
    cv2.imshow('Hand', handImage)

    # Check for user input to close program
    keyPressed = cv2.waitKey(30) # wait 1 milisecond in each iteration of while loop

# Close window and camera after exiting the while loop
cv2.destroyWindow('Camera Output')
videoFrame.release()
