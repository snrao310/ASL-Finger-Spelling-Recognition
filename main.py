# The main code

import cv2
import numpy
import math
import os
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from PIL import Image
import keras

# path to the dataset
paths = ['/home/snrao/IDE/PycharmProjects/ASL Finger Spelling Recognition/asl_dataset']
TOTAL_DATASET = 2515
x_train = []  # training lists
y_train = []
nb_classes = 10  # number of classes
img_rows, img_cols = 256, 256  # size of training images
img_channels = 3  # BGR channels
batch_size = 32
nb_epoch = 25  # iterations for training
data_augmentation = True

# dictionary for classes from char to numbers
classes1 = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'a': 10,
    'b': 11,
    'c': 12,
    'd': 13,
    'e': 14,
    'f': 15,
    'g': 16,
    'h': 17,
    'i': 18,
    'j': 19,
    'k': 20,
    'l': 21,
    'm': 22,
    'n': 23,
    'o': 24,
    'p': 25,
    'q': 26,
    'r': 27,
    's': 28,
    't': 29,
    'u': 30,
    'v': 31,
    'w': 32,
    'x': 33,
    'y': 34,
    'z': 35,
}
classes = {
    'a': 0,
    'b': 1,
    'c': 2,
    'e': 3,
    'h': 4,
    'o': 5,
    'r': 6,
    'u': 7,
    'w': 8,
    'y': 9,
}


# load the dataset and populate xtrain and ytrain
def load_data_set():
    for path in paths:
        for root, directories, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(".jpeg"):
                    fullpath = os.path.join(root, filename)
                    img = load_img(fullpath)
                    img = img_to_array(img)
                    x_train.append(img)
                    t = fullpath.rindex('/')
                    fullpath = fullpath[0:t]
                    n = fullpath.rindex('/')
                    y_train.append(classes[fullpath[n + 1:t]])



# create a model for training and return the model
def make_network(x_train):
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model


# traiing model which was created
def train_model(model, X_train, Y_train):
    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch)


# loads data set, converts the triaining arrays into required formats of numpy arrays and calls make_network to
# create a model and then calls train_model to train it and then saves the model in disk. OR just loads the model
# from disk.
def trainData():
    load_data_set()
    a = numpy.asarray(y_train)
    y_train_new = a.reshape(a.shape[0], 1)

    X_train = numpy.asarray(x_train).astype('float32')
    X_train = X_train / 255.0
    Y_train = np_utils.to_categorical(y_train_new, nb_classes)

    # run this if model is not saved.
    model = make_network(numpy.asarray(x_train))
    train_model(model,X_train,Y_train)
    model.save('/home/snrao/IDE/PycharmProjects/ASL Finger Spelling Recognition/keras1.model')

    # run this if model is already saved on disk.
    # model = keras.models.load_model('/home/snrao/IDE/PycharmProjects/ASL Finger Spelling Recognition/keras.model')

    return model


model = trainData()


# called from main, when gesture is recognized. The gesture image is cropped and sent to this function.
def identifyGesture(handTrainImage):
    # saving the sent image for checking
    # cv2.imwrite("/home/snrao/IDE/PycharmProjects/ASL Finger Spelling Recognition/a0.jpeg", handTrainImage)

    # converting the image to same resolution as training data by padding to reach 1:1 aspect ration and then
    # resizing to 256x256. Same is done with training data in preprocess_image.py. Opencv image is first
    # converted to Pillow image to do this.
    handTrainImage = cv2.cvtColor(handTrainImage, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(handTrainImage)
    img_w, img_h = img.size
    M = max(img_w, img_h)
    background = Image.new('RGB', (M, M), (0, 0, 0))
    bg_w, bg_h = background.size
    offset = ((bg_w - img_w) / 2, (bg_h - img_h) / 2)
    background.paste(img, offset)
    size =256,256
    background = background.resize(size, Image.ANTIALIAS)

    # saving the processed image for checking.
    # background.save("/home/snrao/IDE/PycharmProjects/ASL Finger Spelling Recognition/a.jpeg")

    # get image as numpy array and predict using model
    open_cv_image = numpy.array(background)
    background = open_cv_image.astype('float32')
    background = background / 255
    background = background.reshape((1,) + background.shape)
    predictions = model.predict_classes(background)

    # print predicted class and get the class name (character name) for the given class number and return it
    print predictions
    key = (key for key, value in classes.items() if value == predictions[0]).next()
    return key


def nothing(x):
    pass


# Create a window to display the camera feed
cv2.namedWindow('Camera Output')
cv2.namedWindow('Hand')
cv2.namedWindow('HandTrain')

# TrackBars for fixing skin color of the person
cv2.createTrackbar('B for min', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('G for min', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('R for min', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('B for max', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('G for max', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('R for max', 'Camera Output', 0, 255, nothing)

# Default skin color values in natural lighting
# cv2.setTrackbarPos('B for min','Camera Output',52)
# cv2.setTrackbarPos('G for min','Camera Output',128)
# cv2.setTrackbarPos('R for min','Camera Output',0)
# cv2.setTrackbarPos('B for max','Camera Output',255)
# cv2.setTrackbarPos('G for max','Camera Output',140)
# cv2.setTrackbarPos('R for max','Camera Output',146)

# Default skin color values in indoor lighting
cv2.setTrackbarPos('B for min', 'Camera Output', 0)
cv2.setTrackbarPos('G for min', 'Camera Output', 130)
cv2.setTrackbarPos('R for min', 'Camera Output', 103)
cv2.setTrackbarPos('B for max', 'Camera Output', 255)
cv2.setTrackbarPos('G for max', 'Camera Output', 182)
cv2.setTrackbarPos('R for max', 'Camera Output', 130)

# Get pointer to video frames from primary device
videoFrame = cv2.VideoCapture(0)

# Process the video frames
keyPressed = -1  # -1 indicates no key pressed. Can press any key to exit

# cascade xml file for detecting palm. Haar classifier
palm_cascade = cv2.CascadeClassifier('palm.xml')

# previous values of cropped variable
x_crop_prev, y_crop_prev, w_crop_prev, h_crop_prev = 0, 0, 0, 0

# previous cropped frame if we need to compare histograms of previous image with this to see the change.
# Not used but may need later.
_, prevHandImage = videoFrame.read()

# previous frame contour of hand. Used to compare with new contour to find if gesture has changed.
prevcnt = numpy.array([], dtype=numpy.int32)

# gesture static increments when gesture doesn't change till it reaches 10 (frames) and then resets to 0.
# gesture detected is set to 10 when gesture static reaches 10."Gesture Detected is displayed for next
# 10 frames till gestureDetected decrements to 0.
gestureStatic = 0
gestureDetected = 0

while keyPressed < 0:  # any key pressed has a value >= 0

    # Getting min and max colors for skin
    min_YCrCb = numpy.array([cv2.getTrackbarPos('B for min', 'Camera Output'),
                             cv2.getTrackbarPos('G for min', 'Camera Output'),
                             cv2.getTrackbarPos('R for min', 'Camera Output')], numpy.uint8)
    max_YCrCb = numpy.array([cv2.getTrackbarPos('B for max', 'Camera Output'),
                             cv2.getTrackbarPos('G for max', 'Camera Output'),
                             cv2.getTrackbarPos('R for max', 'Camera Output')], numpy.uint8)

    # Grab video frame, Decode it and return next video frame
    readSucsess, sourceImage = videoFrame.read()

    # Convert image to YCrCb
    imageYCrCb = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2YCR_CB)
    imageYCrCb = cv2.GaussianBlur(imageYCrCb, (5, 5), 0)

    # Find region with skin tone in YCrCb image
    skinRegion = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

    # Do contour detection on skin region
    _, contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sorting contours by area. Largest area first.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # get largest contour and compare with largest contour from previous frame.
    # set previous contour to this one after comparison.
    cnt = contours[0]
    ret = cv2.matchShapes(cnt, prevcnt, 2, 0.0)
    prevcnt = contours[0]

    # once we get contour, extract it without background into a new window called handTrainImage
    stencil = numpy.zeros(sourceImage.shape).astype(sourceImage.dtype)
    color = [255, 255, 255]
    cv2.fillPoly(stencil, [cnt], color)
    handTrainImage = cv2.bitwise_and(sourceImage, stencil)

    # if comparison returns a high value (shapes are different), start gestureStatic over. Else increment it.
    if (ret > 0.70):
        gestureStatic = 0
    else:
        gestureStatic += 1

    # crop coordinates for hand.
    x_crop, y_crop, w_crop, h_crop = cv2.boundingRect(cnt)

    # place a rectange around the hand.
    cv2.rectangle(sourceImage, (x_crop, y_crop), (x_crop + w_crop, y_crop + h_crop), (0, 255, 0), 2)

    # if the crop area has changed drastically form previous frame, update it.
    if (abs(x_crop - x_crop_prev) > 50 or abs(y_crop - y_crop_prev) > 50 or
                abs(w_crop - w_crop_prev) > 50 or abs(h_crop - h_crop_prev) > 50):
        x_crop_prev = x_crop
        y_crop_prev = y_crop
        h_crop_prev = h_crop
        w_crop_prev = w_crop

    # create crop image
    handImage = sourceImage.copy()[max(0, y_crop_prev - 50):y_crop_prev + h_crop_prev + 50,
                max(0, x_crop_prev - 50):x_crop_prev + w_crop_prev + 50]

    # Training image with black background
    handTrainImage = handTrainImage[max(0, y_crop_prev - 15):y_crop_prev + h_crop_prev + 15,
                     max(0, x_crop_prev - 15):x_crop_prev + w_crop_prev + 15]

    # if gesture is static for 10 frames, set gestureDetected to 10 and display "gesture detected"
    # on screen for 10 frames.
    if gestureStatic == 10:
        gestureDetected = 10;
        print("Gesture Detected")
        letterDetected = identifyGesture(handTrainImage)  # todo: Ashish fill this function to return actual character

    if gestureDetected > 0:
        if (letterDetected != None):
            cv2.putText(sourceImage, letterDetected, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
        gestureDetected -= 1

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
        if count_defects == 0:
            center_of_palm = far
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
        if angle <= 90:
            count_defects += 1
            if count_defects < 5:
                # cv2.circle(sourceImage, far, 5, [0, 0, 255], -1)
                center_of_palm = (far[0] + center_of_palm[0]) / 2, (far[1] + center_of_palm[1]) / 2
        cv2.line(sourceImage, start, end, [0, 255, 0], 2)
    # cv2.circle(sourceImage, avr, 10, [255, 255, 255], -1)


    # drawing the largest contour
    cv2.drawContours(sourceImage, contours, 0, (0, 255, 0), 1)

    # Display the source image and cropped image
    cv2.imshow('Camera Output', sourceImage)
    cv2.imshow('Hand', handImage)
    cv2.imshow('HandTrain', handTrainImage)

    # Check for user input to close program
    keyPressed = cv2.waitKey(30)  # wait 30 miliseconds in each iteration of while loop

# Close window and camera after exiting the while loop
cv2.destroyWindow('Camera Output')
videoFrame.release()
