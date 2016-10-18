import os
import re
import numpy
from PIL import Image
import cv2


def getDimensions(filename):
    img = cv2.imread(filename)
    height, width, channel = img.shape
    return height, width


paths = ['C:\Users\Ashish\Work\HAR\Mid Term Project']
dimensions = []
for path in paths:
    for root, directories, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(".png"):
                l, w = getDimensions(os.path.join(root, filename))
                dimensions.append([l, w])

A = numpy.asarray(dimensions)
L, W = A.max(axis=0)
print L, W


def padImage(F, M, L, W):
    img = Image.open(F, 'r')
    img_w, img_h = img.size
    background = Image.new('RGBA', (M, M), (0, 0, 0, 255))
    bg_w, bg_h = background.size
    offset = ((bg_w - img_w) / 2, (bg_h - img_h) / 2)
    background.paste(img, offset)
    size = L, W
    background = background.resize(size, Image.ANTIALIAS)
    outputFile = F.replace("png", "jpeg")
    background.save(outputFile)


for path in paths:
    for root, directories, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(".png"):
                l, w = getDimensions(os.path.join(root, filename))
                m = max(l, w)
                padImage(os.path.join(root, filename), m, 256, 256)
