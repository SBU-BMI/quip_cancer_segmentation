import numpy as np
import cv2
import openslide
import os
from PIL import Image
import sys

slide = sys.argv[1]
#id = '17039791'
output = sys.argv[2]

if not os.path.exists(output):
    os.mkdir(output);

oslide = openslide.OpenSlide(slide)
width = oslide.dimensions[0]
height = oslide.dimensions[1]

level = oslide.level_count - 1

scale_down = oslide.level_downsamples[level]
w, h = oslide.level_dimensions[level]

#print('level: ', level)
#print('size: {}, {}'.format(w, h))

patch = oslide.read_region((0, 0), level, (w, h));

slide_id = slide.split('/')[-1].split('.svs')[0]
fname = '{}/{}_mask.png'.format(output, slide_id);
#fname = '{}/{}_mask.png'.format(output, scale_down);
patch.save('{}/{}_resized.png'.format(output, slide_id));

img = cv2.imread('{}/{}_resized.png'.format(output, slide_id), 0)
img = cv2.GaussianBlur(img, (61, 61), 0)
ret, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imwrite(fname, imgf)

oslide.close()

#imgf = cv2.resize(imgf, (0, 0), fx = 0.3, fy = 0.3)
#cv2.imshow('img', imgf)
