import numpy as np
import cv2
import openslide
import os
from PIL import Image
import sys

try:
    parent = sys.argv[1]
    source = sys.argv[2]
except:
    parent = '/data01/shared/hanle/neutrophils.pred/data/patches'
    source = '/data01/shared/hanle/neutrophils.pred/data/svs'


fols = os.listdir(parent)
for fol in fols:
    if not os.path.exists(os.path.join(parent, fol)): continue
    slide = os.path.join(source, fol)

    oslide = openslide.OpenSlide(slide)
    width = oslide.dimensions[0]
    height = oslide.dimensions[1]

    level = oslide.level_count - 1

    scale_down = oslide.level_downsamples[level]
    w, h = oslide.level_dimensions[level]

    #print('level: ', level)
    #print('size: {}, {}'.format(w, h))

    fname = '{}/{}/{}_{}_mask.png'.format(parent, fol, scale_down, fol.split('.svs')[0]);
    if os.path.exists(fname): continue

    print('processing...', fol)
    patch = oslide.read_region((0, 0), level, (w, h));
    patch.save('temp.png');
    img = cv2.imread('temp.png', 0)
    img = cv2.GaussianBlur(img, (61, 61), 0)
    ret, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite(fname, imgf)
    oslide.close()

#imgf = cv2.resize(imgf, (0, 0), fx = 0.3, fy = 0.3)
#cv2.imshow('img', imgf)
