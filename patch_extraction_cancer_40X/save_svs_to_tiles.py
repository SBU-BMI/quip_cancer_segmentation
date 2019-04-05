import numpy as np
import openslide
import sys
import os
from PIL import Image
import datetime
import time
import cv2
from shutil import copyfile as cp
#from stain_norm_python.color_normalize_single_folder import color_normalize_single_folder

slide_name = sys.argv[2] + '/' + sys.argv[1];
output_folder = sys.argv[3] + '/' + sys.argv[1];
patch_size_40X = 2100;
level = 0

start = time.time()
fdone = '{}/extraction_done.txt'.format(output_folder);
if os.path.isfile(fdone):
    print('fdone {} exist, skipping'.format(fdone));
    exit(0);

print('extracting {}'.format(output_folder));

if not os.path.exists(output_folder):
    os.mkdir(output_folder);

try:
    oslide = openslide.OpenSlide(slide_name);
    if openslide.PROPERTY_NAME_MPP_X in oslide.properties:
        mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
    elif "XResolution" in oslide.properties:
        mag = 10.0 / float(oslide.properties["XResolution"]);
    else:
        mag = 10.0 / float(0.254);

    pw = int(patch_size_40X * mag / 40);  # scale patch size from 40X to 'mag'

    width = oslide.dimensions[0];
    height = oslide.dimensions[1];
except:
    print('{}: exception caught'.format(slide_name));
    exit(1);

mask_file = '{}/{}_mask.png'.format(output_folder, sys.argv[1][:-4])
print('mask_file: ', mask_file)
mask = cv2.imread(mask_file, 0)
mask[mask > 0] = 1
scale = height/mask.shape[0]

print('height/width: {}/{}'.format(height, width))
print('mask size: ', mask.shape)

for x in range(1, width, pw):
    for y in range(1, height, pw):
        if x + pw > width:
            pw_x = width - x;
        else:
            pw_x = pw;

        if y + pw > height:
            pw_y = height - y;
        else:
            pw_y = pw;


        fname = '{}/{}_{}_{}_{}.png'.format(output_folder, x, y, pw, patch_size_40X);

        x_m = int(x/scale); y_m = int(y/scale)
        isWhite = np.sum(mask[y_m:y_m + int(pw_y/scale), x_m:x_m + int(pw_x/scale)])/(pw_x*pw_y/scale/scale) > 0.95
        x_resize = int(np.ceil(patch_size_40X * pw_x/pw)); y_resize = int(np.ceil(patch_size_40X * pw_y/pw))

        #print('pw_x/pw_y: {}/{}'.format(pw_x, pw_y))
        #print('x_resize/y_resize: {}/{}'.format(x_resize, y_resize))
        if isWhite:
            # this is white patch, generate dummy patch for fast computation
            dummy = np.ones((y_resize, x_resize, 3))*255
            dummy = dummy.astype(np.uint8)
            cv2.imwrite(fname, dummy)
        else:
            patch = oslide.read_region((x, y), level, (pw_x, pw_y));
            '''
            location (tuple) – (x, y) tuple giving the top left pixel in the level 0 reference frame
            level (int) – the level number
            size (tuple) – (width, height) tuple giving the region size
            '''
            patch = patch.resize((x_resize, y_resize), Image.ANTIALIAS)
            patch.save(fname);

print('Elapsed time: ', (time.time() - start)/60.0)
#open(fdone, 'w').close();

