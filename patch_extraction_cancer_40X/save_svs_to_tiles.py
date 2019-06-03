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
patch_size_40X = 2100;      # 350*8 --> 8*8 smaller non-overlap patches (350x350)
level = 0

if not os.path.exists(sys.argv[3]): os.mkdir(sys.argv[3])

start = time.time()
fdone = '{}/extraction_done.txt'.format(output_folder);
if os.path.isfile(fdone):
    print('fdone {} exist, skipping'.format(fdone));
    exit(0);

print('extracting {}'.format(output_folder));

if not os.path.exists(output_folder): os.mkdir(output_folder);

try:
    oslide = openslide.OpenSlide(slide_name);
    if openslide.PROPERTY_NAME_MPP_X in oslide.properties:
        mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
    elif "XResolution" in oslide.properties:
        mag = 10.0 / float(oslide.properties["XResolution"]);
    elif 'tiff.XResolution' in oslide.properties:   # for Multiplex IHC WSIs, .tiff images
        mag = 10.0 / float(oslide.properties["tiff.XResolution"]);
    else:
        print('[WARNING] mpp value not found. Assuming it is 40X with mpp=0.254!', slide_name);
        mag = 10.0 / float(0.254);

    pw = int(patch_size_40X * mag / 40);  # scale patch size from 40X to 'mag'

    width = oslide.dimensions[0];
    height = oslide.dimensions[1];
except:
    print('{}: exception caught'.format(slide_name));
    exit(1);

print('height/width: {}/{}'.format(height, width))

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


        fname = '{}/{}_{}_{}_{}.png'.format(output_folder, x, y, pw, patch_size_40X)

        patch = oslide.read_region((x, y), 0, (pw_x, pw_y));
        #shahira: skip where the alpha channel is zero
        patch_arr = np.array(patch);
        if(patch_arr[:,:,3].max() == 0):    # this is blank regions
            continue;
        patch = patch.resize((int(patch_size_40X * pw_x / pw), int(patch_size_40X * pw_y / pw)), Image.ANTIALIAS);
        patch.save(fname);

print('Elapsed time: ', (time.time() - start)/60.0)
#open(fdone, 'w').close();

