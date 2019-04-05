import numpy as np
from scipy import misc
from PIL import Image
from skimage.color import hed2rgb, rgb2hed

PS = 224

def data_aug_img(img, mu, sigma, deterministic=False, idraw=-1, jdraw=-1):
    # mirror and flip
    if np.random.rand(1)[0] < 0.5:
        img = img[:, ::-1, :];
    if np.random.rand(1)[0] < 0.5:
        img = img[:, :, ::-1];

    # transpose
    if np.random.rand(1)[0] < 0.5:
        img = img.transpose((0, 2, 1));

    # for testing VGG only
    img = Image.fromarray(img.transpose().astype(np.uint8))
    img= np.array(img.resize((PS, PS), Image.BICUBIC)).astype(np.float32)
    img = img.transpose()

    if np.array(mu).size == 1:
        img_temp = (img / 255.0 - mu) / sigma
    else:
        for i in range(3):
            x = (img[i:i+1, :, :] / 255.0 - mu[i]) / sigma[i]
            if i == 0:
                img_temp = x.copy()
            else:
                img_temp = np.concatenate((img_temp, x))

    return img_temp

def data_aug(X, mu, sigma, deterministic=False, idraw=-1, jdraw=-1):
    Xc = np.zeros(shape=(X.shape[0], X.shape[1], PS, PS), dtype=np.float32);
    Xcopy = X.copy();
    for i in range(len(Xcopy)):
        Xc[i] = data_aug_img(Xcopy[i], mu, sigma, deterministic=deterministic, idraw=idraw, jdraw=jdraw);
    return Xc;

