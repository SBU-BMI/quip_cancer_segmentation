import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import PIL.Image as Image
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable
import os
import sys
import torch.nn as nn
import cv2
# from skimage.color import hed2rgb, rgb2hed

APS = 400         # for resnet
# APS = 380       # for pnasnet

mean = [0.7238, 0.5716, 0.6779]
std = [0.1120, 0.1459, 0.1089]

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

def load_data_folder(classn, folder, is_train, color = None, mask_path = ''): # only load the image filename and the labels
    img_pos = []
    img_neg = []
    lines = [line.rstrip('\n') for line in open(folder + '/label.txt')]
    no_pos = 0
    no_neg = 0
    for line in lines:
        img = line.split()[0]
        # change the label threshold to generate labels
        if int(line.split()[1]) < -1: continue

        lab = np.array([int(int(line.split()[1]) > 0)])       # class lymphocyte

        # PC_058_0_1-17005-12805-2400-10X-0-macenko.png
        if color != 'none':
            img = img.split('.png')[0] + '_' + color + '.png'

        # check is the segmentation mask available:
        if mask_path != '':
            seg_file = os.path.join(folder, img.split('.png')[0] + '_reinhard_segment.png')
            if not os.path.exists(seg_file):
                print('file not exist: ', seg_file)
                continue

        img_file = folder + '/' + img
        if not os.path.isfile(img_file):
            print('file not exist: ', img_file)
            continue

        if lab > 0:
            img_pos.append((img_file, lab))
        else:
            img_neg.append((img_file, lab))
    return img_pos, img_neg

def load_data_split(classn, folders, is_train, color = None, mask_path = ''):
    X_pos = []
    X_neg = []
    for folder in folders:
        img_pos, img_neg = load_data_folder(classn, folder, is_train, color = color, mask_path = '')
        X_pos += img_pos
        X_neg += img_neg
    return X_pos, X_neg

def shuffle_data(data, N_limit = 1): # data is a list
    rands = np.random.permutation(len(data))
    out = []
    count = 0
    if N_limit == 1: N_limit = len(data)
    for i in rands:
        out.append(data[i])
        count += 1
        if count == N_limit:
            break
    return out

def load_imgs_files(classn = 1, dataset_list = '', training_data_path = '', color = None, mask_path = ''):
    img_test_pos = []
    img_test_neg = []
    img_train_pos = []
    img_train_neg = []
    lines = [line.rstrip('\n') for line in open(dataset_list)]
    valid_i = 0
    for line in lines:
        split_folders = [training_data_path + "/" + s for s in line.split()]
        if valid_i == 0:
            # testing data
            X_pos, X_neg = load_data_split(classn, split_folders, False, color = color, mask_path = '')
            img_test_pos += X_pos
            img_test_neg += X_neg
        else:
            # training dataX_pos
            X_pos, X_neg = load_data_split(classn, split_folders, True, color = color, mask_path = '')
            img_train_pos += X_pos
            img_train_neg += X_neg
        valid_i += 1

    # ========== shuffle train_data, no need to shuffle test data ========
    # N_pos = len(img_train_pos)
    # N_neg = len(img_train_neg)
    # if N_neg > N_pos:
    #     img_train_neg = shuffle_data(img_train_neg, N_pos)

#    img_train_pos = shuffle_data(img_train_pos, min(30000, len(img_train_pos)))
#    img_train_neg = shuffle_data(img_train_neg, min(70000, len(img_train_neg)))

    img_trains = img_train_pos + img_train_neg

    #img_trains = shuffle_data(img_trains, len(img_trains))

    # ==== testing data ====
    img_vals = img_test_pos + img_test_neg
    img_vals = shuffle_data(img_vals)

    print("training set loaded, pos: {}; neg: {}".format(len(img_train_pos), len(img_train_neg)))
    print("val set, pos: {}; neg: {}".format(len(img_test_pos), len(img_test_neg)))
    return img_trains, img_vals

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def get_mean_and_std_batch(dataset, bs = 4096):
    pop_mean = []
    pop_std0 = []
    pop_std1 = []
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=8)
    for i, data in enumerate(dataloader, 0):
        # shape (batch_size, 3, height, width)
        print('{}/{}'.format(i, len(dataloader)))
        sys.stdout.flush()
        numpy_image, _ = data
        numpy_image = numpy_image.numpy()

        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std0 = np.std(numpy_image, axis=(0, 2, 3))
        batch_std1 = np.std(numpy_image, axis=(0, 2, 3), ddof=1)
        print(batch_mean, batch_std0, batch_std1)

        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        pop_std1.append(batch_std1)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    pop_std1 = np.array(pop_std1).mean(axis=0)
    return pop_mean, pop_std0, pop_std1


class data_loader(Dataset):
    """
    Dataset to read image and label for training
    """
    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform = transform
    def __getitem__(self, index):
        img = self.imgs[index]
        lab = np.array([int(int(img[1]) > 0)])[0]
        png = Image.open(img[0]).convert('RGB')     # ori: RGB, do not convert to numpy, keep it as PIL image to apply transform

        #png = np.array(png)
        #png = cv2.cvtColor(png, cv2.COLOR_RGB2YUV)     # for HASHI
        #png = Image.fromarray(png.astype(np.uint8))


        # if png.size[1] != 400:
        #     png = png.resize((400, 400), Image.ANTIALIAS)

        #png = np.array(png)
        #if (png.shape[1] >= APS):
        #    center = int(png.shape[1]/2)
        #    png = png[center - APS//2:center + APS//2, center - APS//2:center + APS//2, :]
        #png = Image.fromarray(png.astype('uint8'), 'RGB')

        if self.transform:
            png = self.transform(png)

        ## visualize the augmentation...
        #png2 = png.numpy().transpose()
        #png2 = cv2.cvtColor(png2, cv2.COLOR_RGB2BGR)
        #png2 = (png2 * std + mean) * 255
        #png2 = png2.astype(np.uint8)
        #cv2.imshow('img input: ', png2)
        #cv2.waitKey(10)

        return png, lab

    def __len__(self):
        return len(self.imgs)


# class data_loader_4channels(Dataset):
#     """
#     Dataset to read image and label for training
#     """
#     def __init__(self, imgs, transform=None, mask_path = '', isValidation = False):
#         self.imgs = imgs
#         self.transform = transform
#         self.mask_path = mask_path
#         self.isValidation = isValidation
#     def __getitem__(self, index):
#         img = self.imgs[index]
#         lab = np.array([int(int(img[1]) > 0)])[0]
#         png = Image.open(img[0])     # size: wxhx3 ori: RGB, do not convert to numpy, keep it as PIL image to apply transform
#
#         # seg_file = os.path.join(self.mask_path, img[0].split('/')[-1].split('.png')[0] + '_reinhard_segmenr.png')
#         seg_file = img[0].split('.png')[0] + '_reinhard_segment.png'
#         seg = Image.open(seg_file)
#
#         png = png.resize((256, 256), Image.ANTIALIAS).convert('RGB')        # resize desized size
#         seg = seg.resize((256, 256), Image.ANTIALIAS).convert('L')
#
#         png = np.array(png).transpose()
#         seg = np.array(seg).transpose()
#
#         # seg_t = seg.copy()
#         # seg[seg_t < 100] = 0
#         # seg[seg_t >= 100] = 255
#
#         seg = np.expand_dims(seg, axis=0)
#         png, seg = data_aug_img_mask(png, seg, deterministic = self.isValidation)
#
#         # cv2.imshow(img[0].split('/')[-1].split('.png')[0], seg_t)
#         # cv2.waitKey()
#
#         png = np.concatenate((png, seg), axis = 0)
#
#         return png, lab
#
#     def __len__(self):
#         return len(self.imgs)
#
#
# class data_loader_data_aug(Dataset):
#     """
#     Dataset to read image and label for training
#     """
#     def __init__(self, imgs, transform=None, mask_path = '', isValidation = False):
#         self.imgs = imgs
#         self.transform = transform
#         self.mask_path = mask_path
#         self.isValidation = isValidation
#     def __getitem__(self, index):
#         img = self.imgs[index]
#         lab = np.array([int(int(img[1]) > 0)])[0]
#         png = Image.open(img[0])     # size: wxhx3 ori: RGB, do not convert to numpy, keep it as PIL image to apply transform
#
#         png = png.resize((256, 256), Image.ANTIALIAS).convert('RGB')        # resize desized size
#         png = np.array(png).transpose()
#         png = data_aug_img(png, deterministic=self.isValidation)
#
#         return png, lab
#
#     def __len__(self):
#         return len(self.imgs)


class data_loader_visualize(Dataset):
    """
    Dataset to read image and label for training
    """
    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform = transform
    def __getitem__(self, index):
        img = self.imgs[index]
        lab = np.array([int(int(img[1]) > 0)])[0]
        png = Image.open(img[0]).convert('RGB')     # ori: RGB, do not convert to numpy, keep it as PIL image to apply transform

        # if png.size[1] != 400:
        #     png = png.resize((400, 400), Image.ANTIALIAS)

        #png = np.array(png)
        #if (png.shape[1] >= APS):
        #    center = int(png.shape[1]/2)
        #    png = png[center - APS//2:center + APS//2, center - APS//2:center + APS//2, :]
        #png = Image.fromarray(png.astype('uint8'), 'RGB')

        if self.transform:
            png = self.transform(png)

        ## visualize the augmentation...
        #png2 = png.numpy().transpose()
        #png2 = cv2.cvtColor(png2, cv2.COLOR_RGB2BGR)
        #png2 = (png2 * std + mean) * 255
        #png2 = png2.astype(np.uint8)
        #cv2.imshow('img input: ', png2)
        #cv2.waitKey(10)

        return png, lab, img[0]

    def __len__(self):
        return len(self.imgs)

class data_loader_noisy(Dataset):
    """
    Dataset to read image and label for training noisy labels dataset
    """
    def __init__(self, imgs, args, transform=None):
        self.imgs = imgs
        self.transform = transform
        self.labels = np.zeros(len(self.imgs), dtype=np.int32)
        self.soft_labels = np.zeros((len(self.imgs), 2), dtype=np.float32)
        self.prediction = np.zeros((len(self.imgs), args.num_epochs, 2), dtype=np.float32)
        self.labels_ori = np.zeros(len(self.imgs), dtype=np.int32)
        self.count = 0
        self.args = args

        for i, data in enumerate(self.imgs):
            img, label = data
            self.labels[i] = np.array([int(int(label) > 0)])[0]
            self.labels_ori[i] = self.labels[i]
            self.soft_labels[i][self.labels[i]] = 1.

    def __getitem__(self, index):
        img = self.imgs[index]

        png = Image.open(img[0]).convert('RGB')     # do not convert to numpy, keep it as PIL image to apply transform

        # png = np.array(png)
        # if (png.shape[1] >= APS):
        #     center = int(png.shape[1]/2)
        #     png = png[center - APS//2:center + APS//2, center - APS//2:center + APS//2, :]
        # png = Image.fromarray(png.astype('uint8'), 'RGB')

        if self.transform:
            png = self.transform(png)

        ## visualize the augmentation...
        # png = png.numpy().transpose()
        # png = (png * sigma + mu) * 255
        # png = png.astype(np.uint8)
        # print('size of png: ', png.shape)
        # cv2.imshow('img input: ', png)
        # cv2.waitKey(5000)

        return png, self.labels[index], self.soft_labels[index], index

    def __len__(self):
        return len(self.imgs)

    def label_update(self, results):
        # While updating the noisy label y_i by the probability s,
        # we used the average output probability of the network of the past 5 epochs as s.
        # idx = (self.count - 1) % self.prediction.shape[1]
        self.prediction[:, self.count] = results
        self.count += 1

        if self.count >= self.args.begin and self.count <= self.args.stop:
            self.soft_labels = self.prediction[:, 0:self.count].mean(axis=1)
            self.labels = np.argmax(self.soft_labels, axis=1).astype(np.int32)

        if self.count == self.args.stop:
            if not (os.path.isdir(self.args.out)): os.system('mkdir ' + self.args.out)
            np.save('{}/labels_last.npy'.format(self.args.out), self.labels)
            np.save('{}/soft_labels_last.npy'.format(self.args.out), self.soft_labels)

# def data_aug_img(img, deterministic=False):
#     # crop
#     APS = 256
#     PS = 224
#     MARGIN = 0
#     icut = APS - PS;
#     jcut = APS - PS;
#     if deterministic:
#         ioff = int(icut // 2);
#         joff = int(jcut // 2);
#     else:
#         ioff = np.random.randint(MARGIN, icut + 1 - MARGIN);
#         joff = np.random.randint(MARGIN, jcut + 1 - MARGIN);
#     img = img[:, ioff : ioff+PS, joff : joff+PS];
#
#     # adjust color
#     if not deterministic:
#         adj_add = np.array([[[0.07, 0.07, 0.007]]], dtype=np.float32);
#         img = np.clip(hed2rgb( \
#                 rgb2hed(img.transpose((2, 1, 0)) / 255.0) + np.random.uniform(-1.0, 1.0, (1, 1, 3))*adj_add \
#               ).transpose((2, 1, 0))*255.0, 0.0, 255.0);
#
#     if not deterministic:
#         adj_range = 0.05;
#         adj_add = 3;
#         rgb_mean = np.mean(img, axis=(1,2), keepdims=True).astype(np.float32);
#         adj_magn = np.random.uniform(1 - adj_range, 1 + adj_range, (3, 1, 1)).astype(np.float32);
#         img = np.clip((img-rgb_mean)*adj_magn + rgb_mean + np.random.uniform(-1.0, 1.0, (3, 1, 1))*adj_add, 0.0, 255.0);
#
#     # mirror and flip
#     if not deterministic:
#         if np.random.rand(1)[0] < 0.5:
#             img = img[:, ::-1, :];
#         if np.random.rand(1)[0] < 0.5:
#             img = img[:, :, ::-1];
#
#     # transpose
#     if not deterministic:
#         if np.random.rand(1)[0] < 0.5:
#             img = img.transpose((0, 2, 1))
#
#     img = img/255.0
#
#     return img
#
# def data_aug_img_mask(img, msk, deterministic=False):
#     # crop
#     APS = 256
#     PS = 224
#     MARGIN = 0
#     icut = APS - PS;
#     jcut = APS - PS;
#     if deterministic:
#         ioff = int(icut // 2);
#         joff = int(jcut // 2);
#     else:
#         ioff = np.random.randint(MARGIN, icut + 1 - MARGIN);
#         joff = np.random.randint(MARGIN, jcut + 1 - MARGIN);
#     img = img[:, ioff : ioff+PS, joff : joff+PS];
#     msk = msk[:, ioff : ioff+PS, joff : joff+PS];
#
#     # adjust color
#     if not deterministic:
#         adj_add = np.array([[[0.07, 0.07, 0.007]]], dtype=np.float32);
#         img = np.clip(hed2rgb( \
#                 rgb2hed(img.transpose((2, 1, 0)) / 255.0) + np.random.uniform(-1.0, 1.0, (1, 1, 3))*adj_add \
#               ).transpose((2, 1, 0))*255.0, 0.0, 255.0);
#
#     if not deterministic:
#         adj_range = 0.05;
#         adj_add = 3;
#         rgb_mean = np.mean(img, axis=(1,2), keepdims=True).astype(np.float32);
#         adj_magn = np.random.uniform(1 - adj_range, 1 + adj_range, (3, 1, 1)).astype(np.float32);
#         img = np.clip((img-rgb_mean)*adj_magn + rgb_mean + np.random.uniform(-1.0, 1.0, (3, 1, 1))*adj_add, 0.0, 255.0);
#
#     # mirror and flip
#     if not deterministic:
#         if np.random.rand(1)[0] < 0.5:
#             img = img[:, ::-1, :];
#             msk = msk[:, ::-1, :];
#         if np.random.rand(1)[0] < 0.5:
#             img = img[:, :, ::-1];
#             msk = msk[:, :, ::-1];
#
#     # transpose
#     if not deterministic:
#         if np.random.rand(1)[0] < 0.5:
#             img = img.transpose((0, 2, 1))
#             msk = msk.transpose((0, 2, 1))
#
#     img = img/255.0
#     msk = msk/255.0
#
#     return img, msk

def parallelize_model(model):
    if torch.cuda.is_available():
        model = model.cuda()
        # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        cudnn.benchmark = True
    return model

def unparallelize_model(model):
    try:
        while 1:
            # to avoid nested dataparallel problem
            model = model.module
    except AttributeError:
        pass
    return model

def net_frozen(args, model, init_lr, frozen_layer):
    print('********************************************************')
    model.frozen_until(frozen_layer)
    if args.trainer.lower() == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                lr=init_lr, weight_decay=args.weight_decay)
    elif args.trainer.lower() == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                lr=init_lr,  weight_decay=args.weight_decay)
    print('********************************************************')
    return model, optimizer

def cvt_to_gpu(X):
    return Variable(X.cuda()) if torch.cuda.is_available() \
    else Variable(X)

class HASHI(nn.Module):
    def __init__(self):
        super(HASHI, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=8, stride=1, padding=0, bias=False)      # out 256x94x94
        self.relu = nn.ReLU(inplace=True)
        self.L2pool = nn.LPPool2d(2, 2, stride=2)        # out 256x47x47
        self.fc = nn.Linear(256*47*47, 2)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #       nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #    elif isinstance(m, nn.BatchNorm2d):
        #        nn.init.constant_(m.weight, 1)
        #        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.L2pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)
