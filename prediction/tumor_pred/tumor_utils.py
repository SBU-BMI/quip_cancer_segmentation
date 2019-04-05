import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import PIL.Image as Image
# import data_aug as DA
#import cv2

import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable
import os
import torch.nn as nn

def get_sizes():
    APS = 360;
    PS = 224;  # original 100
    MARGIN = 58;  # original 90
    mu = [0.74194884, 0.5952509, 0.696611]
    sigma = [0.18926544, 0.24084978, 0.18950582]
    return APS, PS, MARGIN, mu, sigma

APS, PS, MARGIN, mu, sigma = get_sizes()

def load_data_folder(classn, folder, is_train, color = None): # only load the image filename and the labels
    img_pos = []
    img_neg = []
    lines = [line.rstrip('\n') for line in open(folder + '/label.txt')]
    for line in lines:
        img = line.split()[0]
        # change the label threshold to generate labels
        lab = np.array([int(int(line.split()[1]) > 0)])       # class lymphocyte
        # PC_058_0_1-17005-12805-2400-10X-0-macenko.png
        if color != 'none':
            img = img.split('.png')[0] + '-' + color + '.png'

        img_file = folder + '/' + img
        if not os.path.isfile(img_file):
            print('file not exist: ', img_file)
            continue

        if lab > 0:
            img_pos.append((img_file, lab))
        else:
            img_neg.append((img_file, lab))
    return img_pos, img_neg

def load_data_split(classn, folders, is_train, color = None):
    X_pos = []
    X_neg = []
    for folder in folders:
        img_pos, img_neg = load_data_folder(classn, folder, is_train, color = color);
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

def load_imgs_files(classn = 1, dataset_list = '', training_data_path = '', color = None):
    img_test_pos = []
    img_test_neg = []
    img_train_pos = []
    img_train_neg = []
    lines = [line.rstrip('\n') for line in open(dataset_list)]
    valid_i = 0;
    for line in lines:
        split_folders = [training_data_path + "/" + s for s in line.split()]
        if valid_i == 0:
            # testing data
            X_pos, X_neg = load_data_split(classn, split_folders, False, color = color)
            img_test_pos += X_pos
            img_test_neg += X_neg
        else:
            # training dataX_pos
            X_pos, X_neg = load_data_split(classn, split_folders, True, color = color)
            img_train_pos += X_pos
            img_train_neg += X_neg
        valid_i += 1;

    # ========== shuffle train_data, no need to shuffle test data ========
    # N_pos = len(img_train_pos)
    # N_neg = len(img_train_neg)
    # if N_neg > N_pos:
    #     img_train_neg = shuffle_data(img_train_neg, N_pos)

    img_trains = img_train_pos + img_train_neg
    img_trains = shuffle_data(img_trains, len(img_trains))

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
        png = Image.open(img[0]).convert('RGB')     # do not convert to numpy, keep it as PIL image to apply transform

        png = np.array(png)
        if (png.shape[1] >= 400):
            center = int(png.shape[1]/2)
            png = png[center - APS//2:center + APS//2, center - APS//2:center + APS//2, :]

        png = Image.fromarray(png.astype('uint8'), 'RGB')

        if self.transform:
            png = self.transform(png)

        ## visualize the augmentation...
        # png = png.numpy().transpose()
        # png = (png * sigma + mu) * 255
        # png = png.astype(np.uint8)
        # print('size of png: ', png.shape)
        # cv2.imshow('img input: ', png)
        # cv2.waitKey(5000)

        return png, lab

    def __len__(self):
        return len(self.imgs)


class data_loader_multi_res(Dataset):
    """
    Dataset to read image and label for training
    """
    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform_0 = transform[0]
        self.transform_1 = transform[1]
        self.transform_2 = transform[2]
    def __getitem__(self, index):
        img = self.imgs[index]
        lab = np.array([int(int(img[1]) > 0)])[0]
        png = Image.open(img[0]).convert('RGB')     # do not convert to numpy, keep it as PIL image to apply transform

        png = np.array(png)
        if (png.shape[1] >= 400):
            center = int(png.shape[1]/2)
            png = png[center - APS//2:center + APS//2, center - APS//2:center + APS//2, :]

        png = Image.fromarray(png.astype('uint8'), 'RGB')

        png = self.transform_0(png)
        png_1 = self.transform_1(png)
        png_2 = self.transform_2(png)

        return png_1, png_2, lab

    def __len__(self):
        return len(self.imgs)


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


class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(True), nn.Linear(128, 2))
        self._initialize_weights()

    def forward(self, x):
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class MyResNet(nn.Module):
    def __init__(self, depth, num_classes, pretrained = True):
        super(MyResNet, self).__init__()
        if depth == 18:
            model = models.resnet18(pretrained)
        elif depth == 34:
            model = models.resnet34(pretrained)
        elif depth == 50:
            model = models.resnet50(pretrained)
        elif depth == 101:
            model = models.resnet101(pretrained)
        elif depth == 152:
            model = models.resnet152(pretrained)

        self.num_ftrs = model.fc.in_features
        self.shared = nn.Sequential(*list(model.children())[:-1])
        self.target = nn.Sequential(nn.Linear(self.num_ftrs, num_classes))

    def forward(self, x):
        # pdb.set_trace()
        x = self.shared(x)
        x = torch.squeeze(x)
        return self.target(x)

    def frozen_until(self, to_layer):
        print('Frozen shared part until %d-th layer, inclusive'%to_layer)
        # if to_layer = -1, frozen all
        child_counter = 0
        for child in self.shared.children():
            if child_counter <= to_layer:
                print("child ", child_counter, " was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                # frozen deeper children? check
                # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
            else:
                print("child ", child_counter, " was not frozen")
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1


class baseline_resnet(nn.Module):
    def __init__(self, depth, num_classes, pretrained = True):
        super(baseline_resnet, self).__init__()
        if depth == 18:
            model = models.resnet18(pretrained)
        elif depth == 34:
            model = models.resnet34(pretrained)
        elif depth == 50:
            model = models.resnet50(pretrained)
        elif depth == 101:
            model = models.resnet101(pretrained)
        elif depth == 152:
            model = models.resnet152(pretrained)

        self.num_ftrs = model.fc.in_features
        self.shared = nn.Sequential(*list(model.children())[:-1])
        self.target = nn.Sequential(nn.Linear(self.num_ftrs, num_classes))

    def forward(self, x):
        # pdb.set_trace()
        x = self.shared(x)
        x = torch.squeeze(x)
        return self.target(x)

    def frozen_until(self, to_layer):
        print('Frozen shared part until %d-th layer, inclusive'%to_layer)
        # if to_layer = -1, frozen all
        to_layer = 100
        child_counter = 0
        for child in self.shared.children():
            if child_counter <= to_layer:
                print("child ", child_counter, " was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                # frozen deeper children? check
                # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
            else:
                print("child ", child_counter, " was not frozen")
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1


class baseline_vgg16(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(baseline_vgg16, self).__init__()
        model = models.vgg16(pretrained)

        self.num_ftrs = model.classifier[6].in_features
        self.shared = nn.Sequential(*list(model.features.children()))
        model.classifier[6] = nn.Linear(self.num_ftrs, num_classes)
        self.target = nn.Sequential(*list(model.classifier.children()))

    def forward(self, x):
        # pdb.set_trace()
        x = self.shared(x)
        x = x.view(x.size(0), -1)
        return self.target(x)

    def frozen_until(self, to_layer):
        print('Frozen shared part until %d-th layer, inclusive'%to_layer)
        # if to_layer = -1, frozen all
        to_layer = 100
        child_counter = 0
        for child in self.shared.children():
            if child_counter <= to_layer:
                print("child ", child_counter, " was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                # frozen deeper children? check
                # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
            else:
                print("child ", child_counter, " was not frozen")
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1



class baseline_inception_v3(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(baseline_inception_v3, self).__init__()
        model = models.inception_v3(pretrained)

        self.num_ftrs = model.fc.in_features
        self.shared = nn.Sequential(*list(model.children())[:-1])
        self.target = nn.Sequential(nn.Linear(self.num_ftrs, num_classes))

    def forward(self, x):
        # pdb.set_trace()
        x = self.shared(x)
        x = x.view(x.size(0), -1)
        return self.target(x)

    def frozen_until(self, to_layer):
        print('Frozen shared part until %d-th layer, inclusive'%to_layer)
        # if to_layer = -1, frozen all
        to_layer = 100
        child_counter = 0
        for child in self.shared.children():
            if child_counter <= to_layer:
                print("child ", child_counter, " was frozen")
                for param in child.parameters():
                    param.requires_grad = False
                # frozen deeper children? check
                # https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
            else:
                print("child ", child_counter, " was not frozen")
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1


