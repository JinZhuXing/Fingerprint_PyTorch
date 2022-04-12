from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy, optim

import numpy as np
from sklearn.utils import shuffle
import random

from model_net import Model_Net


# global variables
IMAGE_WIDTH_LIMIT = 192
IMAGE_HEIGHT_LIMIT = 192


# data generator definition
class DataGenerator(Dataset):
    # initialize data
    def __init__(self, image_width, image_height, dataset_path):
        self.img_width = image_width
        self.img_height = image_height
        self.batch_size = 32
        self.shuffle = True
        self.dataset_path = dataset_path

        # load data
        print('Load Dataset...')
        img_np_data = np.load(os.path.join(self.dataset_path, 'img_train.npy'))
        label_np_data = np.load(os.path.join(self.dataset_path, 'label_train.npy'))
        self.len = label_np_data.shape[0]
        self.img_data = from_numpy(img_np_data)
        self.label_data = from_numpy(label_np_data)
        
        # release memory
        print('DataSize: ', img_np_data.shape, label_np_data.shape)
        del img_np_data
        del label_np_data

        print('Finished: ', self.img_data.size(), self.label_data.size())

    def __len__(self):
        # return available data count
        return self.len

    def __getitem__(self, index):
        first_idx = index
        second_idx = np.random.randint(0, self.len)

        # get data
        first_img_data = self.img_data[first_idx]
        second_img_data = self.img_data[second_idx]
        first_label_data = self.label_data[first_idx]
        second_label_data = self.label_data[second_idx]
        if (first_label_data.item() == second_label_data.item()):
            label_data = np.array([1.])
        else:
            label_data = np.array([0.])

        # convert label data as float
        label_data = label_data.astype(np.float32)
        label_data = from_numpy(label_data)
        
        # return data
        return (first_img_data, second_img_data, label_data)


# main process
def main(args):
    image_width = args.image_width
    image_height = args.image_height
    dataset_path = args.dataset_path
    check_point = args.check_point
    save_model = args.save_model
    save_model_path = args.save_model_path
    train_epoch = args.train_epoch
    eval_num = args.eval_num

    # check parameter
    if ((image_width > IMAGE_WIDTH_LIMIT) or (image_height > IMAGE_HEIGHT_LIMIT)):
        print('Image size must be smaller than', IMAGE_WIDTH_LIMIT, 'x', IMAGE_HEIGHT_LIMIT)
        return
    
    # show information
    print('Train start')
    print('\tImage Width: ', image_width)
    print('\tImage Height: ', image_height)
    print('\tDataset Path: ', dataset_path)
    print('\tCheckpoint Path: ', check_point)
    print('\tTrain Epoch: ', train_epoch)

    # prepare model
    model_net = Model_Net(image_width, image_height)

    # prepare data generator
    data_generator = DataGenerator(image_width, image_height, dataset_path)
    train_gen = DataLoader(dataset = data_generator, batch_size = 32, shuffle = True, num_workers = 2)

    # prepare loss and optimizer
    criterion = nn.BCELoss(reduction = 'sum')
    optimizer = optim.SGD(model_net.parameters(), lr = 0.01)

    # train
    for epoch in range(train_epoch):
        for i, data in enumerate(train_gen):
            # get the inputs
            first_img_data, second_img_data, label_data = data

            # forward pass
            label_pred = model_net(first_img_data, second_img_data)

            # compute and print loss
            loss = criterion(label_pred, label_data)
            print(f'Epoch {epoch + 1} | Batch: {i+1} | Loss: {loss.item():.4f}')

            # zero gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# argument parser
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # image size information
    parser.add_argument('--image_width', type = int,
        help = 'Process image width', default = 128)
    parser.add_argument('--image_height', type = int,
        help = 'Process image height', default = 128)
    parser.add_argument('--dataset_path', type = str,
        help = 'Path to fingerprint image dataset', default = '../dataset/')
    parser.add_argument('--check_point', type = str,
        help = 'Path to model checkpoint', default = '../model/checkpoint/')
    parser.add_argument('--save_model', type = int,
        help = 'Only save model from checkpoint', default = 0)
    parser.add_argument('--save_model_path', type = str,
        help = 'Path to model', default = '../model/result/')
    parser.add_argument('--train_epoch', type = int,
        help = 'Train epoch count', default = 10000)
    parser.add_argument('--eval_num', type = int,
        help = 'Evaluation count', default = 100)

    return parser.parse_args(argv)


# main
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))