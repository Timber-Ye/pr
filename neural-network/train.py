# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/7 14:47
# @Author  : 
# @File    : train.py

import argparse
import datetime
import logging
from pathlib import Path
import os
from tqdm import tqdm
from data_util.DataLoader import DataSet, DataLoader
from model import multiple_layer_perceptron as mlp
import numpy as np
from time import sleep

categories = ['w_1', 'w_2', 'w_3']
labels = {cls: i for i, cls in enumerate(categories)}
cat2label = labels
label2cat = {}
for i, cat in enumerate(cat2label.keys()):
    label2cat[i] = cat

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description='Multilayer Perceptron Training Process')
    parser.add_argument('--model', type=str, default='mlp', help='model name [default: mlp]')
    parser.add_argument('--dim', type=int, default=3, help='dimension of input sample [default: 3]')
    parser.add_argument('--hidden', type=int, default=8, help='size of hidden layer [default: 8]')
    parser.add_argument('--epoch', type=int, default=16, help='Epoch to run [default: 16]')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 4]')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--eval_data', type=int, nargs='+', default=None,
                        help='Which samples to use for evaluate [default: None]')

    return parser.parse_args()


def main(args):
    
    def log_string(_str):
        logger.info(_str)
        print(_str)

    def squared_loss(y_hat, y):  # @save
        """均方损失"""
        return (y.reshape(y_hat.shape) - y_hat) ** 2 / 2, y.reshape(y_hat.shape) - y_hat

    '''CREATE DIR'''
    time_str = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    output_dir = Path(ROOT_DIR + '/log/')
    output_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        output_dir = output_dir.joinpath(time_str)
    else:
        output_dir = output_dir.joinpath(args.log_dir)
    output_dir.mkdir(exist_ok=True)
    checkpoints_dir = output_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = output_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Training")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    data_root = ROOT_DIR + '/samples.csv'
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    NUM_CLASSES = len(categories)
    INPUT_DIM = args.dim
    HIDDEN = args.hidden

    print("start loading training data ...")
    training_dataset = DataSet(data_root, eval_sample_idx=args.eval_data, split='train')
    training_data_loader = DataLoader(training_dataset, batch_size, shuffle=True)
    log_string("The number of training data is: %d." % len(training_dataset))

    print("start loading validating data ...")
    validating_dataset = DataSet(data_root, eval_sample_idx=args.eval_data, split='validate')
    validating_data_loader = DataLoader(validating_dataset, 1, shuffle=True)
    log_string("The number of validating data is: %d." % len(validating_dataset))

    net = mlp.get_model(INPUT_DIM, NUM_CLASSES, HIDDEN)

    global_epoch = 0  # checkpoint TODO
    for epoch in range(args.epoch):
        lr = args.learning_rate
        if epoch % 100 == 99:
            log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
            log_string('Learning rate:%f' % lr)
        num_batches = len(training_data_loader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0

        for i, (features, target) in tqdm(enumerate(training_data_loader), delay=3, colour='blue',
                                          total=len(training_data_loader), smoothing=0.9):
            pred_target = net(features)
            loss, delta = squared_loss(pred_target, target)
            net.backward(delta)
            net.update(learning_rate)

            pred_label = np.argmax(pred_target, axis=1)
            label = np.argmax(target, axis=1)
            correct = np.sum(pred_label == label)
            total_correct += correct
            total_seen += batch_size
            loss_sum += loss.sum()

        if epoch % 200 == 199:
            sleep(0.3)
            log_string('Training mean loss: %f' % (loss_sum / num_batches))
            log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        # if epoch % 5 == 0:
        #     logger.info('Save model...')
        #     savepath = str(checkpoints_dir) + '/model.pth'
        #     log_string('Saving at %s' % savepath)
        #     state = {
        #         'epoch': epoch,
        #         'model_state_dict': net.state_dict(),
        #     }
        #     log_string('Saving model....')
        # TODO

            # test
            total_correct = 0
            for i, (features, target) in tqdm(enumerate(validating_data_loader), delay=3, colour='red',
                                              total=len(validating_data_loader), smoothing=0.9, leave=True):
                pred_target = net(features)
                pred_label = np.argmax(pred_target, axis=1)
                label = np.argmax(target, axis=1)
                correct = np.sum(pred_label == label)
                total_correct += correct

            log_string('Validating accuracy: %f\n' % (total_correct / float(len(validating_dataset))))

        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)