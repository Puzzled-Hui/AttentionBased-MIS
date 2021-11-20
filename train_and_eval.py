# -*- coding: utf-8 -*-
import sys
sys.path.append("E:\\Studyfile_ZMH\\GraduationProject\\code\\AttentionBased-MIS\\")
import time
import SimpleITK as sitk
import numpy as np

from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.utils import test_train_dice,test_test_dice,calculate_average_metrics

# @author ZhangMinghui,Southeast University
#
# Script Function: Train Model for a specific model and dataset.


if __name__ =="__main__":
    opt_train = TrainOptions().parse()
    opt_eval  = TestOptions().parse()
    train_dataset = create_dataset(opt_train)
    test_dataset  = create_dataset(opt_eval)
    visualizer    = Visualizer(opt_train)
    print('The number of training images = %d' % len(train_dataset))
    print('The number of test images = %d' % len(test_dataset))

    model = create_model(opt_train)
    model.setup(opt_train)
    total_iters = 0
    BestDice = -1.0

    for epoch in range(opt_train.epoch_count,opt_train.total_epoch+1):
        BatchTrainLoss=[]
        if(opt_train.train):
            model.train()
        epoch_start_time = time.time()
        epoch_iter = 0
        for i,data in enumerate(train_dataset):
            total_iters += opt_train.batch_size
            epoch_iter  += opt_train.batch_size
            model.set_input(data)
            model.optimize_parameters()
            StepLoss = model.get_current_losses()
            BatchLoss = model.get_current_lossvalue()[0]
            BatchTrainLoss.append(BatchLoss)
            visualizer.save_step_losses(total_iters,StepLoss)

        Epochloss = sum(BatchTrainLoss) / len(BatchTrainLoss)
        visualizer.print_epoch_losses_value(epoch,Epochloss)

        if(opt_train.use_Adaboost):
            TrainDice,TrainIoU,TrainDiceList,TrainIoUList = test_train_dice(model,opt_train.loss_mode,train_dataset,opt_train.batch_size)
            print('TrainDiceAverage:{},TrainIoUAverage:{}'.format(TrainDice,TrainIoU))
            print('TrainDiceList:{}'.format(TrainDiceList))
            model.feedback(TrainDiceList)
        TestDice,TestIoU,TestDiceList,TrianIoUList   = test_test_dice(model,opt_train.loss_mode,test_dataset,opt_eval.batch_size)
        visualizer.save_epoch_test_metric(epoch,TestDice)
        print('TestDiceAverage: {},TestIoUAverage: {}'.format(TestDice,TestIoU))
        print('TestDiceList:{}'.format(TestDiceList))

        if(TestDice>BestDice):
            BestDice = TestDice
            model.save_networks(epoch)
            model.save_networks('latest')
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.update_learning_rate(TestDice)
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt_train.total_epoch, time.time() - epoch_start_time))




