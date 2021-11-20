# -*- coding: utf-8 -*-
import sys
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
# Script Function: Demo for Train a model.
#                  This script displays the train procedure in our project in detail.
#                  If you carefully and patiently read the code,
#                  you will have a good command of our Project Layout and code style.
#                  Have you Fun!


if __name__ =="__main__":
    #Initilize the TrainOptions. TrainOptions in <options.train_options.py>
    opt_train = TrainOptions().parse()

    #Initilize the dataset.create_dataset in <data.__init__.py>
    #Need opt_train.dataroot and opt_train.dataroot and other options
    #to assign a concrete dataset rootdir and dataset mode
    train_dataset = create_dataset(opt_train)

    #Initilize the visualizer class, to save train log(step loss,epoch loss etc.)
    visualizer    = Visualizer(opt_train)
    print('The number of training images = %d' % len(train_dataset))

    #Initilize the model.create_model in <models.__init__.py>
    #Need opt_train.model to assign a model class,
    #then in XXX_model.py will construct a concrete model by calling networks.py
    model = create_model(opt_train)

    #Set Up the model,load pre-models if necessary
    #opt_train.continue_train and opt_train.epoch are used to loading pre-models for continue/transfer learning.
    model.setup(opt_train)
    total_iters = 0

    for epoch in range(opt_train.epoch_count,opt_train.total_epoch+1):
        BatchTrainLoss=[]
        if(opt_train.train):
            model.train()
        epoch_start_time = time.time()
        epoch_iter = 0
        for i,data in enumerate(train_dataset):
            total_iters += opt_train.batch_size
            epoch_iter  += opt_train.batch_size
            #Process and argument data then load data into model.
            model.set_input(data)

            #model optimization,includes four steps:
            #1.forward
            #2.optimizer.zero_grad()
            #3.backward
            #4.optimizer.step()
            model.optimize_parameters()

        # ==================Get training log and save into disks=====================
            StepLoss = model.get_current_losses()
            BatchLoss = model.get_current_lossvalue()[0]
            BatchTrainLoss.append(BatchLoss)
            visualizer.save_step_losses(total_iters,StepLoss)

        Epochloss = sum(BatchTrainLoss) / len(BatchTrainLoss)
        visualizer.print_epoch_losses_value(epoch,Epochloss)
        # ==================Get training log and save into disks=====================

        # Tailor for Loss Function needs metric feedback.
        # Most common Loss Function does not need this subprocedure.
        # opt_train.use_Adaboost default is False.
        if(opt_train.use_Adaboost):
            TrainDice,TrainIoU,TrainDiceList,TrainIoUList = test_train_dice(model,opt_train.loss_mode,train_dataset,opt_train.batch_size)
            print('TrainDiceAverage:{},TrainIoUAverage:{}'.format(TrainDice,TrainIoU))
            print('TrainDiceList:{}'.format(TrainDiceList))
            model.feedback(TrainDiceList)

        # ==================Save Models and Update Learning rate=====================
        model.save_networks(epoch)
        model.save_networks('latest')
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.update_learning_rate()
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt_train.total_epoch, time.time() - epoch_start_time))
        # ==================Save Models and Update Learning rate=====================



