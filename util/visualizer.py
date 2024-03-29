# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import ntpath
import time
from . import utils
from subprocess import Popen, PIPE


# @author ZhangMinghui,Southeast University
#
# Class Function: class of Visulizer,to log loss and metric.etc

"""
Member function includes:
                __all__ = ['print_current_losses',
                           'print_epoch_losses',
                           'print_epoch_losses_value',
                           'save_step_losses',
                           'save_epoch_test_metric']
"""
class Visualizer():
    """This class includes several functions that can save logging information."""

    def __init__(self, opt):
        """
        Initialize the Visualizer class
        """
        self.opt = opt  # cache the option
        self.saved = False

        # create logging files to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'epoch_loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Epoch Loss (%s) ================\n' % now)

        self.steploss_log_name = os.path.join(opt.checkpoints_dir, opt.name, 'step_loss_log.txt')
        with open(self.steploss_log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Step Loss (%s) ================\n' % now)
            
        self.epoch_test_metric = os.path.join(opt.checkpoints_dir, opt.name, 'epoch_test_metric.txt')
        with open(self.epoch_test_metric , "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Step Loss (%s) ================\n' % now)
            
    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    # losses: same format as |losses| of plot_current_losses
    def print_epoch_losses(self, epoch, losses):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        """
        message = '[epoch:%d =========> ' % (epoch)
        for k, v in losses.items():
            message += ' %s_loss: %.7f ' % (k, v)
        message += ']'
        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
            
    def print_epoch_losses_value(self, epoch, losses):
        """print current losses on console; also save the losses to the disk
    
        Parameters:
            epoch (int) -- current epoch
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        """
        message = '[epoch:%d =========> ' % (epoch)
        message += ' loss: %.7f ' % losses
        message += ']'
        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
            
    # losses: same format as |losses| of plot_current_losses
    def save_step_losses(self, step, losses):
        """print current losses on console; also save the losses to the disk

        Parameters:
            step (int) -- current epoch
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        """
        message = '%d\t' %step
        for k, v in losses.items():
            message += '%.7f\t' %v
        with open(self.steploss_log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    # save the epoch test metric
    def save_epoch_test_metric(self, epoch, metric):
        """print current losses on console; also save the losses to the disk

        Parameters:
            step (int) -- current epoch
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        """
        message = '%d\t' %epoch
        message += '%.5f\t' %metric
        with open(self.epoch_test_metric, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

