# -*- coding: utf-8 -*-
import torch
import numpy as np
from medpy.metric.binary import hd,assd,hd95

# @author ZhangMinghui,Southeast University
#
# Module Function: Calculate Metrics for Medical Image Segmentation
# Function <reshape_prediction_and_ground_truth> is used to reshape tensor shapes

"""
Module includes:
        __all__ = ['IoUscore_each_class',
                   'Dicescore_each_class',
                   'VOEscore_each_class',
                   'RVDscore_each_class',
                   'Calculate_Each_IoU',
                   'Calculate_Each_Dice',
                   'Calculate_Each_VOE',
                   'Calculate_Each_RVD',
                   'Calculate_Each_ASSD',
                   'Calculate_Each_MSD',]
"""

##############################################################################
# Calculate Metrics for Medical Image Segmentation
##############################################################################
def reshape_prediction_and_ground_truth(predict, groundtruth,sigmoid):
    """
    reshape input variables of shape [B, C, D, H, W] to [voxel_n, C]
    if only
    :param predict:       prediciton    B*C*D*H*W
    :param groundtruth:   groundtruth   B*C*D*H*W
    :return:
    """
    #Turn to cpu,sacrifice time for memory
    predict = predict.cpu()
    groundtruth = groundtruth.cpu()

    if not sigmoid:
        #Turn to one-hot format
        B,C,D,H,W = predict.shape
        label = torch.argmax(predict,dim=1).unsqueeze(dim=1)
        m_zeros = torch.zeros(B,C,D,H,W)
        #m_zeros = torch.zeros(B,C,D,H,W).to('cuda:1')
        predict = m_zeros.scatter_(dim=1,index=label,value=1)

    #Turn to [voxel_n, C]
    tensor_dim = len(predict.size())
    num_class = list(predict.size())[1]
    if (tensor_dim == 5):
        groundtruth = groundtruth.permute(0, 2, 3, 4, 1)
        predict = predict.permute(0, 2, 3, 4, 1)
    elif (tensor_dim == 4):
        groundtruth = groundtruth.permute(0, 2, 3, 1)
        predict = predict.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))
    predict = torch.reshape(predict, (-1, num_class))
    groundtruth = torch.reshape(groundtruth, (-1, num_class))
    return predict, groundtruth


def IoUscore_each_class(predict,groundtruth,smooth = 1e-5):
    """
    calculate IoUscore_each_class
    :param predict:
    :param groundtruth:
    :return:
    """
    predict_     = predict > 0.5
    groundtruth_ = groundtruth > 0.5
    Intersection = torch.sum((predict_ & groundtruth_),dim=0)
    Union = torch.sum((predict_ | groundtruth_),dim=0)
    #IoU = ((Intersection + smooth) / (Union + smooth)).cpu().numpy()
    IoU = ((Intersection + smooth) / (Union + smooth)).numpy()
    return IoU


def Dicescore_each_class(predict,groundtruth):
    """
    calculate Dicescore_each_class
    :param predict:
    :param groundtruth:
    :return:Dice
    """
    predict     = (predict > 0.5)
    groundtruth = (groundtruth > 0.5)
    predict_vol     = torch.sum(predict,dim=0)
    groundtruth_vol = torch.sum(groundtruth,dim=0)
    Intersection    = torch.sum(predict*groundtruth,dim=0)
    #Dice = ((2.0 * Intersection + 1e-5)/(predict_vol + groundtruth_vol + 1e-5)).cpu().numpy()
    Dice = ((2.0 * Intersection + 1e-5)/(predict_vol + groundtruth_vol + 1e-5)).numpy()
    return Dice


def VOEscore_each_class(predict,groundtruth):
    predict_     = predict > 0.5
    groundtruth_ = groundtruth > 0.5
    Intersection = torch.sum((predict_ & groundtruth_),dim=0)
    Union = torch.sum((predict_ | groundtruth_),dim=0)
    VOE = (1.0 - (Intersection + 1e-5)/ (Union+ 1e-5)).numpy()
    return VOE


def RVDscore_each_class(predict,groundtruth):
    predict     = (predict > 0.5)
    groundtruth = (groundtruth > 0.5)
    predict_vol     = torch.sum(predict,dim=0)
    groundtruth_vol = torch.sum(groundtruth,dim=0)
    RVD_list=[]
    for  i in range(0,predict_vol.shape[0]):
        if(groundtruth_vol[i] == 0):
            RVD_list.append(0.0)
        else:
            RVD_temp = ((predict_vol[i] - groundtruth_vol[i] + 1e-5) / (groundtruth_vol[i] + 1e-5)).numpy()
            RVD_list.append(RVD_temp)
    RVD = np.asarray(RVD_list)
    return RVD


def Calculate_Each_IoU(predict,groundtruth,sigmoid):
    """
    Calculate_Each_Class_IoU,return 1*C numpy.ndarry of each classes of IoU
    :param predict:
    :param groundtruth:
    :return:
    """
    predict, groundtruth = reshape_prediction_and_ground_truth(predict,groundtruth,sigmoid)
    IoU = IoUscore_each_class(predict,groundtruth)
    return IoU


def Calculate_Each_Dice(predict,groundtruth,sigmoid):
    """
    Calculate_Each_Dice,return 1*C numpy.ndarry of each classes of dice
    if mode== BCE   means only have the foreground
              DICE  have the background in the first of the result
    :param predict:      B*C* D*H*W or H*W
    :param groundtruth:  B*C* D*H*W or H*W  (exclude the background)
    :param mode:         loss mode
    :return:Dicefloat list
    """
    predict, groundtruth = reshape_prediction_and_ground_truth(predict,groundtruth,sigmoid)
    Dice = Dicescore_each_class(predict,groundtruth)
    return Dice


def Calculate_Each_VOE(predict,groundtruth,sigmoid):
    predict, groundtruth = reshape_prediction_and_ground_truth(predict, groundtruth, sigmoid)
    VOE = VOEscore_each_class(predict,groundtruth)
    return VOE


def Calculate_Each_RVD(predict,groundtruth,sigmoid):
    predict, groundtruth = reshape_prediction_and_ground_truth(predict, groundtruth, sigmoid)
    RVD = RVDscore_each_class(predict,groundtruth)
    return RVD


def Calculate_Each_ASSD(predict,groundtruth,sigmoid):
    ASSD=[]
    predict = predict.cpu()
    groundtruth = groundtruth.cpu()
    if not sigmoid:
        B,C,D,H,W = predict.shape
        label = torch.argmax(predict,dim=1).unsqueeze(dim=1)
        m_zeros = torch.zeros(B,C,D,H,W)
        predict = m_zeros.scatter_(dim=1,index=label,value=1)
    predict = predict.numpy()
    groundtruth = groundtruth.numpy()
    #convert to binnary if not softmax
    predict     = (predict > 0.5)
    groundtruth = (groundtruth > 0.5)
    for channel in range(0,predict.shape[1]):
        if (np.max(predict[0,channel,...])==False or np.max(groundtruth[0,channel,...])==False):
            ASSD.append(0.0)
        else:
            ASSD.append(assd(predict[0,channel,...],groundtruth[0,channel,...]))
    ASSD = np.asarray(ASSD)
    return ASSD


def Calculate_Each_MSD(predict,groundtruth,sigmoid):
    MSD=[]
    predict = predict.cpu()
    groundtruth = groundtruth.cpu()
    if not sigmoid:
        B,C,D,H,W = predict.shape
        label = torch.argmax(predict,dim=1).unsqueeze(dim=1)
        m_zeros = torch.zeros(B,C,D,H,W)
        predict = m_zeros.scatter_(dim=1,index=label,value=1)
    predict = predict.numpy()
    groundtruth = groundtruth.numpy()
    #convert to binnary if not softmax
    predict     = (predict > 0.5)
    groundtruth = (groundtruth > 0.5)
    for channel in range(0,predict.shape[1]):
        if (np.max(predict[0,channel,...])==False or np.max(groundtruth[0,channel,...])==False):
            MSD.append(0.0)
        else:
            MSD.append(hd(predict[0,channel,...],groundtruth[0,channel,...]))
    MSD = np.asarray(MSD)
    return MSD
