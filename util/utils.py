from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import SimpleITK as sitk

# @author ZhangMinghui,Southeast University
#
# Module Function: This module contains simple
# helper functions, both used in <train_and_eval.py> and <test.py>

"""
Module includes:
        __all__ = ['mkdirs',
                   'mkdir',
                   'imread_nii',
                   'imwrite_nii',
                   'squeeze_ndimage',
                   'calculate_average_Dice_VOE',
                   'test_train_dice',
                   'test_test_dice',
                   'write_nii_brats',
                   'write_nii_malc',
                   'write_nii_hvsmr',
                   'save_predictions',
                   'calculate_average_metrics',
                   'record_metrics',
                   'display_metrics',
                   'init_metric_vector',
                   'init_metric_name']
"""


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def imread_nii(path,dtype=np.int16):
    """
    :param path:   path to load the image
    :param dtype:  default np.int16
    :return: Array
    """
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(dtype=dtype)


def imwrite_nii(ndarray,path):
    """
    :param ndarray:     image to save
    :param path:        where to save
    """
    sitk.WriteImage(sitk.GetImageFromArray(ndarray),path)


def squeeze_ndimage(ndarray):
    ndim = ndarray.shape[0]
    for i in range(0,ndim):
            if(i==0):
                temp_ndarray = ndarray[i,...] * 0
            else:
                temp_ndarray +=  ndarray[i,...] * i
    return temp_ndarray

##############################################################################
# Train and Eval
##############################################################################
def calculate_average_Dice_VOE(DiceTotal,IoUTotal,loss_mode,dataset_size):
    Dice = DiceTotal[1:] / dataset_size
    IoU  = IoUTotal[1:] / dataset_size
    Diceaverage = np.mean(DiceTotal[1:] / dataset_size)
    IoUaverage = np.mean(IoUTotal[1:] / dataset_size)
    return Diceaverage,IoUaverage,Dice,IoU


def test_train_dice(model,loss_mode,train_dataset,train_batchsize):
    model.eval()
    DiceTotal=0.0
    IoUTotal =0.0
    for i,data in enumerate(train_dataset):
        model.set_input(data)
        model.test()
        Each_Dice,Each_IoU=model.calculate_metrics2()
        DiceTotal += Each_Dice
        IoUTotal  += Each_IoU
    Diceaverage, IoUaverage , Dice, IoU= calculate_average_Dice_VOE(DiceTotal,IoUTotal,loss_mode,len(train_dataset)/train_batchsize)
    return Diceaverage,IoUaverage,Dice, IoU


def test_test_dice(model,loss_mode,test_dataset,test_batchsize):
    model.eval()
    DiceTotal=0.0
    IoUTotal =0.0
    for i,data in enumerate(test_dataset):
        model.set_input(data)
        model.test()
        Each_Dice,Each_IoU=model.calculate_metrics2()
        DiceTotal += Each_Dice
        IoUTotal  += Each_IoU
    Diceaverage, IoUaverage,Dice, IoU= calculate_average_Dice_VOE(DiceTotal,IoUTotal,loss_mode,len(test_dataset)/test_batchsize)
    return Diceaverage,IoUaverage,Dice, IoU

##############################################################################
# Test
##############################################################################
def write_nii_brats(model,opt):
    modality = opt.modality
    str_modals = modality.split(',')
    modality = []
    for str_modal in str_modals:
        modality.append(str_modal)

    tumor_category = opt.tumor_category
    str_tumor_categorys = tumor_category.split(',')
    tumor_category = []
    for str_tumor_category in str_tumor_categorys:
        tumor_category.append(str_tumor_category)

    Filepath, OriginImage, Mask, Prediction = model.get_data()
    OriginImage = OriginImage.data.numpy()
    for i in range(len(modality)):
        Filename = "Origin_" + modality[i] + ".nii.gz"
        OriginImagetemp = OriginImage[0, i, :, :, :]
        OriginImagetemp = np.asarray(OriginImagetemp, dtype=np.int16)
        OriginImage_out_nii = sitk.GetImageFromArray(OriginImagetemp)
        OriginImage_out_path = os.path.join(Filepath, Filename)
        sitk.WriteImage(OriginImage_out_nii, OriginImage_out_path)

    Prediction = Prediction.data.cpu().numpy()
    for i in range(len(tumor_category)):
        Filename = "Prediction_" + tumor_category[i] + ".nii.gz"
        #@fix zmh 0116 add the threshold
        Predictiontemp = Prediction[0,i+1,:,:,:] > 0.5
        Predictiontemp = np.asarray(Predictiontemp, dtype=np.int16)
        Prediction_out_nii = sitk.GetImageFromArray(Predictiontemp)
        Prediction_out_path = os.path.join(Filepath, Filename)
        sitk.WriteImage(Prediction_out_nii, Prediction_out_path)

    Mask = Mask.data.cpu().numpy()
    for i in range(len(tumor_category)):
        if(opt.loss_mode == 'CE'):
            Filename = "Mask_" + tumor_category[i] + ".nii.gz"
            Masktemp = Mask[0, i, :, :, :]
            Masktemp = np.asarray(Masktemp, dtype=np.int16)
            Mask_out_nii = sitk.GetImageFromArray(Masktemp)
            Mask_out_path = os.path.join(Filepath, Filename)
            sitk.WriteImage(Mask_out_nii, Mask_out_path)
        if(opt.loss_mode == 'DICE'):
            Filename = "Mask_" + tumor_category[i] + ".nii.gz"
            Masktemp = Mask[0, i + 1, :, :, :]
            Masktemp = np.asarray(Masktemp, dtype=np.int16)
            Mask_out_nii = sitk.GetImageFromArray(Masktemp)
            Mask_out_path = os.path.join(Filepath, Filename)
            sitk.WriteImage(Mask_out_nii, Mask_out_path)
    for i in range(len(tumor_category)):
        if (i == 0):
            TempPrediction = Prediction[0,i+1,:,:,:] > 0.5
            TempPrediction = np.asarray(TempPrediction,dtype=np.int16)
            TempMask = Mask[0, i + 1, :, :, :]
        else:
            TempPrediction += np.asarray(Prediction[0,i+1,:,:,:] > 0.5, dtype=np.int16) * i
            TempMask += Mask[0, i + 1, :, :, :] * i
    imwrite_nii(TempPrediction,os.path.join(Filepath,'PredictionAll.nii.gz'))
    imwrite_nii(TempMask,os.path.join(Filepath,'MaskAll.nii.gz'))


def write_nii_malc(model,opt):
    Filepath, OriginImage, Mask, Prediction = model.get_data()
    Filename = Filepath[-13:]
    NewFilepath = os.path.join(os.path.join(os.path.join(os.path.join(opt.dataroot,opt.phase),'prediction'),Filepath[-13:-7]))
    mkdir(NewFilepath)
    OriginImagePath = os.path.join(NewFilepath,'OriginImage'+Filename)
    imwrite_nii(OriginImage.data.numpy()[0, 0, ...], OriginImagePath)
    OriginMaskPath  = os.path.join(NewFilepath,'OriginMask'+Filename)
    Mask_new = squeeze_ndimage(Mask.data.cpu().numpy()[0, ...])
    imwrite_nii(Mask_new,OriginMaskPath)
    B, C, D, H, W = Prediction.shape
    label = torch.argmax(Prediction.data.cpu(), dim=1).unsqueeze(dim=1)
    m_zeros = torch.zeros(B, C, D, H, W)
    Prediction = m_zeros.scatter_(dim=1, index=label, value=1)
    Prediction_new = squeeze_ndimage(Prediction.numpy()[0,...])
    PredictionPath  = os.path.join(NewFilepath,'Prediction'+Filename)
    imwrite_nii(Prediction_new,PredictionPath)


def write_nii_hvsmr(model,opt):
    Filepath, OriginImage, Mask, Prediction = model.get_data()
    Filename = Filepath[-11:]
    NewFilepath = os.path.join(os.path.join(os.path.join(os.path.join(opt.dataroot,opt.phase),'prediction'),Filepath[-11:-7]))
    mkdir(NewFilepath)
    OriginImagePath = os.path.join(NewFilepath,'OriginImage'+Filename)
    imwrite_nii(OriginImage.data.numpy()[0, 0, ...], OriginImagePath)
    OriginMaskPath  = os.path.join(NewFilepath,'OriginMask'+Filename)
    Mask_new = squeeze_ndimage(Mask.data.cpu().numpy()[0, ...])
    imwrite_nii(Mask_new,OriginMaskPath)
    PredictionPath  = os.path.join(NewFilepath,'Prediction'+Filename)
    B, C, D, H, W = Prediction.shape
    label = torch.argmax(Prediction.data.cpu(), dim=1).unsqueeze(dim=1)
    m_zeros = torch.zeros(B, C, D, H, W)
    Prediction = m_zeros.scatter_(dim=1, index=label, value=1)
    Prediction_new = squeeze_ndimage(Prediction.numpy()[0,...])
    imwrite_nii(Prediction_new,PredictionPath)


def save_predictions(model,opt,dataset_mode):
    if   (dataset_mode == 'SingleBratsTumor18'):
        write_nii_brats(model,opt)
    elif (dataset_mode == 'MALC'):
        write_nii_malc(model,opt)
    elif (dataset_mode == 'HVSMR'):
        write_nii_hvsmr(model,opt)
    else:
        raise Exception("Invalid dataset_mode!", dataset_mode)


def calculate_average_metrics(Metric_Vector,dataset_size):
    Average_Metric_Vector_List = []
    Average_Metric_Vector = []
    for i in range(0,Metric_Vector.shape[0]):
        Average_Metric_Vector_List.append(Metric_Vector[i][1:]/dataset_size)
        Average_Metric_Vector.append(np.mean(Metric_Vector[i][1:]/dataset_size))
    return Average_Metric_Vector_List,Average_Metric_Vector


def record_metrics(Batch_Metric_Vector,Filepath,doc,doc_dec):
    with open(doc,'a+') as f:
        f.write(Filepath + "\t")
        for i in range(Batch_Metric_Vector.shape[0]):
            f.write(str(np.mean(Batch_Metric_Vector[i][1:]))+"\t")
        f.write('\n')
    f.close()
    with open(doc_dec,'a+') as f:
        f.write(Filepath)
        for i in range(1,Batch_Metric_Vector.shape[1]):
            f.write("\t"+str(Batch_Metric_Vector[0][i]))
        f.write('\n')


def display_metrics(Average_Metric_Vector_List,Average_Metric_Vector,Metric_Name):
    print('TestDiceList:    {}'.format(Average_Metric_Vector_List[0]))
    for i in range(0,len(Average_Metric_Vector)):
        print(Metric_Name[i]+':     {}'.format(Average_Metric_Vector[i]))


def init_metric_vector(row,col):
    return np.zeros((row,col))


def init_metric_name():
    return ['Dice','IoU','VOE','RVD','ASSD','MSD']