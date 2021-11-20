# -*- coding: utf-8 -*-
import sys
sys.path.append("E:\\Studyfile_ZMH\\GraduationProject\\code\\AttentionBased-MIS\\")
import os
import numpy as np
import torch
from tqdm import tqdm
import SimpleITK as sitk

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.utils import imwrite_nii,imread_nii,squeeze_ndimage,mkdir,save_predictions,calculate_average_metrics,\
    record_metrics,display_metrics,init_metric_vector,init_metric_name

# @author ZhangMinghui,Southeast University
#
# Script Function: Test Model for a specific model and dataset.


if __name__ == '__main__':
    opt = TestOptions().parse()                                         # get test options
    dataset = create_dataset(opt)                                       # create a dataset given opt.dataset_mode and other options
    print('The number of training images = %d' % len(dataset))
    model = create_model(opt)                                           # create a model given opt.model and other options
    model.setup(opt)                                                    # regular setup: load and print networks; create schedulers
    if opt.eval:
        model.eval()

    Metric_Name = init_metric_name()
    Metric_Vector = init_metric_vector(row=len(Metric_Name),col=opt.out_channels)

    print('-----------The test procedure starts...-----------')
    for i,data in enumerate(tqdm(dataset,ncols=80)):
        model.set_input(data)
        model.test()
        Batch_Metric_Vector = model.calculate_metrics()
        Metric_Vector += Batch_Metric_Vector
        if (opt.log_info):
            record_metrics(Batch_Metric_Vector,model.get_current_filename(),opt.doc,opt.doc_dec)
        if (opt.write):
            save_predictions(model,opt,opt.dataset_mode)

    Average_Metric_Vector_List, Average_Metric_Vector = calculate_average_metrics(Metric_Vector,len(dataset))
    display_metrics(Average_Metric_Vector_List,Average_Metric_Vector,Metric_Name)
    print('-----------The test procedure ends!-----------')


