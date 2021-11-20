# -*- coding: utf-8 -*-
import sys
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
# Script Function: Demo for Test a model.
#                  This script displays the test procedure in our project in detail.
#                  If you carefully and patiently read the code,
#                  you will have a good command of our Project Layout and code style.
#                  Have you Fun!


if __name__ == '__main__':
    #Initilize the TestOptions. TestOptions in <options.test_options.py>
    opt = TestOptions().parse()

    #Initilize the dataset.create_dataset in <data.__init__.py>
    #Need opt_test.dataroot and opt_test.dataroot and some other options
    #to assign a concrete dataset rootdir and dataset mode
    dataset = create_dataset(opt)
    print('The number of training images = %d' % len(dataset))

    #Initilize the model.create_model in <models.__init__.py>
    #Need opt_train.model to assign a model class,
    #then in XXX_model.py will construct a concrete model by calling networks.py
    model = create_model(opt)

    #Set Up the model,load pre-models.
    model.setup(opt)
    if opt.eval:
        model.eval()

    #Metric_Name and Metric_Vector is used to save test results
    #Metric_Name includes Dice VOE ASSD MSD(Hausdorff distance) etc.
    Metric_Name = init_metric_name()
    Metric_Vector = init_metric_vector(row=len(Metric_Name),col=opt.out_channels)

    print('-----------The test procedure starts...-----------')
    for i,data in enumerate(tqdm(dataset,ncols=80)):
        # Process and argument data then load data into model.
        model.set_input(data)

        # Test the model,only includes forward procedure.
        model.test()

    # ================== Calculate Metric Display and Write results if necessary =====================
        Batch_Metric_Vector = model.calculate_metrics()
        Metric_Vector += Batch_Metric_Vector
        if (opt.log_info):
            record_metrics(Batch_Metric_Vector,model.get_current_filename(),opt.doc,opt.doc_dec)
        if (opt.write):
            save_predictions(model,opt,opt.dataset_mode)
    Average_Metric_Vector_List, Average_Metric_Vector = calculate_average_metrics(Metric_Vector,len(dataset))
    display_metrics(Average_Metric_Vector_List,Average_Metric_Vector,Metric_Name)
    # ================== Calculate Metric Display and Write results if necessary =====================
    print('-----------The test procedure ends!-----------')


