import tables
import SimpleITK as sitk
import types
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom
import json

import os
import numpy as np
import torch
from data.base_dataset import BaseDataset,get_transform_MALC


##############################################################################
# MALC dataset
##############################################################################
class MALCdataset(BaseDataset):
    """
    A dataset class for Multi-Atlas Labelling Challenge
    '/path/train' to train
    '/path/test'  to test
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--depth',    type=int, default=144,help='depth of the MALC volume')
        parser.add_argument('--height',   type=int, default=176,help='height of the MALC volume')
        parser.add_argument('--width',    type=int, default=144,help='width of the MALC volume')
        parser.add_argument('--initial_weight', type=str, default="./datasets/MALC/initial_weight/malc_initial_weights.txt",help='initial weight')
        return parser


    def __init__(self,opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self,opt)
        self.loss_mode  = opt.loss_mode
        self.isTrain    = opt.isTrain
        self.transforms = get_transform_MALC(self.isTrain)
        self.n_classes  = opt.out_channels


        self.rootdir   = os.path.join(opt.dataroot,opt.phase)   #opt.phase--> train or test
        self.data_images_and_labels_path = [os.path.join(self.rootdir,mode) for mode in ['images','masks']]  #[images,labels]
        self.datafile_length = len(os.listdir(self.data_images_and_labels_path[0]))




    def __len__(self):
        """Return the total number of voxels in the dataset."""
        return self.datafile_length

    def __getitem__(self,index):
        """Return ThisImageVoxel and ThisWholeTumorMaskVoxel.
        Parameters:
            index - - a random integer for data indexing
        """
        self.data_images_and_labels = [os.path.join(data,os.listdir(data)[index]) for data in self.data_images_and_labels_path]
        #Image
        ImageLists=[]
        ThisImageVoxelFilepath = self.data_images_and_labels[0]
        #print('ThisImageVoxelFilepath:{}'.format(ThisImageVoxelFilepath))
        ThisImageVoxel_Itk_Image = sitk.ReadImage(ThisImageVoxelFilepath)
        ThisImageVoxelArray = (sitk.GetArrayFromImage(ThisImageVoxel_Itk_Image))   #D*H*W
                  
        ImageLists.append(ThisImageVoxelArray)
        Image = np.asarray(ImageLists,dtype=np.int16)
        OriginImage = Image

        #Labels
        ThisMaskVoxelFilepath   = self.data_images_and_labels[1]
        #print('ThisMaskVoxelFilepath:{}'.format(ThisMaskVoxelFilepath))
        ThisMaskVoxel_Itk_Image = sitk.ReadImage(ThisMaskVoxelFilepath)
        ThisMaskVoxelArray = sitk.GetArrayFromImage(ThisMaskVoxel_Itk_Image).astype(np.uint8)


        MaskLists = []
        for i_class in range(0,self.n_classes):
            Mask_i_class = np.where(ThisMaskVoxelArray==i_class,1,0)
            MaskLists.append(Mask_i_class)
        Mask = np.asarray(MaskLists, dtype=np.uint8)


        #transform
        Image, Mask = self.transforms(Image, Mask)
        #To tensor
        Image = torch.from_numpy(Image).float()
        #print(Image.dtype)
        Mask  = torch.from_numpy(Mask).float()
        #print(Mask.dtype)
        return {'Image': Image, 'Mask': Mask, 'Filepath': self.data_images_and_labels[0],'OriginImage': OriginImage}
