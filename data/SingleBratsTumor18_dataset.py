import tables
import SimpleITK as sitk
import types
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom
import os
import numpy as np
import torch


from data.base_dataset import BaseDataset,get_transform_Brats18


##############################################################################
# add the multi-tumor-categories
##############################################################################
class SingleBratsTumor18Dataset(BaseDataset):
    """
    A dataset class for 2018Brats data to train the whole tumor
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
        parser.add_argument('--modality', type=str, default='t1,t2,t1ce,flair',help='one or more modality')
        parser.add_argument('--tumor_category', type=str, default='whole_tumor,tumor_core,enhancing_tumor',help='one or more mask')
        parser.add_argument('--depth',    type=int, default=144,help='depth of the brats volume')
        parser.add_argument('--height',   type=int, default=224,help='height of the brats volume')
        parser.add_argument('--width',    type=int, default=224,help='width of the brats volume')
        parser.add_argument('--initial_weight', type=str, default="./datasets/Brats/initial_weight/brats_initial_weights.txt",help='initial weight')
        return parser

    def __init__(self,opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self,opt)
        self.loss_mode  = opt.loss_mode
        self.isTrain    = opt.isTrain
        self.transforms = get_transform_Brats18(self.isTrain)
        
        self.modality  = opt.modality
        str_modals = self.modality.split(',')
        self.modality = []
        for str_modal in str_modals:
            self.modality.append(str_modal)

        self.tumor_category = opt.tumor_category
        str_tumor_categorys  = self.tumor_category.split(',')
        self.tumor_category = []
        for str_tumor_category in str_tumor_categorys:
            self.tumor_category.append(str_tumor_category)

        self.rootdir   = os.path.join(opt.dataroot,opt.phase)                   #opt.phase--> train or test
        self.filepathlists =[]
        self.childdir = os.listdir(self.rootdir)
        for i in range(0,len(self.childdir)):
            filepathname = os.path.join(self.rootdir,self.childdir[i])
            self.filepathlists.append(filepathname)



    def __len__(self):
        """Return the total number of voxels in the dataset."""
        length = len(self.filepathlists)
        return length

    def __getitem__(self,index):
        """Return ThisImageVoxel and ThisWholeTumorMaskVoxel.

        Parameters:
            index - - a random integer for data indexing

        """
        if self.loss_mode == 'CE':
            ThisImageVoxellist = []
            for i in range(len(self.modality)):
                ThisImageVoxelname = self.childdir[index] + "_" + self.modality[i] +".nii.gz"
                ThisImageVoxelFilepath = os.path.join(self.filepathlists[index],ThisImageVoxelname)
                ThisImageVoxel_Ikt_Image  = sitk.ReadImage(ThisImageVoxelFilepath)
                ThisImageVoxel = sitk.GetArrayFromImage(ThisImageVoxel_Ikt_Image)
                ThisImageVoxel = ThisImageVoxel[4:148,8:232,8:232]#image 144*224*224
                ThisImageVoxellist.append(ThisImageVoxel)
            Image = np.asarray(ThisImageVoxellist)
            OriginImage = Image

            ThisMaskname = self.childdir[index] + "_" + "seg" +".nii.gz"
            ThisMaskFilepath = os.path.join(self.filepathlists[index], ThisMaskname)
            ThisMask_Ikt_Image = sitk.ReadImage(ThisMaskFilepath)
            ThisMask = sitk.GetArrayFromImage(ThisMask_Ikt_Image)
            ThisMask = ThisMask[4:148,8:232,8:232]

            ThisMasklist = []
            for i in range(len(self.tumor_category)):
                if(self.tumor_category[i]=="whole_tumor"):
                    ThisWholeTumorMask = np.where(ThisMask > 0, 1, 0)
                    ThisMasklist.append(ThisWholeTumorMask)
                if(self.tumor_category[i]=="tumor_core"):
                    ThisTumorCoreMask = np.where(np.logical_or(ThisMask==1,ThisMask==4),1,0)
                    ThisMasklist.append(ThisTumorCoreMask)
                if(self.tumor_category[i]=="enhancing_tumor"):
                    ThisEnhanceTumorMask = np.where(ThisMask==4,1,0)
                    ThisMasklist.append(ThisEnhanceTumorMask)

            Mask = np.asarray(ThisMasklist)
            
            Image,Mask = self.transforms(Image,Mask)

            Image = torch.from_numpy(Image).float()
            Mask  = torch.from_numpy(Mask).float()
            return {'Image': Image, 'Mask': Mask,'Filepath':self.filepathlists[index],'OriginImage':OriginImage}

        if self.loss_mode == 'DICE':
            ThisImageVoxellist = []
            for i in range(len(self.modality)):
                ThisImageVoxelname = self.childdir[index] + "_" + self.modality[i] +".nii.gz"
                ThisImageVoxelFilepath = os.path.join(self.filepathlists[index],ThisImageVoxelname)
                ThisImageVoxel_Ikt_Image  = sitk.ReadImage(ThisImageVoxelFilepath)
                ThisImageVoxel = sitk.GetArrayFromImage(ThisImageVoxel_Ikt_Image)
                ThisImageVoxel = ThisImageVoxel[4:148,8:232,8:232]#image 144*224*224
                ThisImageVoxellist.append(ThisImageVoxel)
            Image = np.asarray(ThisImageVoxellist)
            OriginImage = Image


            ThisMaskname = self.childdir[index] + "_" + "seg" +".nii.gz"
            ThisMaskFilepath = os.path.join(self.filepathlists[index], ThisMaskname)
            ThisMask_Ikt_Image = sitk.ReadImage(ThisMaskFilepath)
            ThisMask = sitk.GetArrayFromImage(ThisMask_Ikt_Image)
            ThisMask = ThisMask[4:148,8:232,8:232]

            ThisMasklist = []
            ThisBackgroundMask = np.ones(shape=ThisMask.shape, dtype=np.int16)
            ThisBackgroundMask = np.where(ThisMask > 0, 0, ThisBackgroundMask)
            ThisMasklist.append(ThisBackgroundMask)

            for i in range(len(self.tumor_category)):
                if(self.tumor_category[i]=="whole_tumor"):
                    ThisWholeTumorMask = np.where(ThisMask > 0, 1, 0)
                    ThisMasklist.append(ThisWholeTumorMask)
                if(self.tumor_category[i]=="tumor_core"):
                    ThisTumorCoreMask = np.where(np.logical_or(ThisMask==1,ThisMask==4),1,0)
                    ThisMasklist.append(ThisTumorCoreMask)
                if(self.tumor_category[i]=="enhancing_tumor"):
                    ThisEnhanceTumorMask = np.where(ThisMask==4,1,0)
                    ThisMasklist.append(ThisEnhanceTumorMask)

            Mask = np.asarray(ThisMasklist)

            Image,Mask = self.transforms(Image,Mask)

            Image = torch.from_numpy(Image).float()
            Mask  = torch.from_numpy(Mask).float()
            return {'Image': Image, 'Mask': Mask,'Filepath':self.filepathlists[index],'OriginImage':OriginImage}

