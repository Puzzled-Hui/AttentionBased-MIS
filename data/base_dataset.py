import random
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
from abc import ABC, abstractmethod
from scipy import ndimage

# @author ZhangMinghui,Southeast University
#
# Class Definition: Base dataset
# Base Transformer: online Data argumentation

"""
Class memberfunction includes:
            __all__ = ['__init__',
                       'modify_commandline_options',
                       '__len__',
                       '__getitem__',]

online Data argumentation includes:
            __all__ = ['ElasticDeformation',
                       'RandomRotation_ndimage',
                       'RandomFlip_ndimage',
                       'Normalize_ndimage',]
"""

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


##############################################################################
# Transform
##############################################################################
class Compose(object):
    def __init__(self,transforms):
        self.transforms = transforms

    def __call__(self,image,mask):
        for t in self.transforms:
            image,mask = t(image,mask)
        return image,mask


class ElasticDeformation(object):
    def __init__(self,prob,points,sigma):
        self.prob = prob
        self.points = points
        self.sigma = sigma

    def __call__(self,image,mask):
        if (random.random()>self.prob):
            [image_deformed,mask_deformed] = elasticdeform.deform_random_grid([image,mask],order=[3,0],
                                                                              points=self.points,
                                                                              sigma=self.sigma,
                                                                              axis=[(1,2,3),(1,2,3)])
        return image_deformed,mask_deformed


class RandomRotation_ndimage(object):
    """
        random rotate the image (shape [C, D, H, W] or [C, H, W])
    Args:
        angle_range_d (tuple/list/None) : rorate angle range along depth axis (degree),
               only used for 3D images
        angle_range_h (tuple/list/None) : rorate angle range along height axis (degree)
        angle_range_w (tuple/list/None) : rorate angle range along width axis (degree)
    """
    def __init__(self, angle_range_d, angle_range_h, angle_range_w, prob):
        self.angle_range_d  = angle_range_d
        self.angle_range_h  = angle_range_h
        self.angle_range_w  = angle_range_w
        self.prob = prob

    def __apply_transformation(self, image, transform_param_list, order = 1):
        """
        apply rotation transformation to an ND image
        Args:
            image (nd array): the input nd image
            transform_param_list (list): a list of roration angle and axes
            order (int): interpolation order
        """
        for angle, axes in transform_param_list:
            image = ndimage.rotate(image, angle, axes, reshape = False, order = order)
        return image

    def __call__(self,image,mask):
        if random.random() > self.prob:
            image_shape = image.shape
            input_dim   = len(image_shape) - 1
            transform_param_list = []
            if (self.angle_range_d is not None):
                angle_d = np.random.uniform(self.angle_range_d[0],self.angle_range_d[1])
                transform_param_list.append([angle_d,(-1,-2)])
            if (input_dim==3):
                if(self.angle_range_h is not None):
                    angle_h = np.random.uniform(self.angle_range_h[0],self.angle_range_h[1])
                    transform_param_list.append([angle_h,(-1,-3)])
                if(self.angle_range_w is not None):
                    angle_w = np.random.uniform(self.angle_range_w[0],self.angle_range_w[1])
                    transform_param_list.append([angle_w,(-2,-3)])
            image = self.__apply_transformation(image=image,transform_param_list=transform_param_list,order=1)
            mask  = self.__apply_transformation(image=mask,transform_param_list=transform_param_list,order=0)
        return image,mask


class RandomFlip_ndimage(object):
    """
    random flip the image (shape [C, D, H, W] or [C, H, W])
    Args:
        flip_depth (bool) : random flip along depth axis or not, only used for 3D images
        flip_height (bool): random flip along height axis or not
        flip_width (bool) : random flip along width axis or not
    """
    def __init__(self, flip_depth, flip_height, flip_width, prob):
        self.flip_depth  = flip_depth
        self.flip_height = flip_height
        self.flip_width  = flip_width
        self.prob = prob

    def __call__(self,image,mask):
        input_shape = image.shape
        input_dim = len(input_shape) - 1
        flip_axis=[]
        if(self.flip_width):
            if(random.random()>self.prob):
                flip_axis.append(-1)
        if(self.flip_height):
            if(random.random()>self.prob):
                flip_axis.append(-2)
        if(input_dim==3 and self.flip_depth):
            if(random.random()>self.prob):
                flip_axis.append(-3)
        if len(flip_axis)>0:
            image = np.flip(image,axis=flip_axis).copy()
            mask  = np.flip(mask,axis=flip_axis).copy()
        return image,mask


class Normalize_ndimage_one_modality(object):
    """
    Normalize_ndimage for one modality (shape [1, D, H, W] or [1, H, W])
    """
    def __call__(self,image,mask):
        nonzero_number = np.count_nonzero(image)
        mean = (np.sum(image)+ 1e-5) / (nonzero_number + 1e-5)
        power2 = np.where(image>0,np.power((image-mean),2),image)
        std  = np.sqrt((np.sum(power2)+ 1e-5) / (nonzero_number+ 1e-5))
        image = np.where(image>0,((image-mean)/std),image)
        mask  = mask
        return image,mask



class Normalize_ndimage(object):
    """
    Normalize_ndimage for multi-modality (shape [C, D, H, W] or [1, H, W])
    """
    def __call__(self,image,mask):
        c,d,h,w = image.shape
        number = d * h * w
        if(c==1):
            image = (image-np.mean(image))/np.std(image)
            mask  = mask
            return image,mask
        if(c>1):
            imagelist=[]
            for channel in range(0,c):
                image_temp = (image[channel,...] - np.mean(image[channel,...])) / np.std(image[channel,...])
                imagelist.append(image_temp)
            image = np.asarray(imagelist)
            mask  = mask
            return image, mask


class Normalize_ndimage_MALC(object):
    """
    Normalize_ndimage for multi-modality (shape [C, D, H, W] or [1, H, W])
    """
    def __call__(self,image,mask):
        mean = np.mean(image)
        std = np.std(image)
        image = (image - mean) /std
        return image,mask



##############################################################################
# concrete
##############################################################################
def get_transform_Brats18(train):
    transforms=[]
    #transforms.append(Normalize_ndimage_one_modality())
    transforms.append(Normalize_ndimage())
    if train:
        transforms.append(RandomFlip_ndimage(False,True,True,0.5))
        transforms.append(RandomRotation_ndimage((-10,10),(-10,10),(-10,10),0.5))
    return Compose(transforms)


def get_transform_MALC(train):
    transforms=[]
    transforms.append(Normalize_ndimage_MALC())
    if train:
        transforms.append(RandomRotation_ndimage((-10,10),(-10,10),(-10,10),0.5))
    return Compose(transforms)


def get_transform_HVSMR(train):
    transforms=[]
    transforms.append(Normalize_ndimage())
    if train:
        transforms.append(RandomFlip_ndimage(False,True,True,0.5))
        transforms.append(RandomRotation_ndimage((-10,10),(-10,10),(-10,10),0.5))
    return Compose(transforms)
