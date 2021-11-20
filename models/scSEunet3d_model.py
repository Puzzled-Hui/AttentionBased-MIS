import torch
import torch.nn as nn
import itertools
from .base_model import BaseModel
from .import networks
from util import metric,losses
from apex import amp
import numpy as np

class scSEunet3dModel(BaseModel):
    """
    This class implements the scA 3D UNet model, for 3D image-segmentation
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:


        Returns:

        """
        parser.add_argument('--slices_hierarchy',type=int,default=16,help='how many hierarchies do an inter-middlestage voxel should have')
        parser.add_argument('--channel_hierarchy',type=int,default=8,help='how many hierarchies do an inter-middlestage feature map should have')
        return parser

    def __init__(self, opt):
        """Initialize.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self,opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['scSE_Unet3d_final_output']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names= ['scSE_Unet3d']

        # define networks correspond
        self.netscSE_Unet3d = networks.define_scSE_Unet3d(in_channels=opt.in_channels,
                                                          out_channels=opt.out_channels,
                                                          depth=opt.depth,
                                                          finalsigmoid=opt.final_sigmoid,
                                                          fmaps_degree=opt.init_fmaps_degree,
                                                          GroupNormNumber=opt.group_normnumber,
                                                          fmaps_layer_number=opt.fmaps_layer_number,
                                                          layer_order=opt.layer_order,
                                                          slices_hierarchy=opt.slices_hierarchy,
                                                          channel_hierarchy=opt.channel_hierarchy,
                                                          SE_channel_pooling_type=opt.se_channel_pooling_type,
                                                          SE_slice_pooling_type=opt.se_slice_pooling_type,
                                                          device=self.device)
        self.loss_mode = opt.loss_mode
        self.sigmoid = opt.final_sigmoid
        self.init_weight  = torch.tensor(np.loadtxt(opt.initial_weight)).float().to(self.device[-1])
        self.coeff_weight = torch.zeros(self.init_weight.shape[0] - 1).to(self.device[-1])
        self.flag = True
                        
        if self.isTrain:
            # define loss functions
            if(opt.loss_mode == 'CE'):
                self.criterionscSE_Unet3d_final_output = nn.CrossEntropyLoss()
            if(opt.loss_mode == 'DICE'):
                self.criterionscSE_Unet3d_final_output = losses.DiceLoss(ignore_index=opt.ignore_index,sigmoid_normalization=opt.sigmoid_normalization)
            if (opt.loss_mode == 'CBL'):
                self.criterionscSE_Unet3d_final_output = losses.CombinedLoss(ignore_index=opt.ignore_index,weight=self.init_weight,sigmoid_normalization=opt.sigmoid_normalization)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if(opt.optim_mode == 'Adam'):
                self.optimizer_scSE_unet3d = torch.optim.Adam(params=self.netscSE_Unet3d.parameters(),lr=opt.init_learningrate)
            if(opt.optim_mode == 'SGD'):
                self.optimizer_scSE_unet3d = torch.optim.SGD(params=self.netscSE_Unet3d.parameters())
            # append optimizers;schedulers will be automatically printed by function<BaseModel.update_learning_rate>
            self.optimizers.append(self.optimizer_scSE_unet3d)

        self.Each_Dice = 0.0
        self.Each_Iou  = 0.0

    def set_input(self,input):
    
        self.Image       = input['Image'].to(self.device[0])      #Input
        self.Mask        = input['Mask'].to(self.device[-1])       #Output
        self.Filepath    = input['Filepath'][0]                 #get the current filepath
        self.OriginImage = input['OriginImage']



    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.Prediction = self.netscSE_Unet3d(self.Image)  # netscSE_Unet3d(Image)��

    def backward(self):
        if (self.loss_mode == 'CE'):
            for i in range(0,self.Mask.shape[1]):
                if(i==0):
                    Mask = self.Mask[:, i,...] * i
                else:
                    Mask += self.Mask[:, i,...] * i
            self.loss_scSE_Unet3d_final_output = self.criterionscSE_Unet3d_final_output(self.Prediction,Mask.long())
        if (self.loss_mode == 'DICE'):
            self.loss_scSE_Unet3d_final_output = self.criterionscSE_Unet3d_final_output(self.Prediction, self.Mask)
        if (self.loss_mode == 'CBL'):
            for i in range(0,self.Mask.shape[1]):
                if(i==0):
                    CEMask = self.Mask[:, i,...] * i
                else:
                    CEMask += self.Mask[:, i,...] * i
            self.loss_scSE_Unet3d_final_output = self.criterionscSE_Unet3d_final_output(self.Prediction, self.Mask, CEMask,self.flag,coeff_weight=self.coeff_weight)
        self.loss_scSE_Unet3d_final_output.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_scSE_unet3d.zero_grad()
        self.backward()
        self.optimizer_scSE_unet3d.step()

    def calculate_metrics(self):
        self.Each_Dice = metric.Calculate_Each_Dice(self.Prediction,self.Mask,self.sigmoid)
        self.Each_Iou  = metric.Calculate_Each_IoU(self.Prediction,self.Mask,self.sigmoid)
        self.Each_VOE  = metric.Calculate_Each_VOE(self.Prediction,self.Mask,self.sigmoid)
        self.Each_RVD  = metric.Calculate_Each_RVD(self.Prediction,self.Mask,self.sigmoid)
        self.Each_ASSD  = metric.Calculate_Each_ASSD(self.Prediction,self.Mask,self.sigmoid)
        self.Each_MSD  = metric.Calculate_Each_MSD(self.Prediction,self.Mask,self.sigmoid)
        return np.array([self.Each_Dice,self.Each_Iou,self.Each_VOE,self.Each_RVD,self.Each_ASSD,self.Each_MSD])
        
    def calculate_metrics2(self):
        self.Each_Dice = metric.Calculate_Each_Dice(self.Prediction,self.Mask,self.sigmoid)
        self.Each_Iou  = metric.Calculate_Each_IoU(self.Prediction,self.Mask,self.sigmoid)
        return self.Each_Dice,self.Each_Iou
        
    def get_data(self):
        return self.Filepath,self.OriginImage,self.Mask,self.Prediction

    def get_current_filename(self):
        return self.Filepath

    def feedback(self,traindicelists):
        min_dice = np.min(traindicelists)
        median_dice = np.median(traindicelists)
        m_t = min_dice - 0.05
        self.coeff_weight= torch.tensor((median_dice-m_t) / (traindicelists-m_t)).to(self.device[-1])

        #print(self.coeff_weight)
        if (self.flag == True):
            self.flag = False
        return None
