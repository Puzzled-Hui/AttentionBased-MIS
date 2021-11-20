import torch
import torch.nn as nn
import itertools
from .base_model import BaseModel
from .import networks
from util import metric,losses
import numpy as np

class cSEVnetModel(BaseModel):
    """This class implements the cSEVnetModel model, for 3D image-segmentation"""
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options."""
        parser.add_argument('--channel_hierarchy', type=int, default=8,help='how many hierarchies do an inter-middlestage feature map should have')
        return parser

    def __init__(self, opt):
        """
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self,opt)
        # ================================Define Loss name================================
        # =======        specify the training losses you want to print out.          =====
        # ======= The training/test scripts will call <BaseModel.get_current_losses> =====
        # =======         loss can be single or multi- loss function/name            =====
        # ============================== By ZMH 0325 modify ==============================
        self.loss_names = ['cSEVnet_final_output']

        # =============================== Define Model name ==============================
        # =======         specify the models you want to save to the disk.           =====
        # ======= The training/test scripts will call <BaseModel.save_networks>      =====
        # =======                                      and <BaseModel.load_networks> =====
        # =======         model can be single or multi- model framework/name         =====
        # ============================== By ZMH 0325 modify ==============================
        self.model_names= ['cSEVnet']

        # =============================== Define Network =================================
        # =======      define the network structure, implemented in networks.py      =====
        # ============================== By ZMH 0325 modify ==============================
        self.netcSEVnet = networks.define_cSEVnet(in_channels=opt.in_channels,
                                                  out_channels=opt.out_channels,
                                                  finalsigmoid=opt.final_sigmoid,
                                                  fmaps_degree=opt.init_fmaps_degree,
                                                  GroupNormNumber=opt.group_normnumber,
                                                  fmaps_layer_number=opt.fmaps_layer_number,
                                                  layer_order=opt.layer_order,
                                                  channel_hierarchy=opt.channel_hierarchy,
                                                  SE_channel_pooling_type=opt.se_channel_pooling_type,
                                                  device=self.device)

        # ======================== Initialize settings parameters ========================
        # =======  Initialize settings parameters for both train and test scripts    =====
        # =======  Setting parameters |       Value       |          Function        =====
        #          self.loss_mode     | opt.loss_mode     | e.g. loss differs in masks  ==
        #          self.sigmoid       | opt.final_sigmoid | use sigmoid or softmax   =====
        #          self.init_weight   | opt.initial_weight| tackle class imbalance   =====
        #          self.coeff_weight  | zero              | similar to AdaBoost      =====
        #          self.flag          | bool default true | use init- orcoeff weight =====
        # ============================== By ZMH 0325 modify ==============================
        self.loss_mode = opt.loss_mode
        self.sigmoid = opt.final_sigmoid
        self.init_weight  = torch.tensor(np.loadtxt(opt.initial_weight)).float().to(self.device[0])
        self.coeff_weight = torch.zeros(self.init_weight.shape[0]-1).to(self.device[-1])
        self.flag = True

        # ====================== Initialize function tools for train======================
        # =======  Initialize function tools for train such as loss/optimizer ...    =====
        # ======= schedulers will be automatically created by function <BaseModel.setup>==
        # =======  append optimizers; printed by function<BaseModel.update_learning_rate>=
        # =======  Function tools     |       Value       |          Function        =====
        #   self.criterion+loss_name  | CE/DICE/others    | measure pd and gt        =====
        #   self.optimizer+model_name | Adam/SGD/others   | way to update parameters =====
        # ============================== By ZMH 0325 modify ==============================
        if self.isTrain:
            if(opt.loss_mode == 'CE'):
                self.criterioncSEVnet_final_output = nn.CrossEntropyLoss()
            if(opt.loss_mode == 'DICE'):
                self.criterioncSEVnet_final_output = losses.DiceLoss(ignore_index=opt.ignore_index,sigmoid_normalization=opt.sigmoid_normalization)
            if (opt.loss_mode == 'CBL'):
                self.criterioncSEVnet_final_output = losses.CombinedLoss(ignore_index=opt.ignore_index,weight=self.init_weight,sigmoid_normalization=opt.sigmoid_normalization)
            if (opt.loss_mode == 'GDL'):
                self.criterioncSEVnet_final_output = losses.GeneralizedDiceLoss(ignore_index=opt.ignore_index,sigmoid_normalization=opt.sigmoid_normalization)

            if(opt.optim_mode == 'Adam'):
                self.optimizer_cSEvnet = torch.optim.Adam(params=self.netcSEVnet.parameters(),lr=opt.init_learningrate)
            if(opt.optim_mode == 'SGD'):
                self.optimizer_cSEvnet = torch.optim.SGD(params=self.netcSEVnet.parameters())
            self.optimizers.append(self.optimizer_cSEvnet)
        #if use apex
        #self.netcSEVnet,self.optimizer_vnet = amp.initialize(self.netcSEVnet, self.optimizer_vnet, opt_level="O1")

        self.Each_Dice = 0.0
        self.Each_Iou  = 0.0

    def set_input(self,input):
        self.Image       = input['Image'].to(self.device[0])
        self.Mask        = input['Mask'].to(self.device[-1])
        self.Filepath    = input['Filepath'][0] # get the current filepath,filepath is a list
        self.OriginImage = input['OriginImage']



    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.Prediction = self.netcSEVnet(self.Image)  # netcSEVnet(Image)


    def backward(self):
        if (self.loss_mode == 'CE'):
            for i in range(0,self.Mask.shape[1]):
                if(i==0):
                    Mask = self.Mask[:, i,...] * i
                else:
                    Mask += self.Mask[:, i,...] * i
            self.loss_cSEVnet_final_output = self.criterioncSEVnet_final_output(self.Prediction,Mask.long())
        if (self.loss_mode == 'DICE' or self.loss_mode == 'GDL'):
            self.loss_cSEVnet_final_output = self.criterioncSEVnet_final_output(self.Prediction, self.Mask)
        if (self.loss_mode == 'CBL'):
            for i in range(0,self.Mask.shape[1]):
                if(i==0):
                    CEMask = self.Mask[:, i,...] * i
                else:
                    CEMask += self.Mask[:, i,...] * i
            self.loss_cSEVnet_final_output = self.criterioncSEVnet_final_output(self.Prediction, self.Mask, CEMask,self.flag,coeff_weight=self.coeff_weight)
        # if use apex
        # with amp.scale_loss(self.loss_cSEVnet_final_output, self.optimizer_vnet) as scaled_loss:
            # scaled_loss.backward()
        self.loss_cSEVnet_final_output.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_cSEvnet.zero_grad()
        self.backward()
        self.optimizer_cSEvnet.step()

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
        self.coeff_weight= torch.tensor((median_dice-m_t) / (traindicelists-m_t)).to(self.device[0])
        if (self.flag == True):
            self.flag = False
        return None





