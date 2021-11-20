# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.optim import lr_scheduler
from torch.utils.checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint_sequential
import numpy as np
import functools

from util.activation import swish,Mish

# @author ZhangMinghui,Southeast University
#
# Module Function: Network library,includes: --basic structures
#                                            --attention module
#                                            --whole model
#                                            --other useful functions.

"""
Useful function includes:
                __all__ = ['Identity',
                           'get_norm_layer',
                           'get_scheduler',
                           'init_weights',
                           'init_net']
                           
Basic structure function includes:
                         __all__ = ['create_feature_maps',
                                    'create_conv',
                                    'SingleConv',
                                    'DoubleConv']
                                    
Model includes:
      __all__ = ['UNet3D', 'cSEUnet3D', 'csSEUnet3D', 'scSEUnet3D', 'agscSEUNet3D', 'PEUNet3d', 
                 'VNet', 'cSEVNet', 'csSEVNet', 'scSEVNet', 'agscSEVNet']
                 
Attention module includes:
                 __all__ = ['cSE', 'csA', 'scA', 'PAM', 'PE', 'AG', 'CBAM', 'Spilit Attention', 'Dual Attention']      
                 
"""

###############################################################################
# Helper Functions
###############################################################################

"""*************************************Some Helpful Functions*****************************"""
class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,patience=opt.plateau_patience)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

"""""""*************************************Some Helpful Functions*****************************"""

"""""""*************************************Define Models*******************************************"""

def define_Unet3d(in_channels,out_channels,finalsigmoid,fmaps_degree,GroupNormNumber,fmaps_layer_number,layer_order,device):
    net = UNet3D(in_channels,out_channels,finalsigmoid,fmaps_degree,GroupNormNumber,fmaps_layer_number,layer_order,device)
    return net

def define_PEUnet3d(in_channels,out_channels,finalsigmoid,fmaps_degree,GroupNormNumber,fmaps_layer_number,layer_order,channel_hierarchy,device):
    net = PEUNet3d(in_channels,out_channels,finalsigmoid,fmaps_degree,GroupNormNumber,fmaps_layer_number,layer_order,channel_hierarchy,device)
    return net

def define_scSE_Unet3d(in_channels,out_channels,depth,finalsigmoid,fmaps_degree,GroupNormNumber,fmaps_layer_number,layer_order,slices_hierarchy,channel_hierarchy,
                       SE_channel_pooling_type,SE_slice_pooling_type,device):
    net = scSEUNet3D(in_channels,out_channels,depth,finalsigmoid,fmaps_degree,GroupNormNumber,fmaps_layer_number,layer_order,slices_hierarchy,channel_hierarchy,
                     SE_channel_pooling_type,SE_slice_pooling_type,device)
    return net

def define_cSE_Unet3d(in_channels,out_channels,finalsigmoid,fmaps_degree,GroupNormNumber,fmaps_layer_number,layer_order,channel_hierarchy,SE_channel_pooling_type,device):
    """modify zmh 0116 add the SE_channel_pooling_type"""
    net = cSEUNet3D(in_channels,out_channels,finalsigmoid,fmaps_degree,GroupNormNumber,fmaps_layer_number,layer_order,channel_hierarchy,SE_channel_pooling_type,device)
    return net

def define_csSE_Unet3d(in_channels,out_channels,depth,finalsigmoid,fmaps_degree,GroupNormNumber,fmaps_layer_number,layer_order,slices_hierarchy,channel_hierarchy,
                       SE_channel_pooling_type, SE_slice_pooling_type,device):
    net = csSEUNet3D(in_channels,out_channels,depth,finalsigmoid,fmaps_degree,GroupNormNumber,fmaps_layer_number,layer_order,slices_hierarchy,channel_hierarchy,
                     SE_channel_pooling_type, SE_slice_pooling_type,device)
    return net

def define_agscSE_Unet3d(in_channels,out_channels,use_spatial_attention,depth,finalsigmoid,fmaps_degree,GroupNormNumber,
                         fmaps_layer_number,layer_order,slices_hierarchy,channel_hierarchy,SE_channel_pooling_type, SE_slice_pooling_type,device):
    net = agscSEUNet3D(in_channels,out_channels,use_spatial_attention,depth,finalsigmoid,fmaps_degree,GroupNormNumber,
                       fmaps_layer_number,layer_order,slices_hierarchy,channel_hierarchy,SE_channel_pooling_type, SE_slice_pooling_type,device)
    return net


def define_Vnet(in_channels,out_channels,finalsigmoid,fmaps_degree,GroupNormNumber,fmaps_layer_number,layer_order,device):
    net = VNet(in_channels,out_channels,finalsigmoid,fmaps_degree,GroupNormNumber,fmaps_layer_number,layer_order,device)
    return net

def define_cSEVnet(in_channels,out_channels,layer_order,GroupNormNumber,finalsigmoid,fmaps_degree,fmaps_layer_number,channel_hierarchy,SE_channel_pooling_type,device):
    net = cSEVNet(in_channels,out_channels,layer_order,GroupNormNumber,finalsigmoid,fmaps_degree,fmaps_layer_number,channel_hierarchy,SE_channel_pooling_type,device)
    return net

def define_csSEVnet(in_channels,out_channels,layer_order,GroupNormNumber,finalsigmoid,fmaps_degree,fmaps_layer_number,depth,
                 channel_hierarchy,slice_hierarchy,SE_channel_pooling_type,SE_slice_pooling_type,device):
    net = csSEVNet(in_channels,out_channels,layer_order,GroupNormNumber,finalsigmoid,fmaps_degree,fmaps_layer_number,depth,
                 channel_hierarchy,slice_hierarchy,SE_channel_pooling_type,SE_slice_pooling_type,device)
    return net

def define_scSEVnet(in_channels,out_channels,layer_order,GroupNormNumber,finalsigmoid,fmaps_degree,fmaps_layer_number,depth,
                 channel_hierarchy,slice_hierarchy,SE_channel_pooling_type,SE_slice_pooling_type,device):
    net = scSEVNet(in_channels,out_channels,layer_order,GroupNormNumber,finalsigmoid,fmaps_degree,fmaps_layer_number,depth,
                 channel_hierarchy,slice_hierarchy,SE_channel_pooling_type,SE_slice_pooling_type,device)
    return net

def define_agscSEVnet(in_channels,out_channels,layer_order,GroupNormNumber,finalsigmoid,fmaps_degree,fmaps_layer_number,depth,
                 channel_hierarchy,slice_hierarchy,SE_channel_pooling_type,SE_slice_pooling_type,device):
    net = agscSEVNet(in_channels,out_channels,layer_order,GroupNormNumber,finalsigmoid,fmaps_degree,fmaps_layer_number,depth,
                 channel_hierarchy,slice_hierarchy,SE_channel_pooling_type,SE_slice_pooling_type,device)
    return net
"""""""*************************************Define Models*******************************************"""

"""********************Unet3D cSEUnet3D scSEUnet3D ... share these functions*******************"""
##############################################################################
# Unet3D cSEUnet3D scSEUnet3D share these functions
##############################################################################
def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]
    #return [32,64,128,256]

def create_conv(in_channels,out_channels,kernel_size,order='cri',GroupNumber=8,padding=1,stride=1,Deconv=False):
    """
    @Compiled by zmh
    create an ordered convlution layer for the UNet

    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :param layer_order: the order of layer common match :
                        'cr'-> conv+relu
                        'crb'->conv+relu+batchnorm
                        'crg'->conv+relu+groupnorm(groupnorm number is designed)
                        ......
    :param GroupNumber:
    :param Padding:
    :return:
    """
    assert 'c' in order , 'Convolution must have a conv operation'
    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'p':
            modules.append(('pReLU',nn.PReLU(num_parameters=out_channels)))
        elif char == 's':
            modules.append(('swish',swish()))
        elif char == 'm':
            modules.append(('Mish',Mish()))
        # elif char == 's':
        #     modules.append('Swish',nn.swish())
        elif char == 'c':
            if not Deconv:
                # add learnable bias only in the absence of gatchnorm/groupnorm
                """@fixzmh
                          bias -- if not has the batchnorm or the groupnorm
                                  bias will be true and transmit to the conv3d  
                """
                bias = not ('g' in order or 'b' in order)
                """@fixzmh
                          conv    --name
                          conv3d  --operation,in a module of 3DUNet need twice conv3d
                                    input_feature-->out_feature//2-->out_feature
                """
                #modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))
                #name is important!
                modules.append(('conv', nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias, padding=padding,stride=stride)))
            else:
                bias = not ('g' in order or 'b' in order)
                modules.append(('convtranspose3d',nn.ConvTranspose3d(in_channels,out_channels,kernel_size,bias=bias,padding=padding,stride=stride)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            assert not is_before_conv, 'GroupNorm MUST go after the Conv3d'
            # number of groups must be less or equal the number of channels
            if out_channels < GroupNumber:
                GroupNumber = out_channels
            modules.append(('groupnorm', nn.GroupNorm(num_groups=GroupNumber, num_channels=out_channels)))
        elif char == 'i':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                #affine true ---> with learnable parameters
                modules.append(('instancenorm',nn.InstanceNorm3d(in_channels,affine=True)))
            else:
                modules.append(('instancenorm',nn.InstanceNorm3d(out_channels,affine=True)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError("Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c','i']")

    return modules

#General SingleConv
class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm.
    The order of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers  default crb
        GroupNumber (int): number of groups for the GroupNorm
    """
    def __init__(self,in_channels,out_channels,kernel_size,order='crg',GroupNumber=8):
        super(SingleConv,self).__init__()
        for name,module in create_conv(in_channels,out_channels,kernel_size,order,GroupNumber):
            self.add_module(name,module)

#General Doubleconv
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, encoder,order='crg', GroupNumber=8):
        super(DoubleConv,self).__init__()
        #Encoder
        if encoder:
            conv1_in_channels  = in_channels
            conv1_out_channels = out_channels // 2
            if(conv1_out_channels < in_channels):
                conv1_out_channels = in_channels
            conv2_in_channels,conv2_out_channels =conv1_out_channels,out_channels
        #Decoder
        else:
            conv1_in_channels,conv1_out_channels = in_channels,out_channels
            conv2_in_channels,conv2_out_channels = out_channels,out_channels

        #Conv1:
        self.add_module(name='Conv1',module=SingleConv(in_channels=conv1_in_channels,
                                                       out_channels=conv1_out_channels,
                                                       kernel_size=kernel_size,
                                                       order=order,
                                                       GroupNumber=GroupNumber))
        #Conv2:
        self.add_module(name='Conv2',module=SingleConv(in_channels=conv2_in_channels,
                                                       out_channels=conv2_out_channels,
                                                       kernel_size=kernel_size,
                                                       order=order,
                                                       GroupNumber=GroupNumber))

"""********************Unet3D cSEUnet3D scSEUnet3D ... share these functions*******************"""


##############################################################################
# 3D DSN
##############################################################################
"""*******************************************DSN_UNet3D***************************************"""
class DSN_UNet3D_Encoder(nn.Module):
    """
    Encode the network
    """
    def __init__(self,in_channels,out_channels,counter,kernel_size=3,pool_kernelsize=(2,2,2),
                 pooling_type='max',apply_pooling=True,Basic_Module = DoubleConv, order ='crg',GroupNumber=8):
        super(DSN_UNet3D_Encoder,self).__init__()
        assert pooling_type in ['max','avg'],'Pooling_Type must be max or avg'
        if apply_pooling:
            if pooling_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernelsize)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernelsize)
        else:
            self.pooling = None

        self.basic_module = Basic_Module(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         encoder=True,order=order,
                                         GroupNumber=GroupNumber)



    def forward(self,x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x

class DSN_UNet3D_Decoder(nn.Module):
    """
    Decode the network
    """
    def __init__(self,in_channels,out_channels,use_attention,kernel_size=3,pool_kernelsize=(2,2,2),
                 Basic_Module = DoubleConv, order ='crb',GroupNumber=8):
        super(DSN_UNet3D_Decoder,self).__init__()
        self.use_attention = use_attention
        self.basic_module = Basic_Module(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         encoder=False,order=order,
                                         GroupNumber=GroupNumber)
        # self.convT3d = nn.ConvTranspose3d(in_channels=out_channels*2,
        #                                   out_channels=out_channels*2,
        #                                   kernel_size=kernel_size,
        #                                   stride=2,
        #                                   padding=1,
        #                                   output_padding=1)


    def forward(self,encoder_feature,x):
        output_size = encoder_feature.size()[2:]
        x = F.interpolate(input=x,size=output_size,mode='trilinear')
        # x = self.convT3d(x)
        #print(x.shape)
        x = torch.cat((encoder_feature,x),dim=1)
        x = self.basic_module(x)
        return x

class DSN_UNet3D(nn.Module):
    """
    3D deeply supervised network for automated segmentation of volumetric medical images
    http://www.cse.cuhk.edu.hk/~qdou/papers/2017/[2017][MedIA]3D%20deeply%20supervised%20network%20for%20automated%20segmentation%20of%20volumetric%20medical%20images.pdf
    """
    def __init__(self,in_channels,out_channels,finalsigmoid,fmaps_degree,GroupNormNumber,
                 fmaps_layer_number,layer_order,use_attention,**kwargs):
        super(DSN_UNet3D,self).__init__()
        assert isinstance(fmaps_degree,int) , 'fmaps_degree must be an integer!'
        fmaps_list = create_feature_maps(fmaps_degree,fmaps_layer_number)
        EncoderLists = []
        for i,out_feature_num in enumerate(fmaps_list):
            if i == 0:
                encoder = DSN_UNet3D_Encoder(in_channels=in_channels,
                                             out_channels=out_feature_num,
                                             apply_pooling=False,
                                             Basic_Module=DoubleConv,
                                             order=layer_order,
                                             GroupNumber=GroupNormNumber,
                                             counter=i)
            else:
                encoder = DSN_UNet3D_Encoder(in_channels=fmaps_list[i-1],
                                         out_channels=out_feature_num,
                                         apply_pooling=True,
                                         Basic_Module=DoubleConv,
                                         order=layer_order,
                                         GroupNumber=GroupNormNumber,
                                         counter=i)
            EncoderLists.append(encoder)
        self.encoders = nn.ModuleList(EncoderLists)

        DecoderLists = []
        DecoderFmapList = list(reversed(fmaps_list))
        for i in range(len(DecoderFmapList)-1):
            in_feature_num  = DecoderFmapList[i] + DecoderFmapList[i+1]
            out_feature_num = DecoderFmapList[i+1]
            decoder = DSN_UNet3D_Decoder(in_channels=in_feature_num,
                                     out_channels=out_feature_num,
                                     Basic_Module=DoubleConv,
                                     order=layer_order,
                                     GroupNumber=GroupNormNumber,
                                     use_attention=use_attention)
            DecoderLists.append(decoder)
        self.decoders = nn.ModuleList(DecoderLists)

        DecoderTransposedList=[]
        for i in range(len(DecoderFmapList)-2):
            DT_in_channels = DecoderFmapList[i+1]
            decodertransposed = nn.ConvTranspose3d(in_channels=DT_in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=3,
                                                   stride=2**(len(DecoderFmapList)-2-i),
                                                   dilation=2**(len(DecoderFmapList)-3-i),
                                                   padding=1,
                                                   output_padding=1)
            DecoderTransposedList.append(decodertransposed)
        self.decodertransposeds = nn.ModuleList(DecoderTransposedList)


        self.final_conv = nn.Conv3d(in_channels=fmaps_list[0],out_channels=out_channels,kernel_size=1)

        if finalsigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        encoder_features=[]
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.insert(0,x)
        encoder_features=encoder_features[1:]


        decoder_transposed_features = []
        for i,(encoder_feature,decoder) in enumerate(zip(encoder_features,self.decoders)):
            x = decoder(encoder_feature,x)
            if (i != len(encoder_features)-1):
                decoder_transposed_features.insert(0,self.decodertransposeds[i](x))

        x = self.final_conv(x)
        if not self.training:
            x = self.final_activation(x)
            for i in range(len(decoder_transposed_features)):
                decoder_transposed_features[i] = self.final_activation(decoder_transposed_features[i])


        out=[feature for feature in decoder_transposed_features]
        out.insert(0,x)
        return out

"""*******************************************DSN_UNet3D***************************************"""


##############################################################################
# Unet2D
##############################################################################
"""*********************************************Unet2d*****************************************"""
class Unet2d_ChannelPool(nn.Module):
    def forward(self,x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(dim=1),torch.mean(x,1).unsqueeze(dim=1)),dim=1)

class Unet2d_SpatialAttention(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,stride):
        super(Unet2d_SpatialAttention, self).__init__()
        self.spatialconv = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=kernel_size,
                                                   padding=padding,
                                                   stride=stride),
                                         nn.ReLU(inplace=True),
                                         nn.BatchNorm2d(out_channels))
        self.spatialacti = nn.Sigmoid()

    def forward(self,x):
        x = self.spatialconv(x)
        x = self.spatialacti(x)
        return x

class Unet2d_SpatialGate(nn.Module):
    def __init__(self,kernel_size=3):
        super(Unet2d_SpatialGate, self).__init__()
        self.kernel_size = kernel_size
        self.compress = Unet2d_ChannelPool()
        self.spatial  = Unet2d_SpatialAttention(in_channels=2,out_channels=1,kernel_size=self.kernel_size,padding=(self.kernel_size - 1)//2,stride=1)

    def forward(self,x):
        x_compress = self.compress(x)
        x_att = self.spatial(x_compress)
        x = x * x_att.expand_as(x)
        return x,x_att

#下采样2维卷积模块封装。
class Unet2d_ConvBlock2D(nn.Module):
    #初始时，继承重载，装入新的参数
    def __init__(self,in_ch,out_ch):
        super(Unet2d_ConvBlock2D).__init__()
        #装载到有序容器中
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,stride=1,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch,stride=1,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

#下采样时maxpool+conv
class Unet2d_Max_and_Conv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet2d_Max_and_Conv).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            Unet2d_ConvBlock2D(in_ch=in_ch,out_ch=out_ch)
        )
    def forward(self, x):
        x = self.layer(x)
        return x

#反卷积模块
class Unet2d_UPblock2D(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet2d_UPblock2D).__init__()
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1,dilation=1),
            #原文中用到了上恢复采样
            #nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )
        self.conv = nn.Sequential(
            Unet2d_ConvBlock2D(in_ch=in_ch,out_ch=out_ch)
        )
    def forward(self, x,down_feature):
        x = self.upconv(x)
        x = torch.cat([x,down_feature],dim=1)
        x = self.conv(x)
        return x

class Unet2D(nn.Module):
    def __init__(self,in_ch=4,out_ch=2,degree=64,show_attention=True):
        super(Unet2D,self).__init__()#python3特有
        self.show_attention = show_attention
        #或者super(Unet2d, self).__init__()
        channels=[]
        #下采样64 128 256 512 1024
        for i in range(5):
            #channels.append(int(math.pow(2,i))*degree)#浮点数转int，pytorch中new()方法只支持整形
            channels.append((2**i)*degree)
        #下采样模块
        self.downlayer1 = Unet2d_ConvBlock2D(in_ch,channels[0])
        self.downlayer2 = Unet2d_Max_and_Conv(channels[0],channels[1])
        self.downlayer3 = Unet2d_Max_and_Conv(channels[1],channels[2])
        self.downlayer4 = Unet2d_Max_and_Conv(channels[2],channels[3])
        self.downlayer5 = Unet2d_Max_and_Conv(channels[3],channels[4])

        self.sam = Unet2d_SpatialGate(kernel_size=3)

        #上恢复模块
        self.uplayer1 = Unet2d_UPblock2D(in_ch=channels[4],out_ch=channels[3])
        self.uplayer2 = Unet2d_UPblock2D(in_ch=channels[3],out_ch=channels[2])
        self.uplayer3 = Unet2d_UPblock2D(in_ch=channels[2],out_ch=channels[1])
        self.uplayer4 = Unet2d_UPblock2D(in_ch=channels[1],out_ch=channels[0])

        #图像大小保持不变
        self.outlayer = nn.Conv2d(in_channels=channels[0],
                                  out_channels=out_ch,kernel_size=3,
                                  stride=1,padding=1)

        # self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        x_dl1 =self.downlayer1(x)
        x_dl2 =self.downlayer2(x_dl1)
        x_dl3 =self.downlayer3(x_dl2)
        x_dl4 =self.downlayer4(x_dl3)
        x_dl5 =self.downlayer5(x_dl4)

        x_product1,x_att1 = self.sam(x_dl5)

        if(self.show_attention):

            x_att1 = x_att1.data.cpu().numpy()
            bz,nc,h,w = x_att1.shape
            #print(bz,nc,h,w)
            x_att1 = np.reshape(x_att1,(h,w))
            x_att1_image = (x_att1 - np.min(x_att1))/np.max(x_att1)
            x_att1_image = np.uint8(255*x_att1_image)
            x_att1_image = cv2.resize(x_att1_image,(256,256))
            #print(np.max(x_att1_image),np.min(x_att1_image))
            heatmap = cv2.applyColorMap(x_att1_image, cv2.COLORMAP_JET)

            cv2.imshow("heatmap",heatmap)
            cv2.waitKey(0)

        #x_ul4 = self.uplayer1(x_dl5,x_dl4)
        x_ul4 = self.uplayer1(x_product1,x_dl4)
        x_ul3 = self.uplayer2(x_ul4,x_dl3)
        x_ul2 = self.uplayer3(x_ul3,x_dl2)
        x_ul1 = self.uplayer4(x_ul2,x_dl1)
        x = self.outlayer(x_ul1)
        # if not self.training:
        #     x = self.final_activation(x)
        return x

"""*********************************************Unet2d*****************************************"""


"""******************************************Unet3D********************************************"""
##############################################################################
# UNet3D
##############################################################################

#Encoder
class UNet3D_Encoder(nn.Module):
    """
    Encode the network
    """
    def __init__(self,in_channels,out_channels,kernel_size=3,pool_kernelsize=(2,2,2),
                 pooling_type='max',apply_pooling=True,Basic_Module = DoubleConv, order ='crg',GroupNumber=8):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param pool_kernelsize:
        :param pooling_type:
        :param apply_pooling:
        :param Basic_Module:
        :param order:
        :param GroupNumber:
        """
        super(UNet3D_Encoder,self).__init__()
        assert pooling_type in ['max','avg'],'Pooling_Type must be max or avg'
        if apply_pooling:
            if pooling_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernelsize)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernelsize)
        else:
            self.pooling = None

        self.basic_module = Basic_Module(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         encoder=True,order=order,
                                         GroupNumber=GroupNumber)
        #CBAM spatial attention
        #self.attention_module = _CBAM_SpatialGate(kernel_size=7,relu=False,IN=True)
		#Split Attention
        #self.channel_splitattention =_Split_Channel_Attention(in_channels=out_channels,out_channels=out_channels)
		
    def forward(self,x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        # CBAM spatial attention
        #x = self.attention_module(x)
		#Split Attention
        #x = self.channel_splitattention(x)
        return x

#Decoder
class UNet3D_Decoder(nn.Module):
    """
    Decode the network
    """
    def __init__(self,in_channels,out_channels,kernel_size=3,pool_kernelsize=(2,2,2),
                 Basic_Module = DoubleConv, order ='crb',GroupNumber=8):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param pool_kernelsize:
        :param Basic_Module:
        :param order:
        :param GroupNumber:
        """
        super(UNet3D_Decoder,self).__init__()
        if Basic_Module == DoubleConv:
            self.upsample = None
        self.basic_module = Basic_Module(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         encoder=False,order=order,
                                         GroupNumber=GroupNumber)
		    #attention
        #self.attention_module = _GridAttentionBlock3D(in_channels=out_channels,inter_channels=out_channels//2,gate_channels=out_channels*2)

        #CBAM spatial attention
        #self.attention_module = _CBAM_SpatialGate(kernel_size=7,relu=False,IN=True)
		
 		#Split Attention       
        #self.channel_splitattention =_Split_Channel_Attention(in_channels=out_channels,out_channels=out_channels)
		
    def forward(self,encoder_feature,x):
        if self.upsample is None:
            #encoder's feature's ---> D*H*W
            output_size = encoder_feature.size()[2:]
            #encoder_feature,att_map = self.attention_module(encoder_feature,x)
            x = F.interpolate(input=x,size=output_size,mode='nearest')
            x = torch.cat((encoder_feature,x),dim=1)
        x = self.basic_module(x)
		#Split Attention
        #x = self.channel_splitattention(x)
        # CBAM spatial attention
        #x = self.attention_module(x)
        return x

#UNet3D
class UNet3D(nn.Module):
    """
    @Compiled by zmh from scratch
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.
    """
    def __init__(self,in_channels,out_channels,finalsigmoid,fmaps_degree,GroupNormNumber,
                 fmaps_layer_number,layer_order,device,**kwargs):
        super(UNet3D,self).__init__()
        self.device = device
        assert isinstance(fmaps_degree,int) , 'fmaps_degree must be an integer!'
        fmaps_list = create_feature_maps(fmaps_degree,fmaps_layer_number)
        
        self.EncoderLayer1 = UNet3D_Encoder(in_channels=in_channels,out_channels=fmaps_list[0],apply_pooling=False,
                                            Basic_Module=DoubleConv,order=layer_order,
                                            GroupNumber=GroupNormNumber).to(self.device[0])
        self.EncoderLayer2 = UNet3D_Encoder(in_channels=fmaps_list[0],out_channels=fmaps_list[1],apply_pooling=True,
                                            Basic_Module=DoubleConv,order=layer_order,
                                            GroupNumber=GroupNormNumber).to(self.device[0])
        self.EncoderLayer3 = UNet3D_Encoder(in_channels=fmaps_list[1],out_channels=fmaps_list[2],apply_pooling=True,
                                            Basic_Module=DoubleConv,order=layer_order,
                                            GroupNumber=GroupNormNumber).to(self.device[0])
        self.EncoderLayer4 = UNet3D_Encoder(in_channels=fmaps_list[2],out_channels=fmaps_list[3],apply_pooling=True,
                                            Basic_Module=DoubleConv,order=layer_order,
                                            GroupNumber=GroupNormNumber).to(self.device[0])
        
        DecoderFmapList = list(reversed(fmaps_list))
        
        self.DecoderLayer1 = UNet3D_Decoder(in_channels=DecoderFmapList[0]+DecoderFmapList[1],
                                            out_channels=DecoderFmapList[1],
                                            Basic_Module=DoubleConv,order=layer_order,GroupNumber=GroupNormNumber).to(self.device[0])
        self.DecoderLayer2 = UNet3D_Decoder(in_channels=DecoderFmapList[1]+DecoderFmapList[2],
                                            out_channels=DecoderFmapList[2],
                                            Basic_Module=DoubleConv,order=layer_order,GroupNumber=GroupNormNumber).to(self.device[0])
        self.DecoderLayer3 = UNet3D_Decoder(in_channels=DecoderFmapList[2]+DecoderFmapList[3],
                                            out_channels=DecoderFmapList[3],
                                            Basic_Module=DoubleConv,order=layer_order,GroupNumber=GroupNormNumber).to(self.device[-1])
        
        self.final_conv = nn.Conv3d(in_channels=fmaps_list[0], out_channels=out_channels, kernel_size=1).to(self.device[-1])
        
        if finalsigmoid:
            self.final_activation = nn.Sigmoid().to(self.device[-1])
        else:
            self.final_activation = nn.Softmax(dim=1).to(self.device[-1])


    def forward(self, x):
        encoder_features = []
        x1 = self.EncoderLayer1(x)
        encoder_features.insert(0,x1.to(self.device[-1]))
        x2 = self.EncoderLayer2(x1)
        encoder_features.insert(0,x2)
        x3 = self.EncoderLayer3(x2)
        encoder_features.insert(0,x3)
        x4 = self.EncoderLayer4(x3)


        x  = self.DecoderLayer1(encoder_features[0],x4)
        x  = self.DecoderLayer2(encoder_features[1],x).to(self.device[-1])
        x  = self.DecoderLayer3(encoder_features[2],x)

        x  = self.final_conv(x)

        if not self.training:
            x = self.final_activation(x)
        return x

"""******************************************Unet3D********************************************"""

"""******************************************PE UNet3D********************************************"""
#PE module
class ProjectExciteLayer(nn.Module):
    """
        Project & Excite Module, specifically designed for 3D inputs
        *quote*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ProjectExciteLayer, self).__init__()
        self.reduction_ratio = reduction_ratio
        num_channels_reduced = int(num_channels // self.reduction_ratio)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.PReLU(num_parameters=num_channels_reduced)
        self.conv_c = nn.Conv3d(in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1)
        self.conv_cT = nn.Conv3d(in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()

        # Project:
        # Average along channels and different axes
        squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (1, 1, W))

        squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (1, H, 1))

        squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (D, 1, 1))
        
        final_squeeze_tensor = squeeze_tensor_d+squeeze_tensor_h+squeeze_tensor_w
        final_squeeze_tensor = torch.div((squeeze_tensor_d+squeeze_tensor_h+squeeze_tensor_w),3.0)

        # tile tensors to original size and add:
        #final_squeeze_tensor = sum([squeeze_tensor_w.view(batch_size, num_channels, 1, 1, W),
                                    #squeeze_tensor_h.view(batch_size, num_channels, 1, H, 1),
                                    #squeeze_tensor_d.view(batch_size, num_channels, D, 1, 1)])
        #print(final_squeeze_tensor.shape)
        # Excitation:
        final_squeeze_tensor = self.sigmoid(self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor))))
        output_tensor = torch.mul(input_tensor, final_squeeze_tensor)

        return output_tensor

#PE Encoder
class PEUNet3D_Encoder(nn.Module):
    """
    Encode the network
    """
    def __init__(self,in_channels,out_channels,channel_hierarchy,kernel_size=3,pool_kernelsize=(2,2,2),
                 pooling_type='max',apply_pooling=True,Basic_Module = DoubleConv, order ='crg',GroupNumber=8):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param pool_kernelsize:
        :param pooling_type:
        :param apply_pooling:
        :param Basic_Module:
        :param order:
        :param GroupNumber:
        """
        super(PEUNet3D_Encoder,self).__init__()
        assert pooling_type in ['max','avg'],'Pooling_Type must be max or avg'
        if apply_pooling:
            if pooling_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernelsize)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernelsize)
        else:
            self.pooling = None

        self.basic_module = Basic_Module(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         encoder=True,order=order,
                                         GroupNumber=GroupNumber)
        self.pe_layer = ProjectExciteLayer(num_channels=out_channels,reduction_ratio=channel_hierarchy)

    def forward(self,x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        x = self.pe_layer(x)
        return x

#PE Decoder
class PEUNet3D_Decoder(nn.Module):
    """
    Decode the network
    """
    def __init__(self,in_channels,out_channels,channel_hierarchy,kernel_size=3,pool_kernelsize=(2,2,2),
                 Basic_Module = DoubleConv, order ='crb',GroupNumber=8):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param pool_kernelsize:
        :param Basic_Module:
        :param order:
        :param GroupNumber:
        """
        super(PEUNet3D_Decoder,self).__init__()

        self.basic_module = Basic_Module(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         encoder=False,order=order,
                                         GroupNumber=GroupNumber)
        #self.pe_layer = ProjectExciteLayer(num_channels=out_channels,reduction_ratio=channel_hierarchy)

    def forward(self,encoder_feature,x):
        #encoder's feature's ---> D*H*W
        output_size = encoder_feature.size()[2:]
        x = F.interpolate(input=x,size=output_size,mode='trilinear')
        x = torch.cat((encoder_feature,x),dim=1)
        x = self.basic_module(x)
        #x = self.pe_layer(x)
        return x


#UNet3D
class PEUNet3d(nn.Module):
    """
    @Compiled by zmh from scratch
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.
    """
    def __init__(self,in_channels,out_channels,finalsigmoid,fmaps_degree,GroupNormNumber,
                 fmaps_layer_number,layer_order,channel_hierarchy,device,**kwargs):
        super(PEUNet3d,self).__init__()
        self.device = device
        assert isinstance(fmaps_degree,int) , 'fmaps_degree must be an integer!'
        fmaps_list = create_feature_maps(fmaps_degree,fmaps_layer_number)
        
        self.EncoderLayer1 = PEUNet3D_Encoder(in_channels=in_channels,
                                              out_channels=fmaps_list[0],
                                              channel_hierarchy=channel_hierarchy,
                                              apply_pooling=False,
                                              Basic_Module=DoubleConv,
                                              order=layer_order,
                                              GroupNumber=GroupNormNumber).to(self.device[0])
                                              
        self.EncoderLayer2 = PEUNet3D_Encoder(in_channels=fmaps_list[0],
                                              out_channels=fmaps_list[1],
                                              channel_hierarchy=channel_hierarchy,
                                              apply_pooling=True,
                                              Basic_Module=DoubleConv,order=layer_order,
                                              GroupNumber=GroupNormNumber).to(self.device[0])
                                              
        self.EncoderLayer3 = PEUNet3D_Encoder(in_channels=fmaps_list[1],
                                              out_channels=fmaps_list[2],
                                              channel_hierarchy=channel_hierarchy,
                                              apply_pooling=True,
                                              Basic_Module=DoubleConv,
                                              order=layer_order,
                                              GroupNumber=GroupNormNumber).to(self.device[0])
                                              
        self.EncoderLayer4 = PEUNet3D_Encoder(in_channels=fmaps_list[2],
                                              out_channels=fmaps_list[3],
                                              channel_hierarchy=channel_hierarchy,
                                              apply_pooling=True,
                                              Basic_Module=DoubleConv,order=layer_order,
                                              GroupNumber=GroupNormNumber).to(self.device[0])
        
        DecoderFmapList = list(reversed(fmaps_list))
        
        self.DecoderLayer1 = PEUNet3D_Decoder(in_channels=DecoderFmapList[0]+DecoderFmapList[1],
                                              out_channels=DecoderFmapList[1],
                                              channel_hierarchy=channel_hierarchy,
                                              Basic_Module=DoubleConv,
                                              order=layer_order,
                                              GroupNumber=GroupNormNumber).to(self.device[0])
                                              
        self.DecoderLayer2 = PEUNet3D_Decoder(in_channels=DecoderFmapList[1]+DecoderFmapList[2],
                                              out_channels=DecoderFmapList[2],
                                              channel_hierarchy=channel_hierarchy,
                                              Basic_Module=DoubleConv,
                                              order=layer_order,
                                              GroupNumber=GroupNormNumber).to(self.device[0])
                                              
        self.DecoderLayer3 = PEUNet3D_Decoder(in_channels=DecoderFmapList[2]+DecoderFmapList[3],
                                              out_channels=DecoderFmapList[3],
                                              channel_hierarchy=channel_hierarchy,
                                              Basic_Module=DoubleConv,
                                              order=layer_order,
                                              GroupNumber=GroupNormNumber).to(self.device[-1])
        
        self.final_conv = nn.Conv3d(in_channels=fmaps_list[0], out_channels=out_channels, kernel_size=1).to(self.device[-1])
        
        if finalsigmoid:
            self.final_activation = nn.Sigmoid().to(self.device[-1])
        else:
            self.final_activation = nn.Softmax(dim=1).to(self.device[-1])


    def forward(self, x):
        encoder_features = []
        x1 = self.EncoderLayer1(x)
        encoder_features.insert(0,x1.to(self.device[-1]))
        x2 = self.EncoderLayer2(x1)
        encoder_features.insert(0,x2)
        x3 = self.EncoderLayer3(x2)
        encoder_features.insert(0,x3)
        x4 = self.EncoderLayer4(x3)


        x  = self.DecoderLayer1(encoder_features[0],x4)
        x  = self.DecoderLayer2(encoder_features[1],x).to(self.device[-1])
        x  = self.DecoderLayer3(encoder_features[2],x)

        x  = self.final_conv(x)

        if not self.training:
            x = self.final_activation(x)
        return x


"""****************************slice和channel的SE-module的通用部分********************************"""
##############################################################################
# scSEUNet3D
##############################################################################

#scSEUNet3D_create_slices_maps
def scSEUNet3D_create_slices_maps(init_image_shape,number_of_fmaps):
    Slices_maps=[]
    for i in range(number_of_fmaps):
        Slice_map = init_image_shape // (2 ** i)
        Slices_maps.append(Slice_map)
    return Slices_maps

#scSEUNet3D cSE-module
class scSEUNet3D_cSEmodule(nn.Module):
    """Modify 0116 zmh Add the SE_channel_pooling_type"""
    def __init__(self,channel,channel_hierarchy,SE_channel_pooling_type):
        super(scSEUNet3D_cSEmodule,self).__init__()
        self.SE_channel_pooling_type = SE_channel_pooling_type
        #reduction = int(channel/channel_hierarchy)
        if(channel_hierarchy>channel):
            reduction = channel
        else:
            reduction = channel_hierarchy
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True)
        )
        # final no-linear action function
        self.final_nlaf = nn.Sigmoid()

    def forward(self, x):
        if (self.SE_channel_pooling_type == 'avg'):
            b, c, d, h, w = x.size()
            y = self.avg_pool(x).view(b, c)
            y = (self.final_nlaf(self.fc(y))).view(b, c, 1, 1, 1)
            return x * y.expand_as(x)
        elif (self.SE_channel_pooling_type == 'max'):
            b, c, d, h, w = x.size()
            y = self.max_pool(x).view(b, c)
            y = (self.final_nlaf(self.fc(y))).view(b, c, 1, 1, 1)
            return x * y.expand_as(x)
        elif (self.SE_channel_pooling_type == 'avg_and_max'):
            b, c, d, h, w = x.size()
            y_avg = self.avg_pool(x).view(b, c)
            y_max = self.max_pool(x).view(b, c)
            y_avg_and_max = self.fc(y_avg) + self.fc(y_max)
            y = self.final_nlaf(y_avg_and_max).view(b, c, 1, 1, 1)
            return x * y.expand_as(x)
        else:
            raise ValueError('SE_channel_pooling_type must be avg,max,avg_and_max')
            return x


#SlicesSE-module
class scSEUNet3D_Eachchannel_slicesSEmodule(nn.Module):
    """Modify 0116 zmh Add the SE_slice_pooling_type"""
    def __init__(self,in_channels,reduction,SE_slice_pooling_type):
        super(scSEUNet3D_Eachchannel_slicesSEmodule,self).__init__()
        self.SE_slice_pooling = SE_slice_pooling_type
        #tailor for each H*W channel
        self.avg_pool_hw = nn.AdaptiveAvgPool2d(1)
        self.max_pool_hw = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels,in_channels//reduction,bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//reduction,in_channels,bias=True),
        )
        # final no-linear action function
        self.final_nlaf = nn.Sigmoid()

    def forward(self,x):
        if(self.SE_slice_pooling  == 'avg'):
            b,d,h,w = x.size()
            y = self.avg_pool_hw(x).view(b,d)
            y = (self.final_nlaf(self.fc(y))).view(b,d,1,1)
            with torch.no_grad():
                temp = y.view(d)
                #add probability output
                if(d==192):
                    with open('docs/heatmap/slice_heatmap_hvsmr.txt','a+') as f:
                        f.write(str(temp)+"\n")
                        f.close()
            x = x*y.expand_as(x)
            return x
        elif(self.SE_slice_pooling == 'max'):
            b,d,h,w = x.size()
            y = self.max_pool_hw(x).view(b,d)
            y = (self.final_nlaf(self.fc(y))).view(b,d,1,1)
            x = x*y.expand_as(x)
            return x
        elif(self.SE_slice_pooling == 'avg_and_max'):
            b, d, h, w = x.size()
            y_avg = self.avg_pool_hw(x).view(b,d)
            y_max = self.max_pool_hw(x).view(b,d)
            y_avg_and_max = self.fc(y_avg) + self.fc(y_max)
            y = self.final_nlaf(y_avg_and_max).view(b,d,1,1)
            x = x*y.expand_as(x)
            return x
        else:
            raise ValueError('SE_slice_pooling_type must be avg,max,avg_and_max')

class scSEUNet3D_Wholechannels_slicesSEmodule(nn.Module):
    """Modify 0116 zmh Add the SE_slice_pooling_type"""
    def __init__(self,in_channels,slices_number,slices_hierarchy,SE_slice_pooling_type):
        super(scSEUNet3D_Wholechannels_slicesSEmodule,self).__init__()
        Slice_SElists=[]
        #reduction = int(slices_number/slices_hierarchy)
        if(slices_hierarchy>slices_number):
            reduction = slices_number
        else:
            reduction = slices_hierarchy
        for i in range(in_channels):
            Slice_SEmodule = scSEUNet3D_Eachchannel_slicesSEmodule(in_channels=slices_number,reduction=reduction,SE_slice_pooling_type=SE_slice_pooling_type)
            Slice_SElists.append(Slice_SEmodule)
        self.Slice_SE_encoder = nn.ModuleList(Slice_SElists)
    def forward(self,x):
        for index,Slice_SEmodule in enumerate(self.Slice_SE_encoder):
            y = x[:,index,:,:,:].clone()
            y= Slice_SEmodule(y)
            x[:, index, :, :, :]=y
        return x

"""****************************slice和channel的SE-module的通用部分********************************"""


##############################################################################
# scSEUNet3D
##############################################################################
"""****************************************scSEUnet3D*******************************************"""
#Encoder
class scSEUNet3D_Encoder(nn.Module):
    """
    Encode the network
    """
    def __init__(self,in_channels,out_channels,slice_number,slices_hierarchy,channel_hierarchy,
                 SE_channel_pooling_type,SE_slice_pooling_type,kernel_size=3,pool_kernelsize=(2,2,2),
                 pooling_type='max',apply_pooling=True,Basic_Module = DoubleConv, order ='crg',GroupNumber=8):
        """
        :param in_channels:
        :param out_channels:
        :param slice_number: how many slices need to be SE
        :param kernel_size:
        :param pool_kernelsize:
        :param pooling_type:
        :param apply_pooling:
        :param Basic_Module:
        :param order:
        :param GroupNumber:
        """
        super(scSEUNet3D_Encoder,self).__init__()
        assert pooling_type in ['max','avg'],'Pooling_Type must be max or avg'
        if apply_pooling:
            if pooling_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernelsize)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernelsize)
        else:
            self.pooling = None

        self.basic_module = Basic_Module(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         encoder=True,order=order,
                                         GroupNumber=GroupNumber)
        """
        @Anno by zmh
        append Slice_SElists by each encoder layer
        append Channel_SE    by each encoder layer
        """
        self.encoder_sSE = scSEUNet3D_Wholechannels_slicesSEmodule(in_channels=out_channels,
                                                                   slices_number=slice_number,
                                                                   slices_hierarchy=slices_hierarchy,
                                                                   SE_slice_pooling_type=SE_slice_pooling_type)


        self.encoder_cSE  = scSEUNet3D_cSEmodule(channel=out_channels,
                                                 channel_hierarchy=channel_hierarchy,
                                                 SE_channel_pooling_type=SE_channel_pooling_type)

    def forward(self,x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        x = self.encoder_sSE(x)
        x = self.encoder_cSE(x)
        return x

#Decoder
class scSEUNet3D_Decoder(nn.Module):
    """
    Decode the network
    """
    def __init__(self,in_channels,out_channels,slice_number,slices_hierarchy,channel_hierarchy,
                 SE_channel_pooling_type,SE_slice_pooling_type,kernel_size=3,pool_kernelsize=(2,2,2),
                 Basic_Module = DoubleConv, order ='crb',GroupNumber=8):
        """

        :param in_channels:
        :param out_channels:
        :param slice_number:
        :param kernel_size:
        :param pool_kernelsize:
        :param Basic_Module:
        :param order:
        :param GroupNumber:
        """
        super(scSEUNet3D_Decoder,self).__init__()
        if Basic_Module == DoubleConv:
            #interpolate not transpose3d
            self.upsample = None
        self.basic_module = Basic_Module(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         encoder=False,order=order,
                                         GroupNumber=GroupNumber)
        """
         @Anno by zmh
         append Channel_SE    by each decoder layer
         append Slice_SElists by each decoder layer
         """
        self.decoder_sSE = scSEUNet3D_Wholechannels_slicesSEmodule(in_channels=out_channels,
                                                                   slices_number=slice_number,
                                                                   slices_hierarchy=slices_hierarchy,
                                                                   SE_slice_pooling_type=SE_slice_pooling_type)

        self.decoder_cSE = scSEUNet3D_cSEmodule(channel=out_channels,
                                                channel_hierarchy=channel_hierarchy,
                                                SE_channel_pooling_type=SE_channel_pooling_type)


    def forward(self,encoder_feature,x):
        if self.upsample is None:
            #encoder's feature's ---> D*H*W
            output_size = encoder_feature.size()[2:]
            x = F.interpolate(input=x,size=output_size,mode='nearest')
            x = torch.cat((encoder_feature,x),dim=1)
        x = self.basic_module(x)
        x = self.decoder_sSE(x)
        x = self.decoder_cSE(x)
        return x

#scSEUNet3D
class scSEUNet3D(nn.Module):
    """
    @Compiled by zmh from scratch
    scSE_UNet3D
        image_shape  decides the dynamic number of slices SEmodule
    """
    """Modify 0116 zmh Add the SE_channel_pooling_type and SE_slice_pooling_type"""
    def __init__(self,in_channels,out_channels,depth,finalsigmoid,fmaps_degree,GroupNormNumber,
                 fmaps_layer_number,layer_order,slices_hierarchy,channel_hierarchy,
                 SE_channel_pooling_type,SE_slice_pooling_type,device,**kwargs):
        super(scSEUNet3D,self).__init__()
        self.device = device
        assert isinstance(fmaps_degree,int) , 'fmaps_degree must be an integer!'
        fmaps_list      = create_feature_maps(fmaps_degree,fmaps_layer_number)
        #e.g [128,64,32,16]
        Slicesmaps_list = scSEUNet3D_create_slices_maps(depth,fmaps_layer_number)
        self.EncoderLayer1 = scSEUNet3D_Encoder(in_channels=in_channels,
                                                out_channels=fmaps_list[0],
                                                slice_number=Slicesmaps_list[0],
                                                apply_pooling=False,
                                                Basic_Module=DoubleConv,
                                                order=layer_order,
                                                GroupNumber=GroupNormNumber,
                                                slices_hierarchy=slices_hierarchy,
                                                channel_hierarchy=channel_hierarchy,
                                                SE_channel_pooling_type=SE_channel_pooling_type,
                                                SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.EncoderLayer2 = scSEUNet3D_Encoder(in_channels=fmaps_list[0],
                                                out_channels=fmaps_list[1],
                                                slice_number=Slicesmaps_list[1],
                                                apply_pooling=True,
                                                Basic_Module=DoubleConv,
                                                order=layer_order,
                                                GroupNumber=GroupNormNumber,
                                                slices_hierarchy=slices_hierarchy,
                                                channel_hierarchy=channel_hierarchy,
                                                SE_channel_pooling_type=SE_channel_pooling_type,
                                                SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.EncoderLayer3 = scSEUNet3D_Encoder(in_channels=fmaps_list[1],
                                                out_channels=fmaps_list[2],
                                                slice_number=Slicesmaps_list[2],
                                                apply_pooling=True,
                                                Basic_Module=DoubleConv,
                                                order=layer_order,
                                                GroupNumber=GroupNormNumber,
                                                slices_hierarchy=slices_hierarchy,
                                                channel_hierarchy=channel_hierarchy,
                                                SE_channel_pooling_type=SE_channel_pooling_type,
                                                SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.EncoderLayer4 = scSEUNet3D_Encoder(in_channels=fmaps_list[2],
                                                out_channels=fmaps_list[3],
                                                slice_number=Slicesmaps_list[3],
                                                apply_pooling=True,
                                                Basic_Module=DoubleConv,
                                                order=layer_order,
                                                GroupNumber=GroupNormNumber,
                                                slices_hierarchy=slices_hierarchy,
                                                channel_hierarchy=channel_hierarchy,
                                                SE_channel_pooling_type=SE_channel_pooling_type,
                                                SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])
        
        DecoderFmapList = list(reversed(fmaps_list))
        DecoderSmapList = list(reversed(Slicesmaps_list))
        
        self.DecoderLayer1 = scSEUNet3D_Decoder(in_channels=DecoderFmapList[0]+DecoderFmapList[1],
                                                out_channels=DecoderFmapList[1],
                                                slice_number=DecoderSmapList[1],
                                                Basic_Module=DoubleConv,
                                                order=layer_order,
                                                GroupNumber=GroupNormNumber,
                                                slices_hierarchy=slices_hierarchy,
                                                channel_hierarchy=channel_hierarchy,
                                                SE_channel_pooling_type=SE_channel_pooling_type,
                                                SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.DecoderLayer2 = scSEUNet3D_Decoder(in_channels=DecoderFmapList[1]+DecoderFmapList[2],
                                                out_channels=DecoderFmapList[2],
                                                slice_number=DecoderSmapList[2],
                                                Basic_Module=DoubleConv,
                                                order=layer_order,
                                                GroupNumber=GroupNormNumber,
                                                slices_hierarchy=slices_hierarchy,
                                                channel_hierarchy=channel_hierarchy,
                                                SE_channel_pooling_type=SE_channel_pooling_type,
                                                SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.DecoderLayer3 = scSEUNet3D_Decoder(in_channels=DecoderFmapList[2]+DecoderFmapList[3],
                                                out_channels=DecoderFmapList[3],
                                                slice_number=DecoderSmapList[3],
                                                Basic_Module=DoubleConv,
                                                order=layer_order,
                                                GroupNumber=GroupNormNumber,
                                                slices_hierarchy=slices_hierarchy,
                                                channel_hierarchy=channel_hierarchy,
                                                SE_channel_pooling_type=SE_channel_pooling_type,
                                                SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[-1])
        
        self.final_conv = nn.Conv3d(in_channels=fmaps_list[0], out_channels=out_channels, kernel_size=1).to(self.device[-1])
        
        if finalsigmoid:
            self.final_activation = nn.Sigmoid().to(self.device[-1])
        else:
            self.final_activation = nn.Softmax(dim=1).to(self.device[-1])


    def forward(self, x):
        encoder_features = []
        x1 = self.EncoderLayer1(x)
        encoder_features.insert(0,x1.to(self.device[-1]))
        x2 = self.EncoderLayer2(x1)
        encoder_features.insert(0,x2)
        x3 = self.EncoderLayer3(x2)
        encoder_features.insert(0,x3)
        x4 = self.EncoderLayer4(x3)


        x  = self.DecoderLayer1(encoder_features[0],x4)
        x  = self.DecoderLayer2(encoder_features[1],x).to(self.device[-1])
        x  = self.DecoderLayer3(encoder_features[2],x)

        x  = self.final_conv(x)

        if not self.training:
            x = self.final_activation(x)
        return x

"""****************************************scSEUnet3D*******************************************"""


"""****************************************csSEUnet3D*******************************************"""
##############################################################################
# csSEUNet3D
##############################################################################
#Encoder
class csSEUNet3D_Encoder(nn.Module):
    """
    Encode the network
    """
    def __init__(self,in_channels,out_channels,slice_number,slices_hierarchy,channel_hierarchy,
                 SE_channel_pooling_type, SE_slice_pooling_type,kernel_size=3,pool_kernelsize=(2,2,2),
                 pooling_type='max',apply_pooling=True,Basic_Module = DoubleConv, order ='crg',GroupNumber=8):
        super(csSEUNet3D_Encoder,self).__init__()
        assert pooling_type in ['max','avg'],'Pooling_Type must be max or avg'
        if apply_pooling:
            if pooling_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernelsize)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernelsize)
        else:
            self.pooling = None

        self.basic_module = Basic_Module(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         encoder=True,order=order,
                                         GroupNumber=GroupNumber)
        """
        @Anno by zmh
        append Slice_SElists by each encoder layer
        append Channel_SE    by each encoder layer
        """
        self.encoder_cSE  = scSEUNet3D_cSEmodule(channel=out_channels,
                                                 channel_hierarchy=channel_hierarchy,
                                                 SE_channel_pooling_type=SE_channel_pooling_type)
        
        self.encoder_sSE = scSEUNet3D_Wholechannels_slicesSEmodule(in_channels=out_channels,
                                                                   slices_number=slice_number,
                                                                   slices_hierarchy=slices_hierarchy,
                                                                   SE_slice_pooling_type=SE_slice_pooling_type)


    def forward(self,x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        #follow the order of cSE-->sSE
        x = self.encoder_cSE(x)
        x = self.encoder_sSE(x)
        return x

#Decoder
class csSEUNet3D_Decoder(nn.Module):
    """
    Decode the network
    """
    def __init__(self,in_channels,out_channels,slice_number,slices_hierarchy,channel_hierarchy,
                 SE_channel_pooling_type, SE_slice_pooling_type,kernel_size=3,pool_kernelsize=(2,2,2),
                 Basic_Module = DoubleConv, order ='crb',GroupNumber=8):
        super(csSEUNet3D_Decoder,self).__init__()
        if Basic_Module == DoubleConv:
            #interpolate not transpose3d
            self.upsample = None
        self.basic_module = Basic_Module(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         encoder=False,order=order,
                                         GroupNumber=GroupNumber)
        """
         @Anno by zmh
         append Channel_SE    by each decoder layer
         append Slice_SElists by each decoder layer
         """
        self.decoder_cSE = scSEUNet3D_cSEmodule(channel=out_channels,
                                                channel_hierarchy=channel_hierarchy,
                                                SE_channel_pooling_type=SE_channel_pooling_type)

        self.decoder_sSE = scSEUNet3D_Wholechannels_slicesSEmodule(in_channels=out_channels,
                                                                   slices_number=slice_number,
                                                                   slices_hierarchy=slices_hierarchy,
                                                                   SE_slice_pooling_type=SE_slice_pooling_type)

    def forward(self,encoder_feature,x):
        if self.upsample is None:
            #encoder's feature's ---> D*H*W
            output_size = encoder_feature.size()[2:]
            x = F.interpolate(input=x,size=output_size,mode='nearest')
            x = torch.cat((encoder_feature,x),dim=1)
        x = self.basic_module(x)
        x = self.decoder_cSE(x)
        x = self.decoder_sSE(x)
        return x

#csSEUNet3D
class csSEUNet3D(nn.Module):
    """
    @Compiled by zmh from scratch
    csSE_UNet3D
        image_shape  decides the dynamic number of slices SEmodule
    """
    def __init__(self,in_channels,out_channels,depth,finalsigmoid,fmaps_degree,GroupNormNumber,
                 fmaps_layer_number,layer_order,slices_hierarchy,channel_hierarchy,
                 SE_channel_pooling_type, SE_slice_pooling_type,device,**kwargs):
        super(csSEUNet3D,self).__init__()
        self.device = device
        assert isinstance(fmaps_degree,int) , 'fmaps_degree must be an integer!'
        fmaps_list      = create_feature_maps(fmaps_degree,fmaps_layer_number)
        #e.g [128,64,32,16]
        # 可与scSEUNet3d 通用
        Slicesmaps_list = scSEUNet3D_create_slices_maps(depth,fmaps_layer_number)
        
        self.EncoderLayer1 = csSEUNet3D_Encoder(in_channels=in_channels,
                                                out_channels=fmaps_list[0],
                                                slice_number=Slicesmaps_list[0],
                                                apply_pooling=False,
                                                Basic_Module=DoubleConv,
                                                order=layer_order,
                                                GroupNumber=GroupNormNumber,
                                                slices_hierarchy=slices_hierarchy,
                                                channel_hierarchy=channel_hierarchy,
                                                SE_channel_pooling_type=SE_channel_pooling_type,
                                                SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.EncoderLayer2 = csSEUNet3D_Encoder(in_channels=fmaps_list[0],
                                                out_channels=fmaps_list[1],
                                                slice_number=Slicesmaps_list[1],
                                                apply_pooling=True,
                                                Basic_Module=DoubleConv,
                                                order=layer_order,
                                                GroupNumber=GroupNormNumber,
                                                slices_hierarchy=slices_hierarchy,
                                                channel_hierarchy=channel_hierarchy,
                                                SE_channel_pooling_type=SE_channel_pooling_type,
                                                SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.EncoderLayer3 = csSEUNet3D_Encoder(in_channels=fmaps_list[1],
                                                out_channels=fmaps_list[2],
                                                slice_number=Slicesmaps_list[2],
                                                apply_pooling=True,
                                                Basic_Module=DoubleConv,
                                                order=layer_order,
                                                GroupNumber=GroupNormNumber,
                                                slices_hierarchy=slices_hierarchy,
                                                channel_hierarchy=channel_hierarchy,
                                                SE_channel_pooling_type=SE_channel_pooling_type,
                                                SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.EncoderLayer4 = csSEUNet3D_Encoder(in_channels=fmaps_list[2],
                                                out_channels=fmaps_list[3],
                                                slice_number=Slicesmaps_list[3],
                                                apply_pooling=True,
                                                Basic_Module=DoubleConv,
                                                order=layer_order,
                                                GroupNumber=GroupNormNumber,
                                                slices_hierarchy=slices_hierarchy,
                                                channel_hierarchy=channel_hierarchy,
                                                SE_channel_pooling_type=SE_channel_pooling_type,
                                                SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])
        
        DecoderFmapList = list(reversed(fmaps_list))
        DecoderSmapList = list(reversed(Slicesmaps_list))
        
        self.DecoderLayer1 = csSEUNet3D_Decoder(in_channels=DecoderFmapList[0]+DecoderFmapList[1],
                                                out_channels=DecoderFmapList[1],
                                                slice_number=DecoderSmapList[1],
                                                Basic_Module=DoubleConv,
                                                order=layer_order,
                                                GroupNumber=GroupNormNumber,
                                                slices_hierarchy=slices_hierarchy,
                                                channel_hierarchy=channel_hierarchy,
                                                SE_channel_pooling_type = SE_channel_pooling_type,
                                                SE_slice_pooling_type = SE_slice_pooling_type).to(self.device[0])

        self.DecoderLayer2 = csSEUNet3D_Decoder(in_channels=DecoderFmapList[1]+DecoderFmapList[2],
                                                out_channels=DecoderFmapList[2],
                                                slice_number=DecoderSmapList[2],
                                                Basic_Module=DoubleConv,
                                                order=layer_order,
                                                GroupNumber=GroupNormNumber,
                                                slices_hierarchy=slices_hierarchy,
                                                channel_hierarchy=channel_hierarchy,
                                                SE_channel_pooling_type=SE_channel_pooling_type,
                                                SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.DecoderLayer3 = csSEUNet3D_Decoder(in_channels=DecoderFmapList[2]+DecoderFmapList[3],
                                                out_channels=DecoderFmapList[3],
                                                slice_number=DecoderSmapList[3],
                                                Basic_Module=DoubleConv,
                                                order=layer_order,
                                                GroupNumber=GroupNormNumber,
                                                slices_hierarchy=slices_hierarchy,
                                                channel_hierarchy=channel_hierarchy,
                                                SE_channel_pooling_type = SE_channel_pooling_type,
                                                SE_slice_pooling_type = SE_slice_pooling_type).to(self.device[-1])
        
        self.final_conv = nn.Conv3d(in_channels=fmaps_list[0], out_channels=out_channels, kernel_size=1).to(self.device[-1])
        
        if finalsigmoid:
            self.final_activation = nn.Sigmoid().to(self.device[-1])
        else:
            self.final_activation = nn.Softmax(dim=1).to(self.device[-1])


    def forward(self, x):
        encoder_features = []
        x1 = self.EncoderLayer1(x)
        encoder_features.insert(0,x1.to(self.device[-1]))
        x2 = self.EncoderLayer2(x1)
        encoder_features.insert(0,x2)
        x3 = self.EncoderLayer3(x2)
        encoder_features.insert(0,x3)
        x4 = self.EncoderLayer4(x3)


        x  = self.DecoderLayer1(encoder_features[0],x4)
        x  = self.DecoderLayer2(encoder_features[1],x).to(self.device[-1])
        x  = self.DecoderLayer3(encoder_features[2],x)

        x  = self.final_conv(x)

        if not self.training:
            x = self.final_activation(x)
        return x

"""****************************************csSEUnet3D*******************************************"""


"""*****************************************cSEUnet3D*******************************************"""
##############################################################################
# cSEUNet3D
##############################################################################

#cSEUNet3D cSE-module
class cSEUNet3D_cSEmodule(nn.Module):
    """
        cSEUNet3D_cSEmodule
    """
    """modify zmh 0116
       add the SE_channel_pooling_type"""
    def __init__(self,channel,channel_hierarchy,SE_channel_pooling_type):
        super(cSEUNet3D_cSEmodule,self).__init__()
        self.SE_channel_pooling_type = SE_channel_pooling_type
        #reduction = int(channel/channel_hierarchy)
        if(channel_hierarchy>channel):
            reduction = channel
        else:
            reduction = channel_hierarchy
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel//reduction,bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel,bias=True)
        )
        #final no-linear action function
        self.final_nlaf = nn.Sigmoid()

    def forward(self,x):
        if(self.SE_channel_pooling_type == 'avg'):
            b,c,d,h,w = x.size()
            y = self.avg_pool(x).view(b,c)
            y = (self.final_nlaf(self.fc(y))).view(b,c,1,1,1)
            return x*y.expand_as(x)
        elif(self.SE_channel_pooling_type == 'max'):
            b,c,d,h,w = x.size()
            y = self.max_pool(x).view(b,c)
            y = (self.final_nlaf(self.fc(y))).view(b, c, 1, 1, 1)
            return x*y.expand_as(x)
        elif(self.SE_channel_pooling_type == 'avg_and_max'):
            b,c,d,h,w = x.size()
            y_avg = self.avg_pool(x).view(b,c)
            y_max = self.max_pool(x).view(b,c)
            y_avg_and_max = self.fc(y_avg) + self.fc(y_max)
            y = self.final_nlaf(y_avg_and_max).view(b,c,1,1,1)
            return x*y.expand_as(x)
        else:
            raise ValueError('SE_channel_pooling_type must be avg,max,avg_and_max')
            return x

#Encoder
class cSEUNet3D_Encoder(nn.Module):
    """
    Encode the network
    """
    def __init__(self,in_channels,out_channels,channel_hierarchy,SE_channel_pooling_type,kernel_size=3,pool_kernelsize=(2,2,2),
                 pooling_type='max',apply_pooling=True,Basic_Module = DoubleConv, order ='crg',GroupNumber=8):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param pool_kernelsize:
        :param pooling_type:
        :param apply_pooling:
        :param Basic_Module:
        :param order:
        :param GroupNumber:
        """
        super(cSEUNet3D_Encoder,self).__init__()
        assert pooling_type in ['max','avg'],'Pooling_Type must be max or avg'
        if apply_pooling:
            if pooling_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernelsize)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernelsize)
        else:
            self.pooling = None

        self.basic_module = Basic_Module(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         encoder=True,order=order,
                                         GroupNumber=GroupNumber)
        """
        @Anno by zmh
        append encoder_cSE by each encoder layer
        """
        self.encoder_cSE  = cSEUNet3D_cSEmodule(channel=out_channels,
                                                channel_hierarchy=channel_hierarchy,
                                                SE_channel_pooling_type=SE_channel_pooling_type)

    def forward(self,x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        x = self.encoder_cSE(x)
        return x

#Decoder
class cSEUNet3D_Decoder(nn.Module):
    """
    Decode the network
    """
    def __init__(self,in_channels,out_channels,channel_hierarchy,SE_channel_pooling_type,kernel_size=3,pool_kernelsize=(2,2,2),
                 Basic_Module = DoubleConv, order ='crb',GroupNumber=8):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param pool_kernelsize:
        :param Basic_Module:
        :param order:
        :param GroupNumber:
        """
        super(cSEUNet3D_Decoder,self).__init__()
        if Basic_Module == DoubleConv:
            #interpolate not transpose3d
            self.upsample = None
        self.basic_module = Basic_Module(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         encoder=False,order=order,
                                         GroupNumber=GroupNumber)
        """
         @Anno by zmh
         append decoder_cSE by each encoder layer
         """
        self.decoder_cSE = cSEUNet3D_cSEmodule(channel=out_channels,
                                               channel_hierarchy=channel_hierarchy,
                                               SE_channel_pooling_type=SE_channel_pooling_type)

    def forward(self,encoder_feature,x):
        if self.upsample is None:
            #encoder's feature's ---> D*H*W
            output_size = encoder_feature.size()[2:]
            x = F.interpolate(input=x,size=output_size,mode='nearest')
            x = torch.cat((encoder_feature,x),dim=1)
        x = self.basic_module(x)
        x = self.decoder_cSE(x)
        return x

#cSEUNet3D
class cSEUNet3D(nn.Module):
    """
    @Compiled by zmh from scratch
    cSE_UNet3D
    """
    def __init__(self,in_channels,out_channels,finalsigmoid,fmaps_degree,GroupNormNumber,
                 fmaps_layer_number,layer_order,channel_hierarchy,SE_channel_pooling_type,device,**kwargs):
        super(cSEUNet3D,self).__init__()
        self.device = device
        assert isinstance(fmaps_degree,int) , 'fmaps_degree must be an integer!'
        fmaps_list = create_feature_maps(fmaps_degree,fmaps_layer_number)
        
        self.EncoderLayer1 = cSEUNet3D_Encoder(in_channels=in_channels,
                                               out_channels=fmaps_list[0],
                                               apply_pooling=False,
                                               Basic_Module=DoubleConv,
                                               order=layer_order,
                                               GroupNumber=GroupNormNumber,
                                               channel_hierarchy=channel_hierarchy,
                                               SE_channel_pooling_type=SE_channel_pooling_type).to(self.device[0])

        self.EncoderLayer2 = cSEUNet3D_Encoder(in_channels=fmaps_list[0],
                                               out_channels=fmaps_list[1],
                                               apply_pooling=True,
                                               Basic_Module=DoubleConv,
                                               order=layer_order,
                                               GroupNumber=GroupNormNumber,
                                               channel_hierarchy=channel_hierarchy,
                                               SE_channel_pooling_type=SE_channel_pooling_type).to(self.device[0])

        self.EncoderLayer3 = cSEUNet3D_Encoder(in_channels=fmaps_list[1],
                                               out_channels=fmaps_list[2],
                                               apply_pooling=True,
                                               Basic_Module=DoubleConv,
                                               order=layer_order,
                                               GroupNumber=GroupNormNumber,
                                               channel_hierarchy=channel_hierarchy,
                                               SE_channel_pooling_type=SE_channel_pooling_type).to(self.device[0])

        self.EncoderLayer4 = cSEUNet3D_Encoder(in_channels=fmaps_list[2],
                                               out_channels=fmaps_list[3],
                                               apply_pooling=True,
                                               Basic_Module=DoubleConv,
                                               order=layer_order,
                                               GroupNumber=GroupNormNumber,
                                               channel_hierarchy=channel_hierarchy,
                                               SE_channel_pooling_type=SE_channel_pooling_type).to(self.device[0])
        
        DecoderFmapList = list(reversed(fmaps_list))
        
        self.DecoderLayer1 = cSEUNet3D_Decoder(in_channels=DecoderFmapList[0]+DecoderFmapList[1],
                                               out_channels=DecoderFmapList[1],
                                               Basic_Module=DoubleConv,
                                               order=layer_order,
                                               GroupNumber=GroupNormNumber,
                                               channel_hierarchy=channel_hierarchy,
                                               SE_channel_pooling_type=SE_channel_pooling_type).to(self.device[0])

        self.DecoderLayer2 = cSEUNet3D_Decoder(in_channels=DecoderFmapList[1]+DecoderFmapList[2],
                                               out_channels=DecoderFmapList[2],
                                               Basic_Module=DoubleConv,
                                               order=layer_order,
                                               GroupNumber=GroupNormNumber,
                                               channel_hierarchy=channel_hierarchy,
                                               SE_channel_pooling_type=SE_channel_pooling_type).to(self.device[0])

        self.DecoderLayer3 = cSEUNet3D_Decoder(in_channels=DecoderFmapList[2]+DecoderFmapList[3],
                                               out_channels=DecoderFmapList[3],
                                               Basic_Module=DoubleConv,
                                               order=layer_order,
                                               GroupNumber=GroupNormNumber,
                                               channel_hierarchy=channel_hierarchy,
                                               SE_channel_pooling_type=SE_channel_pooling_type).to(self.device[-1])

        self.final_conv = nn.Conv3d(in_channels=fmaps_list[0], out_channels=out_channels, kernel_size=1).to(self.device[-1])
        
        if finalsigmoid:
            self.final_activation = nn.Sigmoid().to(self.device[-1])
        else:
            self.final_activation = nn.Softmax(dim=1).to(self.device[-1])


    def forward(self, x):
        encoder_features = []
        x1 = self.EncoderLayer1(x)
        encoder_features.insert(0,x1.to(self.device[-1]))
        x2 = self.EncoderLayer2(x1)
        encoder_features.insert(0,x2)
        x3 = self.EncoderLayer3(x2)
        encoder_features.insert(0,x3)
        x4 = self.EncoderLayer4(x3)


        x  = self.DecoderLayer1(encoder_features[0],x4)
        x  = self.DecoderLayer2(encoder_features[1],x).to(self.device[-1])
        x  = self.DecoderLayer3(encoder_features[2],x)

        x  = self.final_conv(x)

        if not self.training:
            x = self.final_activation(x)
        return x

"""*****************************************cSEUnet3D*******************************************"""


##############################################################################
# agscSE_Unet3D advance gradually
##############################################################################
"""*****************************agscSE_Unet3D advance gradually*********************************"""
#ag_cSE-module  ag:advance gradually
class ag_cSEmodule(nn.Module):
    def __init__(self,channel,channel_hierarchy,SE_channel_pooling_type):
        super(ag_cSEmodule,self).__init__()
        self.SE_channel_pooling_type = SE_channel_pooling_type
        #reduction = int(channel/channel_hierarchy)
        if(channel_hierarchy>channel):
            reduction = channel
        else:
            reduction = channel_hierarchy
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel//reduction,bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel,bias=True),
        )
        self.final_nlaf = nn.Sigmoid()

    def forward(self,x,weight_vectors):
        if(self.SE_channel_pooling_type == 'avg'):
            b,c,d,h,w = x.size()
            with torch.no_grad():
                temp_cube_tensor_lists=[]
                for i in range(c):
                    weight_vector = weight_vectors[i]
                    temp_cube = []
                    for j in range(d):
                        if(weight_vector[j]==True):
                            temp_cube.append((x[:,i,j,:,:]).data.cpu().numpy())
                    temp_cube_nparray= np.asarray(temp_cube)
                    temp_cube_tensor = torch.from_numpy(temp_cube_nparray).float()
                    temp_cube_tensor = temp_cube_tensor.permute(1,0,2,3)
                    temp_cube_tensor_lists.append(temp_cube_tensor)
            y = self.avg_pool(x).view(b,c)
            y_clone =y.clone()
            for i in range(len(temp_cube_tensor_lists)):
                tempy = self.avg_pool(temp_cube_tensor_lists[i])
                y_clone[:,i] = tempy[0,0,0,0]
            y = y_clone
            y = (self.final_nlaf(self.fc(y))).view(b,c,1,1,1)
            return x*y.expand_as(x)

        elif(self.SE_channel_pooling_type == 'max'):
            b,c,d,h,w = x.size()
            with torch.no_grad():
                temp_cube_tensor_lists=[]
                for i in range(c):
                    weight_vector = weight_vectors[i]
                    temp_cube = []
                    for j in range(d):
                        if(weight_vector[j]==True):
                            temp_cube.append((x[:,i,j,:,:]).data.cpu().numpy())
                    temp_cube_nparray= np.asarray(temp_cube)
                    temp_cube_tensor = torch.from_numpy(temp_cube_nparray).float()
                    temp_cube_tensor = temp_cube_tensor.permute(1,0,2,3)
                    temp_cube_tensor_lists.append(temp_cube_tensor)
            y = self.max_pool(x).view(b,c)
            y_clone =y.clone()
            for i in range(len(temp_cube_tensor_lists)):
                tempy = self.max_pool(temp_cube_tensor_lists[i])
                y_clone[:,i] = tempy[0,0,0,0]
            y = y_clone
            y = (self.final_nlaf(self.fc(y))).view(b,c,1,1,1)
            return x*y.expand_as(x)

        elif(self.SE_channel_pooling_type == 'avg_and_max'):
            b,c,d,h,w = x.size()
            with torch.no_grad():
                temp_cube_tensor_lists=[]
                for i in range(c):
                    weight_vector = weight_vectors[i]
                    temp_cube = []
                    for j in range(d):
                        if(weight_vector[j]==True):
                            temp_cube.append((x[:,i,j,:,:]).data.cpu().numpy())
                    temp_cube_nparray= np.asarray(temp_cube)
                    temp_cube_tensor = torch.from_numpy(temp_cube_nparray).float()
                    temp_cube_tensor = temp_cube_tensor.permute(1,0,2,3)
                    temp_cube_tensor_lists.append(temp_cube_tensor)
            y_avg = self.avg_pool(x).view(b,c)
            y_max = self.max_pool(x).view(b,c)
            y_avg_clone = y_avg.clone()
            y_max_clone = y_max.clone()

            for i in range(len(temp_cube_tensor_lists)):
                tempmax = self.max_pool(temp_cube_tensor_lists[i])
                y_max_clone[:,i] = tempmax[0,0,0,0]
                tempavg = self.avg_pool(temp_cube_tensor_lists[i])
                y_avg_clone[:,i] = tempavg[0,0,0,0]
            y_avg_and_max = self.fc(y_max_clone)+self.fc(y_avg_clone)
            y = (self.final_nlaf(y_avg_and_max)).view(b,c,1,1,1)
            return x*y.expand_as(x)

#ag_SlicesSE-module
class ag_Eachchannel_slicesSEmodule(nn.Module):
    def __init__(self,in_channels,reduction,SE_slice_pooling_type):
        super(ag_Eachchannel_slicesSEmodule,self).__init__()
        self.SE_slice_pooling_type = SE_slice_pooling_type
        #tailor for each H*W channel
        self.avg_pool_hw = nn.AdaptiveAvgPool2d(1)
        self.max_pool_hw = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels,in_channels//reduction,bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//reduction,in_channels,bias=True),
        )
        self.final_nlaf = nn.Sigmoid()

    def forward(self,x):
        if (self.SE_slice_pooling_type == 'avg'):
            b,d,h,w = x.size()
            y = self.avg_pool_hw(x).view(b,d)
            y = (self.final_nlaf(self.fc(y))).view(b,d,1,1)
            with torch.no_grad():
                temp = y.view(d)
                #if(d==144):
                    #with open('docs/heatmap/slice_heatmap_malc.txt','a+') as f:
                        #f.write(str(temp)+"\n")
                        #f.close()
                weight_vector = nn.functional.softmax(input=temp,dim=0)
                mean_weight = torch.mean(weight_vector,dim=0)
                weight_vector = weight_vector >= mean_weight
                # print(mean_weight)
                # print(type(weight_vector))
                # print(weight_vector)
            x = x*y.expand_as(x)
            return x,weight_vector

        elif(self.SE_slice_pooling_type == 'max'):
            b,d,h,w = x.size()
            y = self.max_pool_hw(x).view(b,d)
            y = (self.final_nlaf(self.fc(y))).view(b,d,1,1)
            with torch.no_grad():
                temp = y.view(d)
                weight_vector = nn.functional.softmax(input=temp,dim=0)
                mean_weight = torch.mean(weight_vector,dim=0)
                weight_vector = weight_vector >= mean_weight
            x = x*y.expand_as(x)
            return x,weight_vector

        elif (self.SE_slice_pooling_type == 'avg_and_max'):
            b,d,h,w = x.size()
            y_max = self.max_pool_hw(x).view(b,d)
            y_avg = self.avg_pool_hw(x).view(b,d)
            y_avg_and_max = self.fc(y_avg) + self.fc(y_max)
            y = self.final_nlaf(y_avg_and_max).view(b,d,1,1)
            with torch.no_grad():
                temp = y.view(d)
                weight_vector = nn.functional.softmax(input=temp,dim=0)
                mean_weight = torch.mean(weight_vector,dim=0)
                weight_vector = weight_vector >= mean_weight
            x = x*y.expand_as(x)
            return x,weight_vector
        else:
            raise ValueError('SE_slice_pooling_type must be avg,max,avg_and_max')

class ag_Wholechannels_slicesSEmodule(nn.Module):
    def __init__(self,in_channels,slices_number,slices_hierarchy,SE_slice_pooling_type):
        super(ag_Wholechannels_slicesSEmodule,self).__init__()
        Slice_SElists  = []
        #reduction = int(slices_number/slices_hierarchy)
        reduction = slices_hierarchy
        #print('in_channels:{}'.format(in_channels))
        for i in range(in_channels):
            Slice_SEmodule = ag_Eachchannel_slicesSEmodule(in_channels=slices_number,reduction=reduction,SE_slice_pooling_type=SE_slice_pooling_type)
            Slice_SElists.append(Slice_SEmodule)
        self.Slice_SE_encoder = nn.ModuleList(Slice_SElists)
    def forward(self,x):
        Weight_vectors = []
        for index,Slice_SEmodule in enumerate(self.Slice_SE_encoder):
            y = x[:,index,:,:,:].clone()
            y,weight_vector = Slice_SEmodule(y)
            with torch.no_grad():
                Weight_vectors.append(weight_vector)
            x[:, index, :, :, :]=y
        #这里返回的是一个weight_vectors[Nchannel * Kslices]
        #print(len(Weight_vectors))
        return x,Weight_vectors

#Encoder
class ag_Encoder(nn.Module):
    """
    Encode the network
    """
    def __init__(self,in_channels,out_channels,slice_number,slices_hierarchy,channel_hierarchy,
                 SE_channel_pooling_type,SE_slice_pooling_type,kernel_size=3,pool_kernelsize=(2,2,2),
                 pooling_type='max',apply_pooling=True,Basic_Module = DoubleConv, order ='crg',GroupNumber=8):
        """
        :param in_channels:
        :param out_channels:
        :param slice_number: how many slices need to be SE
        :param kernel_size:
        :param pool_kernelsize:
        :param pooling_type:
        :param apply_pooling:
        :param Basic_Module:
        :param order:
        :param GroupNumber:
        """
        super(ag_Encoder,self).__init__()
        assert pooling_type in ['max','avg'],'Pooling_Type must be max or avg'
        if apply_pooling:
            if pooling_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernelsize)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernelsize)
        else:
            self.pooling = None

        self.basic_module = Basic_Module(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         encoder=True,order=order,
                                         GroupNumber=GroupNumber)


        self.encoder_sSE = ag_Wholechannels_slicesSEmodule(in_channels=out_channels,
                                                           slices_number=slice_number,
                                                           slices_hierarchy=slices_hierarchy,
                                                           SE_slice_pooling_type=SE_slice_pooling_type)


        self.encoder_cSE  = ag_cSEmodule(channel=out_channels,
                                         channel_hierarchy=channel_hierarchy,
                                         SE_channel_pooling_type=SE_channel_pooling_type)

    def forward(self,x):
        if self.pooling is not None:
            x = self.pooling(x)
        
        #x = checkpoint_sequential(self.basic_module,2,x)
        #print('hello')
        #x,weight_vectors= self.encoder_sSE(x)
        #x = checkpoint(self.encoder_cSE,x,weight_vectors)
        
        x = self.basic_module(x)
        x,weight_vectors= self.encoder_sSE(x)
        x = self.encoder_cSE(x,weight_vectors)
        return x

#Decoder
class ag_Decoder(nn.Module):
    """
    Decode the network
    """
    def __init__(self,in_channels,out_channels,use_spatial_attention,slice_number,slices_hierarchy,channel_hierarchy,
                 SE_channel_pooling_type,SE_slice_pooling_type,kernel_size=3,pool_kernelsize=(2,2,2),
                 Basic_Module = DoubleConv, order ='crb',GroupNumber=8):
        """

        :param in_channels:
        :param out_channels:
        :param slice_number:
        :param kernel_size:
        :param pool_kernelsize:
        :param Basic_Module:
        :param order:
        :param GroupNumber:
        """
        super(ag_Decoder,self).__init__()
        self.use_spatial_attention = use_spatial_attention
        if Basic_Module == DoubleConv:
            #interpolate not transpose3d
            self.upsample = None
        self.basic_module = Basic_Module(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         encoder=False,order=order,
                                         GroupNumber=GroupNumber)

        self.decoder_sSE = ag_Wholechannels_slicesSEmodule(in_channels=out_channels,
                                                           slices_number=slice_number,
                                                           slices_hierarchy=slices_hierarchy,
                                                           SE_slice_pooling_type=SE_slice_pooling_type)

        self.decoder_cSE = ag_cSEmodule(channel=out_channels,
                                        channel_hierarchy=channel_hierarchy,
                                        SE_channel_pooling_type=SE_channel_pooling_type)
        #attention
        if(self.use_spatial_attention):
            self.attention_module = _GridAttentionBlock3D(in_channels=out_channels,inter_channels=out_channels//2,gate_channels=out_channels*2)



    def forward(self,encoder_feature,x):
        if self.upsample is None:
            #encoder's feature's ---> D*H*W
            output_size = encoder_feature.size()[2:]
            if(self.use_spatial_attention):
                encoder_feature,att_map = self.attention_module(encoder_feature,x)
            x = F.interpolate(input=x,size=output_size,mode='trilinear')
            x = torch.cat((encoder_feature,x),dim=1)
            
        #x = checkpoint_sequential(self.basic_module,2,x)
        #x,weight_vectors = self.decoder_sSE(x)
        #x = checkpoint(self.decoder_cSE,x,weight_vectors)
        
        x = self.basic_module(x)
        x,weight_vectors = self.decoder_sSE(x)
        x = self.decoder_cSE(x,weight_vectors)
        return x

#agscSEUNet3D
class agscSEUNet3D(nn.Module):
    """
    @Compiled by zmh from scratch
    scSE_UNet3D
        image_shape  decides the dynamic number of slices SEmodule
    """
    def __init__(self,in_channels,out_channels,use_spatial_attention,depth,finalsigmoid,fmaps_degree,
                 GroupNormNumber,fmaps_layer_number,layer_order,slices_hierarchy,channel_hierarchy,
                 SE_channel_pooling_type, SE_slice_pooling_type,device,**kwargs):
        super(agscSEUNet3D,self).__init__()
        self.device = device
        assert isinstance(fmaps_degree,int) , 'fmaps_degree must be an integer!'
        fmaps_list      = create_feature_maps(fmaps_degree,fmaps_layer_number)
        #e.g [128,64,32,16]
        Slicesmaps_list = scSEUNet3D_create_slices_maps(depth,fmaps_layer_number)

        self.EncoderLayer1 = ag_Encoder(in_channels=in_channels,
                                        out_channels=fmaps_list[0],
                                        slice_number=Slicesmaps_list[0],
                                        apply_pooling=False,
                                        Basic_Module=DoubleConv,
                                        order=layer_order,
                                        GroupNumber=GroupNormNumber,
                                        slices_hierarchy=slices_hierarchy,
                                        channel_hierarchy=channel_hierarchy,
                                        SE_channel_pooling_type=SE_channel_pooling_type,
                                        SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.EncoderLayer2 = ag_Encoder(in_channels=fmaps_list[0],
                                        out_channels=fmaps_list[1],
                                        slice_number=Slicesmaps_list[1],
                                        apply_pooling=True,
                                        Basic_Module=DoubleConv,
                                        order=layer_order,
                                        GroupNumber=GroupNormNumber,
                                        slices_hierarchy=slices_hierarchy,
                                        channel_hierarchy=channel_hierarchy,
                                        SE_channel_pooling_type=SE_channel_pooling_type,
                                        SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.EncoderLayer3 = ag_Encoder(in_channels=fmaps_list[1],
                                        out_channels=fmaps_list[2],
                                        slice_number=Slicesmaps_list[2],
                                        apply_pooling=True,
                                        Basic_Module=DoubleConv,
                                        order=layer_order,
                                        GroupNumber=GroupNormNumber,
                                        slices_hierarchy=slices_hierarchy,
                                        channel_hierarchy=channel_hierarchy,
                                        SE_channel_pooling_type=SE_channel_pooling_type,
                                        SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.EncoderLayer4 = ag_Encoder(in_channels=fmaps_list[2],
                                        out_channels=fmaps_list[3],
                                        slice_number=Slicesmaps_list[3],
                                        apply_pooling=True,
                                        Basic_Module=DoubleConv,
                                        order=layer_order,
                                        GroupNumber=GroupNormNumber,
                                        slices_hierarchy=slices_hierarchy,
                                        channel_hierarchy=channel_hierarchy,
                                        SE_channel_pooling_type=SE_channel_pooling_type,
                                        SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        DecoderFmapList = list(reversed(fmaps_list))
        DecoderSmapList = list(reversed(Slicesmaps_list))

        self.DecoderLayer1 = ag_Decoder(in_channels=DecoderFmapList[0]+DecoderFmapList[1],
                                        out_channels=DecoderFmapList[1],
                                        slice_number=DecoderSmapList[1],
                                        Basic_Module=DoubleConv,
                                        order=layer_order,
                                        use_spatial_attention=use_spatial_attention,
                                        GroupNumber=GroupNormNumber,
                                        slices_hierarchy=slices_hierarchy,
                                        channel_hierarchy=channel_hierarchy,
                                        SE_channel_pooling_type=SE_channel_pooling_type,
                                        SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.DecoderLayer2 = ag_Decoder(in_channels=DecoderFmapList[1]+DecoderFmapList[2],
                                        out_channels=DecoderFmapList[2],
                                        slice_number=DecoderSmapList[2],
                                        Basic_Module=DoubleConv,
                                        order=layer_order,
                                        use_spatial_attention=use_spatial_attention,
                                        GroupNumber=GroupNormNumber,
                                        slices_hierarchy=slices_hierarchy,
                                        channel_hierarchy=channel_hierarchy,
                                        SE_channel_pooling_type=SE_channel_pooling_type,
                                        SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.DecoderLayer3 = ag_Decoder(in_channels=DecoderFmapList[2]+DecoderFmapList[3],
                                        out_channels=DecoderFmapList[3],
                                        slice_number=DecoderSmapList[3],
                                        Basic_Module=DoubleConv,
                                        order=layer_order,
                                        use_spatial_attention=use_spatial_attention,
                                        GroupNumber=GroupNormNumber,
                                        slices_hierarchy=slices_hierarchy,
                                        channel_hierarchy=channel_hierarchy,
                                        SE_channel_pooling_type=SE_channel_pooling_type,
                                        SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[-1])

        self.final_conv = nn.Conv3d(in_channels=fmaps_list[0],out_channels=out_channels,kernel_size=1).to(self.device[-1])
        if finalsigmoid:
            self.final_activation = nn.Sigmoid().to(self.device[-1])
        else:
            self.final_activation = nn.Softmax(dim=1).to(self.device[-1])

    def forward(self, x):
        encoder_features = []
        #0219 modify
        #x.requires_grad_()
        #x1 = checkpoint(self.EncoderLayer1,x)
        #x2 = checkpoint(self.EncoderLayer2,x1)
        #x3 = checkpoint(self.EncoderLayer3,x2)
        #x4 = checkpoint(self.EncoderLayer4,x3)
        
        #x  = checkpoint(self.DecoderLayer1,x3,x4)
        #x  = checkpoint(self.DecoderLayer2,x2,x).to(self.device[1])
        #x  = checkpoint(self.DecoderLayer3,x1.to(self.device[1]),x)
        #x  = checkpoint(self.final_conv,x)
        
        
        
        #x1 = self.EncoderLayer1(x)
        #x2 = self.EncoderLayer2(x1)
        #x3 = self.EncoderLayer3(x2)
        #x4 = self.EncoderLayer4(x3)
            
        
        #x  = self.DecoderLayer1(x3,x4)
        #x  = self.DecoderLayer2(x2,x).to(self.device[1])
        #x  = self.DecoderLayer3(x1.to(self.device[1]),x)
        #x  = self.final_conv(x)
       
       
        x1 = self.EncoderLayer1(x)
        encoder_features.insert(0,x1.to(self.device[-1]))
        x2 = self.EncoderLayer2(x1)
        encoder_features.insert(0,x2)
        x3 = self.EncoderLayer3(x2)
        encoder_features.insert(0,x3)
        x4 = self.EncoderLayer4(x3)


        x  = self.DecoderLayer1(encoder_features[0],x4)
        x  = self.DecoderLayer2(encoder_features[1],x).to(self.device[-1])
        x  = self.DecoderLayer3(encoder_features[2],x)

        x  = self.final_conv(x)

        if not self.training:
            x = self.final_activation(x)
        return x
"""*****************************agscSE_Unet3D advance gradually*********************************"""



"""********************************************VNet*********************************************"""
#VNet residual conv block
class VNetResidualConvBlock(nn.Module):
    def __init__(self,repetition,in_channels,out_channels,order='cri',kernel_size=5,padding=2,GroupNormNumber=8):
        super(VNetResidualConvBlock, self).__init__()
        modules = []

        for i in range(0,repetition):
            if (i==0):
                conv_in_channels = in_channels
            else:
                conv_in_channels = out_channels

            for j,char in enumerate(order):
                if char == 'c':
                    bias = not ('g' in order or 'b' in order)
                    modules.append(nn.Conv3d(in_channels=conv_in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding,bias=bias))
                elif (char == 'r' and i != repetition-1):
                    modules.append(nn.ReLU(inplace=True))
                elif (char == 'l' and i != repetition-1):
                    modules.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
                elif (char == 'e' and i != repetition-1):
                    modules.append(nn.ELU(inplace=True))
                elif (char == 'p' and i != repetition-1):
                    modules.append(nn.PReLU(num_parameters=out_channels))
                elif (char == 's' and i != repetition-1):
                    modules.append(swish())
                elif (char == 'm' and i != repetition-1):
                    modules.append(Mish())
                elif char == 'g':
                    is_before_conv = j < order.index('c')
                    assert not is_before_conv, 'GroupNorm MUST go after the Conv3d'
                    # number of groups must be less or equal the number of channels
                    if out_channels < GroupNormNumber:
                        GroupNormNumber = out_channels
                    modules.append( nn.GroupNorm(num_groups=GroupNormNumber, num_channels=out_channels))
                elif char == 'i':
                    is_before_conv = j < order.index('c')
                    if is_before_conv:
                        #affine true ---> with learnable parameters
                        modules.append(nn.InstanceNorm3d(in_channels,affine=True))
                    else:
                        modules.append(nn.InstanceNorm3d(out_channels,affine=True))
                elif char == 'b':
                    is_before_conv = j < order.index('c')
                    if is_before_conv:
                        modules.append(nn.BatchNorm3d(in_channels))
                    else:
                        modules.append( nn.BatchNorm3d(out_channels))
                else:
                    continue

        self.convblock = nn.Sequential(*modules)
        for char in order:
            if(char == 'r'):
                self.final_nolinear_activation = nn.ReLU(inplace=True)
            elif(char == 'l'):
                self.final_nolinear_activation = nn.LeakyReLU(negative_slope=0.1,inplace=True)
            elif(char == 'e'):
                self.final_nolinear_activation = nn.ELU(inplace=True)
            elif(char == 'p'):
                self.final_nolinear_activation = nn.PReLU(num_parameters=out_channels)
            elif (char =='s'):
                self.final_nolinear_activation = swish()
            elif (char == 'm'):
                self.final_nolinear_activation = Mish()
            else:
                continue

    def forward(self, x,identity):
        x = (self.convblock(x)+identity)
        x = self.final_nolinear_activation(x)
        return x

#VNet dowmsample[different from UNET3d]
class VNetDownSampleBlock(nn.Sequential):
    def __init__(self,in_channels,out_channels,order,kernel_size=2,stride=2,GroupNormNumber=8,padding=0):
        super(VNetDownSampleBlock, self).__init__()
        for name,module in create_conv(in_channels,out_channels,kernel_size,order,GroupNormNumber,padding,stride):
            self.add_module(name,module)

#VNet Upsample[different from UNET3d]
class VNetUpSampleBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, order, kernel_size=2, stride=2, GroupNormNumber=8, padding=0,Deconv = True):
        super(VNetUpSampleBlock, self).__init__()
        for name,module in create_conv(in_channels, out_channels, kernel_size, order, GroupNormNumber, padding,stride,Deconv=Deconv):
            self.add_module(name,module)

#VNet Bottleneck
class VNetBottleneckBlock(nn.Sequential):
    def __init__(self,repetition,in_channels, order, kernel_size=5,GroupNormNumber=8, padding=2):
        super(VNetBottleneckBlock, self).__init__()
        for name,module in create_conv(in_channels,in_channels,kernel_size,order,GroupNormNumber,padding):
            self.add_module(name,module)

#VNet Encoder
class VNet_Encoder(nn.Module):
    def __init__(self,repetition,in_channels,out_channels,order,GroupNormNumber):
        super(VNet_Encoder, self).__init__()
        self.conv_layer  = VNetResidualConvBlock(repetition,in_channels,out_channels//2,order)
        self.downsample_layer = VNetDownSampleBlock(out_channels//2,out_channels,order)

    def forward(self,x):
        feature = self.conv_layer(x,x)
        x =self.downsample_layer(feature)
        return x,feature

# VNet Decoder
class VNet_Decoder(nn.Module):
    def __init__(self, repetition, in_channels, out_channels, order, GroupNormNumber):
        super(VNet_Decoder, self).__init__()
        self.conv_layer = VNetResidualConvBlock(repetition, in_channels, out_channels, order)
        self.upsample_layer = VNetUpSampleBlock(in_channels, out_channels, order)

    def forward(self, x,feature):
        x = self.upsample_layer(x)
        x = self.conv_layer(torch.cat((feature,x),dim=1),x)
        return x


#VNet
class VNet(nn.Module):
    def __init__(self,in_channels,out_channels,finalsigmoid,fmaps_degree,GroupNormNumber,fmaps_layer_number,layer_order,device):
        super(VNet, self).__init__()
        self.device = device

        
        fmaps = create_feature_maps(fmaps_degree,fmaps_layer_number)

        self.Encoder_layer1 = VNet_Encoder(repetition=1,in_channels=in_channels,out_channels=fmaps[0],order=layer_order,GroupNormNumber=GroupNormNumber).to(self.device[0])
        self.Encoder_layer2 = VNet_Encoder(repetition=2,in_channels=fmaps[0],out_channels=fmaps[1],order=layer_order,GroupNormNumber=GroupNormNumber).to(self.device[0])
        self.Encoder_layer3 = VNet_Encoder(repetition=3,in_channels=fmaps[1],out_channels=fmaps[2],order=layer_order,GroupNormNumber=GroupNormNumber).to(self.device[0])
        self.Encoder_layer4 = VNet_Encoder(repetition=3,in_channels=fmaps[2],out_channels=fmaps[3],order=layer_order,GroupNormNumber=GroupNormNumber).to(self.device[0])

        self.Bottleneck_layer = VNetBottleneckBlock(repetition=3,in_channels=fmaps[3],order=layer_order,GroupNormNumber=GroupNormNumber).to(self.device[0])

        self.Decoder_layer1 = VNet_Decoder(repetition=3,in_channels=fmaps[3],out_channels=fmaps[2],order=layer_order,GroupNormNumber=GroupNormNumber).to(self.device[0])
        self.Decoder_layer2 = VNet_Decoder(repetition=3,in_channels=fmaps[2],out_channels=fmaps[1],order=layer_order,GroupNormNumber=GroupNormNumber).to(self.device[0])
        self.Decoder_layer3 = VNet_Decoder(repetition=2,in_channels=fmaps[1],out_channels=fmaps[0],order=layer_order,GroupNormNumber=GroupNormNumber).to(self.device[0])
        self.Decoder_layer4 = VNet_Decoder(repetition=1,in_channels=fmaps[0],out_channels=fmaps[0]//2,order=layer_order,GroupNormNumber=GroupNormNumber).to(self.device[-1])

        self.final_conv = nn.Conv3d(fmaps[0]//2,out_channels,1,padding=0).to(self.device[-1])

        if finalsigmoid:
            self.final_activation = nn.Sigmoid().to(self.device[-1])
        else:
            self.final_activation = nn.Softmax(dim=1).to(self.device[-1])


    def forward(self, x):
        x,feature_x1 = self.Encoder_layer1(x)
        x,feature_x2 = self.Encoder_layer2(x)
        x,feature_x3 = self.Encoder_layer3(x)
        x,feature_x4 = self.Encoder_layer4(x)
        x = self.Bottleneck_layer(x)

        x = self.Decoder_layer1(x,feature_x4)
        x = self.Decoder_layer2(x,feature_x3)
        x = self.Decoder_layer3(x,feature_x2).to(self.device[-1])
        x = self.Decoder_layer4(x,feature_x1.to(self.device[-1]))

        x = self.final_conv(x)
        if not  self.training:
            x = self.final_activation(x)
        return x

"""********************************************VNet*********************************************"""


"""*****************************************cSEVNet*********************************************"""
#cSEVNet Encoder
class cSEVNet_Encoder(nn.Module):
    def __init__(self,repetition,in_channels,out_channels,order,GroupNormNumber,channel_hierarchy,SE_channel_pooling_type):
        super(cSEVNet_Encoder, self).__init__()
        self.conv_layer  = VNetResidualConvBlock(repetition,in_channels,out_channels//2,order)
        self.downsample_layer = VNetDownSampleBlock(out_channels//2,out_channels,order)
        self.cSE = cSEUNet3D_cSEmodule(channel=out_channels//2,channel_hierarchy=channel_hierarchy,SE_channel_pooling_type=SE_channel_pooling_type)

    def forward(self,x):
        feature = self.conv_layer(x,x)
        feature = self.cSE(feature)
        x = self.downsample_layer(feature)
        return x,feature

#cSEVNet Decoder
class cSEVNet_Decoder(nn.Module):
    def __init__(self, repetition, in_channels, out_channels, order, GroupNormNumber,channel_hierarchy,SE_channel_pooling_type):
        super(cSEVNet_Decoder, self).__init__()
        self.conv_layer = VNetResidualConvBlock(repetition, in_channels, out_channels, order)
        self.upsample_layer = VNetUpSampleBlock(in_channels, out_channels, order)
        self.cSE = cSEUNet3D_cSEmodule(channel=out_channels, channel_hierarchy=channel_hierarchy,
                                       SE_channel_pooling_type=SE_channel_pooling_type)

    def forward(self, x,feature):
        x = self.upsample_layer(x)
        #x = self.conv_layer(x+feature)
        x = self.conv_layer(torch.cat((feature,x),dim=1),x)
        x = self.cSE(x)
        return x

#cSE VNet
class cSEVNet(nn.Module):
    def __init__(self,in_channels,out_channels,layer_order,GroupNormNumber,finalsigmoid,fmaps_degree,
                fmaps_layer_number,channel_hierarchy,SE_channel_pooling_type,device):
        super(cSEVNet, self).__init__()
        fmaps = create_feature_maps(fmaps_degree,fmaps_layer_number)
        self.device = device
        self.Encoder_layer1 = cSEVNet_Encoder(repetition=1,
                                           in_channels=in_channels,
                                           out_channels=fmaps[0],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type).to(self.device[0])
        self.Encoder_layer2 = cSEVNet_Encoder(repetition=2,
                                           in_channels=fmaps[0],
                                           out_channels=fmaps[1],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type).to(self.device[0])
        self.Encoder_layer3 = cSEVNet_Encoder(repetition=3,
                                           in_channels=fmaps[1],
                                           out_channels=fmaps[2],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type).to(self.device[0])

        self.Encoder_layer4 = cSEVNet_Encoder(repetition=3,
                                           in_channels=fmaps[2],
                                           out_channels=fmaps[3],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type).to(self.device[0])

        self.Bottleneck_layer = VNetBottleneckBlock(repetition=3,in_channels=fmaps[3],order=layer_order,GroupNormNumber=GroupNormNumber).to(self.device[0])
        self.Bottleneck_cSE   = cSEUNet3D_cSEmodule(channel=fmaps[3],channel_hierarchy=channel_hierarchy,SE_channel_pooling_type=SE_channel_pooling_type).to(self.device[0])

        self.Decoder_layer1 = cSEVNet_Decoder(repetition=3,
                                           in_channels=fmaps[3],
                                           out_channels=fmaps[2],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type).to(self.device[0])
        self.Decoder_layer2 = cSEVNet_Decoder(repetition=3,
                                           in_channels=fmaps[2],
                                           out_channels=fmaps[1],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type).to(self.device[0])
        self.Decoder_layer3 = cSEVNet_Decoder(repetition=2,
                                           in_channels=fmaps[1],
                                           out_channels=fmaps[0],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type).to(self.device[0])
        self.Decoder_layer4 = cSEVNet_Decoder(repetition=1,
                                           in_channels=fmaps[0],
                                           out_channels=fmaps[0]//2,
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type).to(self.device[-1])

        self.final_conv = nn.Conv3d(fmaps[0]//2,out_channels,1,padding=0).to(self.device[-1])

        if finalsigmoid:
            self.final_activation = nn.Sigmoid().to(self.device[-1])
        else:
            self.final_activation = nn.Softmax(dim=1).to(self.device[-1])


    def forward(self, x):
        x,feature_x1 = self.Encoder_layer1(x)
        x,feature_x2 = self.Encoder_layer2(x)
        x,feature_x3 = self.Encoder_layer3(x)
        x,feature_x4 = self.Encoder_layer4(x)

        x = self.Bottleneck_layer(x)
        x = self.Bottleneck_cSE(x)

        x = self.Decoder_layer1(x,feature_x4)
        x = self.Decoder_layer2(x,feature_x3)
        x = self.Decoder_layer3(x,feature_x2).to(self.device[-1])
        x = self.Decoder_layer4(x,feature_x1.to(self.device[-1]))

        x = self.final_conv(x)
        if not  self.training:
            x = self.final_activation(x)
        return x
"""*****************************************cSEVNet*********************************************"""

"""*****************************************csSEVNet*********************************************"""
#csSEVNet Encoder
class csSEVNet_Encoder(nn.Module):
    def __init__(self,repetition,in_channels,out_channels,order,GroupNormNumber,channel_hierarchy,slice_number,
                 slice_hierarchy,SE_channel_pooling_type,SE_slice_pooling_type):
        super(csSEVNet_Encoder, self).__init__()
        self.conv_layer  = VNetResidualConvBlock(repetition,in_channels,out_channels//2,order)
        self.downsample_layer = VNetDownSampleBlock(out_channels//2,out_channels,order)
        self.cSE = cSEUNet3D_cSEmodule(channel=out_channels//2,channel_hierarchy=channel_hierarchy,SE_channel_pooling_type=SE_channel_pooling_type)
        self.sSE = scSEUNet3D_Wholechannels_slicesSEmodule(in_channels=out_channels//2,
                                                           slices_number=slice_number,
                                                           slices_hierarchy=slice_hierarchy,
                                                           SE_slice_pooling_type=SE_slice_pooling_type)
    def forward(self,x):
        feature = self.conv_layer(x,x)
        feature = self.cSE(feature)
        feature = self.sSE(feature)
        x = self.downsample_layer(feature)
        return x,feature

#csSEVNet Decoder
class csSEVNet_Decoder(nn.Module):
    def __init__(self, repetition, in_channels, out_channels, order, GroupNormNumber,channel_hierarchy,slice_number,
                 slice_hierarchy,SE_channel_pooling_type,SE_slice_pooling_type):
        super(csSEVNet_Decoder, self).__init__()
        self.conv_layer = VNetResidualConvBlock(repetition, in_channels, out_channels, order)
        self.upsample_layer = VNetUpSampleBlock(in_channels, out_channels, order)
        self.cSE = cSEUNet3D_cSEmodule(channel=out_channels, channel_hierarchy=channel_hierarchy,
                                       SE_channel_pooling_type=SE_channel_pooling_type)
        self.sSE = scSEUNet3D_Wholechannels_slicesSEmodule(in_channels=out_channels,
                                                           slices_number=slice_number,
                                                           slices_hierarchy=slice_hierarchy,
                                                           SE_slice_pooling_type=SE_slice_pooling_type)

    def forward(self, x,feature):
        x = self.upsample_layer(x)
        #x = self.conv_layer(x+feature)
        x = self.conv_layer(torch.cat((feature,x),dim=1),x)
        x = self.cSE(x)
        x = self.sSE(x)
        return x

#csSE VNet
class csSEVNet(nn.Module):
    def __init__(self,in_channels,out_channels,layer_order,GroupNormNumber,finalsigmoid,fmaps_degree,fmaps_layer_number,depth,
                 channel_hierarchy,slice_hierarchy,SE_channel_pooling_type,SE_slice_pooling_type,device,**kwargs):
        super(csSEVNet, self).__init__()
        self.device = device
        fmaps = create_feature_maps(fmaps_degree,fmaps_layer_number)
        slice_maps = scSEUNet3D_create_slices_maps(depth,fmaps_layer_number+1) # +1 for bottleneck layer
        self.Encoder_layer1 = csSEVNet_Encoder(repetition=1,
                                           in_channels=in_channels,
                                           out_channels=fmaps[0],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[0],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.Encoder_layer2 = csSEVNet_Encoder(repetition=2,
                                           in_channels=fmaps[0],
                                           out_channels=fmaps[1],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[1],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.Encoder_layer3 = csSEVNet_Encoder(repetition=3,
                                           in_channels=fmaps[1],
                                           out_channels=fmaps[2],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[2],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.Encoder_layer4 = csSEVNet_Encoder(repetition=3,
                                           in_channels=fmaps[2],
                                           out_channels=fmaps[3],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[3],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.Bottleneck_layer = VNetBottleneckBlock(repetition=3,in_channels=fmaps[3],order=layer_order,GroupNormNumber=GroupNormNumber).to(self.device[0])
        self.Bottleneck_cSE   = cSEUNet3D_cSEmodule(channel=fmaps[3],channel_hierarchy=channel_hierarchy,SE_channel_pooling_type=SE_channel_pooling_type).to(self.device[0])

        self.Decoder_layer1 = csSEVNet_Decoder(repetition=3,
                                           in_channels=fmaps[3],
                                           out_channels=fmaps[2],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[3],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])
                                           
        self.Decoder_layer2 = csSEVNet_Decoder(repetition=3,
                                           in_channels=fmaps[2],
                                           out_channels=fmaps[1],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[2],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.Decoder_layer3 = csSEVNet_Decoder(repetition=2,
                                           in_channels=fmaps[1],
                                           out_channels=fmaps[0],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[1],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.Decoder_layer4 = csSEVNet_Decoder(repetition=1,
                                           in_channels=fmaps[0],
                                           out_channels=fmaps[0]//2,
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[0],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[-1])

        self.final_conv = nn.Conv3d(fmaps[0]//2,out_channels,1,padding=0).to(self.device[-1])

        if finalsigmoid:
            self.final_activation = nn.Sigmoid().to(self.device[-1])
        else:
            self.final_activation = nn.Softmax(dim=1).to(self.device[-1])


    def forward(self, x):

        x,feature_x1 = self.Encoder_layer1(x)

        x,feature_x2 = self.Encoder_layer2(x)

        x,feature_x3 = self.Encoder_layer3(x)

        x,feature_x4 = self.Encoder_layer4(x)

        
        x = self.Bottleneck_layer(x)
        x = self.Bottleneck_cSE(x)

        x = self.Decoder_layer1(x,feature_x4)
        x = self.Decoder_layer2(x,feature_x3)
        x = self.Decoder_layer3(x,feature_x2).to(self.device[-1])
        x = self.Decoder_layer4(x,feature_x1.to(self.device[-1]))

        x = self.final_conv(x)
        if not  self.training:
            x = self.final_activation(x)
        return x
"""*****************************************csSEVNet*********************************************"""


"""*****************************************scSEVNet*********************************************"""
#scSEVNet Encoder
class scSEVNet_Encoder(nn.Module):
    def __init__(self,repetition,in_channels,out_channels,order,GroupNormNumber,channel_hierarchy,slice_number,
                 slice_hierarchy,SE_channel_pooling_type,SE_slice_pooling_type):
        super(scSEVNet_Encoder, self).__init__()
        self.conv_layer  = VNetResidualConvBlock(repetition,in_channels,out_channels//2,order)
        self.downsample_layer = VNetDownSampleBlock(out_channels//2,out_channels,order)

        self.cSE = cSEUNet3D_cSEmodule(channel=out_channels//2,channel_hierarchy=channel_hierarchy,SE_channel_pooling_type=SE_channel_pooling_type)
        self.sSE = scSEUNet3D_Wholechannels_slicesSEmodule(in_channels=out_channels//2,
                                                           slices_number=slice_number,
                                                           slices_hierarchy=slice_hierarchy,
                                                           SE_slice_pooling_type=SE_slice_pooling_type)
    def forward(self,x):
        feature = self.conv_layer(x,x)
        feature = self.sSE(feature)
        feature = self.cSE(feature)
        x = self.downsample_layer(feature)
        return x,feature

#scSEVNet Decoder
class scSEVNet_Decoder(nn.Module):
    def __init__(self, repetition, in_channels, out_channels, order, GroupNormNumber,channel_hierarchy,slice_number,
                 slice_hierarchy,SE_channel_pooling_type,SE_slice_pooling_type):
        super(scSEVNet_Decoder, self).__init__()
        self.conv_layer = VNetResidualConvBlock(repetition, in_channels, out_channels, order)
        self.upsample_layer = VNetUpSampleBlock(in_channels, out_channels, order)
        self.cSE = cSEUNet3D_cSEmodule(channel=out_channels, channel_hierarchy=channel_hierarchy,
                                       SE_channel_pooling_type=SE_channel_pooling_type)
        self.sSE = scSEUNet3D_Wholechannels_slicesSEmodule(in_channels=out_channels,
                                                           slices_number=slice_number,
                                                           slices_hierarchy=slice_hierarchy,
                                                           SE_slice_pooling_type=SE_slice_pooling_type)

    def forward(self, x,feature):
        x = self.upsample_layer(x)
        #x = self.conv_layer(x+feature)
        x = self.conv_layer(torch.cat((feature,x),dim=1),x)
        x = self.sSE(x)
        x = self.cSE(x)
        return x

#scSE VNet
class scSEVNet(nn.Module):
    def __init__(self,in_channels,out_channels,layer_order,GroupNormNumber,finalsigmoid,fmaps_degree,fmaps_layer_number,depth,
                 channel_hierarchy,slice_hierarchy,SE_channel_pooling_type,SE_slice_pooling_type,device,**kwargs):
        super(scSEVNet, self).__init__()
        self.device = device
        fmaps = create_feature_maps(fmaps_degree,fmaps_layer_number)
        slice_maps = scSEUNet3D_create_slices_maps(depth,fmaps_layer_number+1) # +1 for bottleneck layer
        self.Encoder_layer1 = scSEVNet_Encoder(repetition=1,
                                           in_channels=in_channels,
                                           out_channels=fmaps[0],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[0],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.Encoder_layer2 = scSEVNet_Encoder(repetition=2,
                                           in_channels=fmaps[0],
                                           out_channels=fmaps[1],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[1],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.Encoder_layer3 = scSEVNet_Encoder(repetition=3,
                                           in_channels=fmaps[1],
                                           out_channels=fmaps[2],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[2],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.Encoder_layer4 = scSEVNet_Encoder(repetition=3,
                                           in_channels=fmaps[2],
                                           out_channels=fmaps[3],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[3],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.Bottleneck_layer = VNetBottleneckBlock(repetition=3,in_channels=fmaps[3],order=layer_order,GroupNormNumber=GroupNormNumber).to(self.device[0])
        self.Bottleneck_cSE   = cSEUNet3D_cSEmodule(channel=fmaps[3],channel_hierarchy=channel_hierarchy,SE_channel_pooling_type=SE_channel_pooling_type).to(self.device[0])

        self.Decoder_layer1 = scSEVNet_Decoder(repetition=3,
                                           in_channels=fmaps[3],
                                           out_channels=fmaps[2],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[3],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.Decoder_layer2 = scSEVNet_Decoder(repetition=3,
                                           in_channels=fmaps[2],
                                           out_channels=fmaps[1],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[2],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.Decoder_layer3 = scSEVNet_Decoder(repetition=2,
                                           in_channels=fmaps[1],
                                           out_channels=fmaps[0],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[1],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.Decoder_layer4 = scSEVNet_Decoder(repetition=1,
                                           in_channels=fmaps[0],
                                           out_channels=fmaps[0]//2,
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[0],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[-1])

        self.final_conv = nn.Conv3d(fmaps[0]//2,out_channels,1,padding=0).to(self.device[-1])

        if finalsigmoid:
            self.final_activation = nn.Sigmoid().to(self.device[-1])
        else:
            self.final_activation = nn.Softmax(dim=1).to(self.device[-1])


    def forward(self, x):
        x,feature_x1 = self.Encoder_layer1(x)
        x,feature_x2 = self.Encoder_layer2(x)
        x,feature_x3 = self.Encoder_layer3(x)
        x,feature_x4 = self.Encoder_layer4(x)

        x = self.Bottleneck_layer(x)
        x = self.Bottleneck_cSE(x)

        x = self.Decoder_layer1(x,feature_x4)
        x = self.Decoder_layer2(x,feature_x3)
        x = self.Decoder_layer3(x,feature_x2).to(self.device[-1])
        x = self.Decoder_layer4(x,feature_x1.to(self.device[-1]))

        x = self.final_conv(x)
        if not  self.training:
            x = self.final_activation(x)
        return x
"""*****************************************scSEVNet*********************************************"""


"""****************************************agscSEVNet********************************************"""
#agscSEVNet Encoder
class agscSEVNet_Encoder(nn.Module):
    def __init__(self,repetition,in_channels,out_channels,order,GroupNormNumber,channel_hierarchy,slice_number,
                 slice_hierarchy,SE_channel_pooling_type,SE_slice_pooling_type):
        super(agscSEVNet_Encoder, self).__init__()
        self.conv_layer  = VNetResidualConvBlock(repetition,in_channels,out_channels//2,order)
        self.downsample_layer = VNetDownSampleBlock(out_channels//2,out_channels,order)

        self.cSE = ag_cSEmodule(channel=out_channels//2,channel_hierarchy=channel_hierarchy,
                                SE_channel_pooling_type=SE_channel_pooling_type)
        self.sSE = ag_Wholechannels_slicesSEmodule(in_channels=out_channels//2,
                                                   slices_number=slice_number,
                                                   slices_hierarchy=slice_hierarchy,
                                                   SE_slice_pooling_type=SE_slice_pooling_type)
    def forward(self,x):
        feature = self.conv_layer(x,x)
        feature,weight_vectors = self.sSE(feature)
        feature = self.cSE(feature,weight_vectors)
        x = self.downsample_layer(feature)
        return x,feature

#agscSEVNet Decoder
class agscSEVNet_Decoder(nn.Module):
    def __init__(self, repetition, in_channels, out_channels, order, GroupNormNumber,channel_hierarchy,slice_number,
                 slice_hierarchy,SE_channel_pooling_type,SE_slice_pooling_type):
        super(agscSEVNet_Decoder, self).__init__()
        self.conv_layer = VNetResidualConvBlock(repetition, in_channels, out_channels, order)
        self.upsample_layer = VNetUpSampleBlock(in_channels, out_channels, order)
        self.cSE = ag_cSEmodule(channel=out_channels, channel_hierarchy=channel_hierarchy,
                                SE_channel_pooling_type=SE_channel_pooling_type)
        self.sSE = ag_Wholechannels_slicesSEmodule(in_channels=out_channels,
                                                   slices_number=slice_number,
                                                   slices_hierarchy=slice_hierarchy,
                                                   SE_slice_pooling_type=SE_slice_pooling_type)

    def forward(self, x,feature):
        x = self.upsample_layer(x)
        #x = self.conv_layer(x+feature)
        x = self.conv_layer(torch.cat((feature,x),dim=1),x)
        x,weight_vectors = self.sSE(x)
        x = self.cSE(x,weight_vectors)
        return x

#agscSE VNet
class agscSEVNet(nn.Module):
    def __init__(self,in_channels,out_channels,layer_order,GroupNormNumber,finalsigmoid,fmaps_degree,fmaps_layer_number,depth,
                 channel_hierarchy,slice_hierarchy,SE_channel_pooling_type,SE_slice_pooling_type,device,**kwargs):
        super(agscSEVNet, self).__init__()
        self.device = device 
        fmaps = create_feature_maps(fmaps_degree,fmaps_layer_number)
        slice_maps = scSEUNet3D_create_slices_maps(depth,fmaps_layer_number+1) # +1 for bottleneck layer
        self.Encoder_layer1 = agscSEVNet_Encoder(repetition=1,
                                           in_channels=in_channels,
                                           out_channels=fmaps[0],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[0],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.Encoder_layer2 = agscSEVNet_Encoder(repetition=2,
                                           in_channels=fmaps[0],
                                           out_channels=fmaps[1],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[1],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.Encoder_layer3 = agscSEVNet_Encoder(repetition=3,
                                           in_channels=fmaps[1],
                                           out_channels=fmaps[2],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[2],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.Encoder_layer4 = agscSEVNet_Encoder(repetition=3,
                                           in_channels=fmaps[2],
                                           out_channels=fmaps[3],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[3],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.Bottleneck_layer = VNetBottleneckBlock(repetition=3,in_channels=fmaps[3],order=layer_order,GroupNormNumber=GroupNormNumber).to(self.device[0])
        self.Bottleneck_cSE   = cSEUNet3D_cSEmodule(channel=fmaps[3],channel_hierarchy=channel_hierarchy,SE_channel_pooling_type=SE_channel_pooling_type).to(self.device[0])

        self.Decoder_layer1 = agscSEVNet_Decoder(repetition=3,
                                           in_channels=fmaps[3],
                                           out_channels=fmaps[2],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[3],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.Decoder_layer2 = agscSEVNet_Decoder(repetition=3,
                                           in_channels=fmaps[2],
                                           out_channels=fmaps[1],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[2],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.Decoder_layer3 = agscSEVNet_Decoder(repetition=2,
                                           in_channels=fmaps[1],
                                           out_channels=fmaps[0],
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[1],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[0])

        self.Decoder_layer4 = agscSEVNet_Decoder(repetition=1,
                                           in_channels=fmaps[0],
                                           out_channels=fmaps[0]//2,
                                           order=layer_order,
                                           GroupNormNumber=GroupNormNumber,
                                           channel_hierarchy=channel_hierarchy,
                                           slice_number=slice_maps[0],
                                           slice_hierarchy=slice_hierarchy,
                                           SE_channel_pooling_type=SE_channel_pooling_type,
                                           SE_slice_pooling_type=SE_slice_pooling_type).to(self.device[-1])

        self.final_conv = nn.Conv3d(fmaps[0]//2,out_channels,1,padding=0).to(self.device[-1])

        if finalsigmoid:
            self.final_activation = nn.Sigmoid().to(self.device[-1])
        else:
            self.final_activation = nn.Softmax(dim=1).to(self.device[-1])


    def forward(self, x):
        x,feature_x1 = self.Encoder_layer1(x)
        x,feature_x2 = self.Encoder_layer2(x)
        x,feature_x3 = self.Encoder_layer3(x)
        x,feature_x4 = self.Encoder_layer4(x)

        x = self.Bottleneck_layer(x)
        x = self.Bottleneck_cSE(x)

        x = self.Decoder_layer1(x,feature_x4)
        x = self.Decoder_layer2(x,feature_x3)
        x = self.Decoder_layer3(x,feature_x2).to(self.device[-1])
        x = self.Decoder_layer4(x,feature_x1.to(self.device[-1]))

        x = self.final_conv(x)
        if not  self.training:
            x = self.final_activation(x)
        return x

"""****************************************agscSEVNet********************************************"""


"""**************************************AttentionGate*******************************************"""
class _GridAttentionBlock3D(nn.Module):
    "Method from Attention U-Net"
    def __init__(self,in_channels,inter_channels,gate_channels):
        super(_GridAttentionBlock3D, self).__init__()
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        # self.theta processes the input signal: x_l [2C D H W]
        self.theta = nn.Conv3d(in_channels=in_channels,out_channels=inter_channels,kernel_size=2,stride=2,padding=0,bias=False)
        # self.Phi^T processes the gate signal: g [C 2D 2H 2W]
        self.phi = nn.Conv3d(in_channels=gate_channels,out_channels=inter_channels,kernel_size=1,stride=1,padding=0,bias=True)
        # self.Psi processes the inter signal: g [interchannel D H W]
        self.psi = nn.Conv3d(in_channels=inter_channels,out_channels=1,kernel_size=1,stride=1,padding=0,bias=True)

        self.final_conv = nn.Sequential(nn.Conv3d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,padding=0),
                                        nn.InstanceNorm3d(in_channels,affine=True))

        self.feature1 = None

    def forward(self,x,g):
        input_size = x.size()
        theta_x = self.theta(x)
        phi_g = self.phi(g)
        f  = F.relu(theta_x+phi_g,inplace=True)
        sigm_psi_f = torch.sigmoid(self.psi(f))
        sigm_psi_f = F.interpolate(sigm_psi_f,size=input_size[2:],mode='trilinear')
        y = sigm_psi_f.expand_as(x)*x
        y = self.final_conv(y)
        if not self.training:
            self.feature1 = sigm_psi_f.detach()
        return y,sigm_psi_f

"""**************************************AttentionGate*******************************************"""

"""**************************************CBAM Spatial Gate*******************************************"""
class _CBAM_ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(tensors=(torch.max(x,1)[0].unsqueeze(1),torch.mean(x,1).unsqueeze(1)),dim=1)

class _CBAM_BasicConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,relu=True,IN=True,bias=False):
        super(_CBAM_BasicConv, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                              stride=stride,padding=padding)
        self.instancenorm = nn.InstanceNorm3d(out_channels,affine=True) if IN else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self,x):
        x = self.conv(x)
        if self.instancenorm is not None:
            x = self.instancenorm(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class _CBAM_SpatialGate(nn.Module):
    """
    CBAM: Convolutional Block Attention Module
                https://arxiv.org/abs/1807.06521
    """
    def __init__(self,kernel_size,relu,IN):
        super(_CBAM_SpatialGate, self).__init__()
        self.kernel_size = kernel_size
        self.compress = _CBAM_ChannelPool()
        self.spatial  = _CBAM_BasicConv(in_channels=2,out_channels=1,kernel_size=self.kernel_size,stride=1,
                                        padding=(self.kernel_size-1)//2,relu=relu,IN=IN)
        self.sigmoid_layer = nn.Sigmoid()
        self.feature1 = None
    def forward(self,x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = self.sigmoid_layer(x)
        if not self.training:
            self.feature1 = scale.detach()
        x = x*scale #broadcast
        return x

"""**************************************CBAM Spatial Gate*******************************************"""


"""**************************************Split Attention*********************************************"""
class SplitChannelAtConv3D(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,
                 groups,dilation,bias=True,radix=2,reduction_factor=4,use_in=True,**kwargs):
        super(SplitChannelAtConv3D, self).__init__()
        inter_channels = max(in_channels*radix//reduction_factor,32)
        self.cardinality = groups
        self.radix = radix
        self.out_channels = out_channels
        self.avg_pool = nn.AdaptiveMaxPool3d(1)
        self.conv = nn.Conv3d(in_channels=in_channels,out_channels=out_channels*self.radix,
                              kernel_size=kernel_size,stride=stride,padding=padding,
                              dilation=dilation,groups=groups*radix,bias=bias)
        self.in1 = nn.InstanceNorm3d(out_channels*self.radix) if use_in else None
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv3d(in_channels=out_channels,out_channels=inter_channels,
                             kernel_size=1,groups=self.cardinality)
        self.in2 = nn.InstanceNorm3d(inter_channels) if use_in else None
        self.fc2 = nn.Conv3d(in_channels=inter_channels,out_channels=out_channels*self.radix,
                             kernel_size=1,groups=self.cardinality)
        self.activation = nn.Softmax(dim=1)

    def forward(self,x):
        #conv 3*3
        x = self.conv(x)
        if self.in1 is not None:
            x = self.in1(x)
        x = self.relu(x)
        #Split Attention
        batch,channel = x.shape[:2]
        splited = torch.split(x,split_size_or_sections=channel//self.radix,dim=1)
        gap = sum(splited)
        gap = self.avg_pool(gap)
        gap = self.fc1(gap)
        if self.in2 is not None:
            gap = self.in2(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap).view((batch,self.radix,self.out_channels))
        atten = self.activation(atten).view(batch,-1,1,1,1)
        atten = torch.split(atten,channel//self.radix,dim=1)
        out = sum([att * split for (att, split) in zip(atten, splited)])
        return out

class _Split_Channel_Attention(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,padding=1,dilation=1,
                 groups=1,radix=2,cardinality=1,**kwargs):
        super(_Split_Channel_Attention, self).__init__()
        self.radix = radix
        group_width = out_channels*cardinality
        self.conv1 = nn.Conv3d(in_channels,group_width,kernel_size=1,bias=False)
        self.in1 = nn.InstanceNorm3d(group_width)
        self.conv2 = SplitChannelAtConv3D(group_width,group_width,kernel_size=3,stride=stride,padding=padding,
                                 groups=cardinality,dilation=dilation,bias=False,radix=self.radix,use_in=True)
        self.conv3 = nn.Conv3d(group_width,out_channels,kernel_size=1,bias=False)
        self.in2 = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)



    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.in1(x)
        out = self.relu(x)
        out = self.conv2(x)
        out = self.conv3(x)
        out = self.in2(x)
        out +=residual
        out = self.relu(out)
        return out
"""**************************************Split Attention*********************************************"""


"""**************************************Dual Attention*********************************************"""
class DA_PAM_3D(nn.Module):
    """DA PAM 3D for backbone?"""
    def __init__(self,in_channels,reduction=4):
        super(DA_PAM_3D, self).__init__()
        self.in_channels=in_channels
        self.query_conv = nn.Conv3d(in_channels=in_channels,out_channels=in_channels//4,kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_channels,out_channels=in_channels//4,kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_channels,out_channels=in_channels,kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)


    def forward(self,x):
        """
        :param x:
        :return:
        """
        batchsize,channel,d,h,w = x.size()
        proj_query = self.query_conv(x).view(batchsize,-1,h*w).permute(0,2,1)
        proj_key = self.key_conv(x).view(batchsize,-1,h*w)
        Energy = torch.matmul(proj_query,proj_key)
        attention = self.softmax(Energy)
        proj_value = self.value_conv(x).view(batchsize,-1,h*w)

        out = torch.matmul(proj_value,attention.permute(0,2,1))
        out = out.view(batchsize,channel,d,h,w)

        out = self.gamma*out + x
        return out


class DA_PAMDaxis_3D(nn.Module):
    """DA PAM 3D for backbone?"""
    def __init__(self,in_channels,reduction=4):
        super(DA_PAMDaxis_3D, self).__init__()
        self.in_channels=in_channels
        self.query_conv = nn.Conv3d(in_channels=in_channels,out_channels=in_channels//4,kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_channels,out_channels=in_channels//4,kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_channels,out_channels=in_channels,kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)


    def forward(self,x):
        """
        :param x:
        :return:
        """
        batchsize,channel,d,h,w = x.size()
        proj_query = self.query_conv(x).view(batchsize,-1,d).permute(0,2,1)
        proj_key = self.key_conv(x).view(batchsize,-1,d)
        Energy = torch.matmul(proj_query,proj_key)
        attention = self.softmax(Energy)
        proj_value = self.value_conv(x).view(batchsize,-1,d)

        out = torch.matmul(proj_value,attention.permute(0,2,1))
        out = out.view(batchsize,channel,d,h,w)

        out = self.gamma*out + x
        return out


class DA_CAM_3D(nn.Module):
    """DA CAM 3D for backbone?"""
    def __init__(self,in_channels):
        super(DA_CAM_3D, self).__init__()
        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        """
        :param x:
        :return:
                out = attention value + input feature
                attention: B*C*C
        """
        batchsize,channel,d,h,w = x.size()
        proj_query = x.view(batchsize,channel,-1)
        proj_key = x.view(batchsize,channel,-1).permute(0,2,1)
        Energy = torch.matmul(proj_query,proj_key)
        #Origin code implementation But why need that?
        #Energy = torch.max(Energy,-1,keepdim=True)[0].expand_as(Energy)-Energy
        attention = self.softmax(Energy)
        proj_value = x.view(batchsize,channel,-1)

        out = torch.matmul(attention,proj_value)
        out = out.view(batchsize,channel,d,h,w)
        out = self.gamma * out + x
        return out

class _DualAttention_3D(nn.Module):
    """Dual Attention for 3D data"""
    def __init__(self,in_channels,out_channels,reduction=4):
        super(_DualAttention_3D, self).__init__()
        self.inter_channels = in_channels//reduction

        self.conv_p1 = nn.Sequential(SingleConv(in_channels=in_channels,
                                                out_channels=self.inter_channels,
                                                kernel_size=3,order='cpi'))
        self.conv_c1 = nn.Sequential(SingleConv(in_channels=in_channels,
                                                out_channels=self.inter_channels,
                                                kernel_size=3,order='cpi'))

        self.PAM = DA_PAM_3D(in_channels=self.inter_channels)
        #self.PAM = DA_PAMDaxis_3D(in_channels=self.inter_channels)
        self.CAM = DA_CAM_3D(in_channels=self.inter_channels)

        self.conv_p2 = nn.Sequential(SingleConv(in_channels=self.inter_channels,
                                                out_channels=self.inter_channels,
                                                kernel_size=3,order='cpi'))
        self.conv_c2 = nn.Sequential(SingleConv(in_channels=self.inter_channels,
                                                out_channels=self.inter_channels,
                                                kernel_size=3,order='cpi'))

        #
        # self.conv_p3 = nn.Sequential(nn.Dropout3d(0.1,False),
        #                              *create_conv(in_channels=self.inter_channels,
        #                                           out_channels=out_channels,
        #                                           kernel_size=3,order='cpi'))
        # self.conv_c3 = nn.Sequential(nn.Dropout3d(0.1,False),
        #                              *create_conv(in_channels=self.inter_channels,
        #                                           out_channels=out_channels,
        #                                           kernel_size=3,order='cpi'))

        self.final_conv = nn.Sequential(nn.Dropout3d(0.1,False),
                                        SingleConv(in_channels=self.inter_channels,
                                                     out_channels=out_channels,
                                                     kernel_size=3,order='cpi'))


    def forward(self,x):
        p1  = self.conv_p1(x)
        pa1 = self.PAM(p1)
        pa1_conv = self.conv_p2(pa1)

        c1  = self.conv_c1(x)
        ca1 = self.CAM(c1)
        ca1_conv = self.conv_c2(ca1)
        feature_sum = pa1_conv+ca1_conv

        pasa = self.final_conv(feature_sum)

        return pasa
"""**************************************Dual Attention*********************************************"""
