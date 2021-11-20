import argparse
import os
import torch
import models
import data
from util import utils


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self,parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', type=str,default="E:\\Dataset\\MALC11_Fold1\\MALC11_1.0\\",help='path to images,train and test fold')
        parser.add_argument('--name', type=str, default='unet3d',help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0',help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='E:\Studyfile_ZMH\GraduationProject\code\AttentionBased-MIS\checkpoints\\brats\\unet3d\\',help='models are saved here')

        # model parameters
        parser.add_argument('--model', type=str, default='unet3d',help='chooses which model to use. [unet3d|cSEunet3d|scSEunet3d|agscSEunet3d]')
        """Modify 0116 zmh"""
        parser.add_argument('--se_channel_pooling_type', type=str, default='avg',help='chooses which pool function to use. [max|avg|avg_and_max]')
        parser.add_argument('--se_slice_pooling_type', type=str, default='avg',help='chooses which pool function to use. [max|avg|avg_and_max]')
        parser.add_argument('--use_spatial_attention', action='store_true',help='whether use attention in 3D UNet')
        parser.add_argument('--in_channels', type=int, default=1,help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--out_channels', type=int, default=11,help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--final_sigmoid', action='store_true',help='whether it is binary segmentation,Yes,use sigmoid,No,use softmax')
        parser.add_argument('--init_fmaps_degree', type=int, default=16, help='#the number of init_fmaps_degree channels')
        parser.add_argument('--layer_order', type=str, default='cpi',help='three functions permutation and combination, e.g.conv-relu-instancenorm')
        parser.add_argument('--fmaps_layer_number', type=int, default=4,help='the number of layers of the network')
        parser.add_argument('--group_normnumber', type=int, default=2,help='the number use for GroupNorm')
        parser.add_argument('--init_type', type=str, default='normal',help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,help='scaling factor for normal, xavier and orthogonal.')

        # dataset parameters
        parser.add_argument('--loss_mode', type=str, default='CBL', help='BCE,DICE,CE,CBL,etc')
        parser.add_argument('--ignore_index', type=int, default=None, help='use to control the dice_loss_dimension choose which to ignore')
        parser.add_argument('--sigmoid_normalization', action='store_true',help='True,use the sigmoid, False,use the softmax')
        parser.add_argument('--dataset_mode', type=str, default='SingleBratsTumor18', help='chooses how datasets are loaded')
        parser.add_argument('--shuffle', type=bool,default=True,help='if true,takes data batch randomly')
        parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')

        # additional parameters
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--use_Adaboost',  action='store_true', help='Use adaboost function? feeback the train dice')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default=0, help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        self.initialized = True
        return parser


    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()


    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        utils.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')


    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        #set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        #if len(opt.gpu_ids) > 0:
            #torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt