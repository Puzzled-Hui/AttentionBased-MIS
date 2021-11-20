from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """
    def initialize(self,parser):
        parser = BaseOptions.initialize(self,parser)
        # visualization parameters
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')

        # training parameters
        parser.add_argument('--optim_mode', type=str, default='Adam', help='Adam,SGD,etc')
        parser.add_argument('--init_learningrate', default=0.002, type=float, help='the initial learning rate')
        parser.add_argument('--lr_policy', type=str, default='step',help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--plateau_patience', type=int, default=100,help='plateau_patience')
        parser.add_argument('--lr_decay_iters', type=int, default=10,help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--total_epoch', type=int, default=80,help='how many epochs to train')
        parser.add_argument('--train', type=bool, default=True, help='use train mode during test time.')

        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=200, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5,help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', type=bool,default=False, help='whether saves model by iteration')

        parser.add_argument('--epoch_count', type=int, default=1,help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        self.isTrain = True
        return parser

