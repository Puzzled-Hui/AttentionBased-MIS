from .base_options import BaseOptions



class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options

        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--eval',type=bool,default=True, help='use eval mode during test time.')
        parser.add_argument('--log_info',type=bool,default=True, help='use eval mode during test time.')
        parser.add_argument('--write',type=bool,default=True, help='use eval mode during test time.')
        parser.add_argument('--doc',type=str,default='./docs/MALC_3D UNet_Metrics.txt', help='record different average metrics.')
        parser.add_argument('--doc_dec',type=str,default='./docs/MALC_3D UNet_Dice.txt', help='doc_dice_each_class.')
        # Modify the BaseOptions
        parser.set_defaults(batch_size=1)
        parser.set_defaults(num_threads=0)
        parser.set_defaults(shuffle=False)
        self.isTrain = False
        return parser