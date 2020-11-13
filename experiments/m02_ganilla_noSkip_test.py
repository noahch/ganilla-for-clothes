from argparse import Namespace
import test
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

opt_v = '01'
model = 'm02'
epoch = 'latest'
epochs_to_test = ['latest', '1', '15', '30', '45', '60', '70']

if opt_v == '01':
    dataset = '1K'
    experiment_name = 'm02_ganilla_noSkip_' + dataset
    opt_01 = Namespace(aspect_ratio=1.0, batch_size=1, checkpoints_dir='checkpoints/',
                    cityscape_fnames='./datasets/cityscapes_test_file_names.txt', cityscapes=False,
                    dataroot='../datasets/'+dataset, dataset_mode='unaligned', direction='AtoB', display_winsize=256,
                    epoch=epoch, eval=False, fineSize=64, fpn_weights=[1.0, 1.0, 1.0, 1.0], gpu_ids=[0], init_gain=0.02,
                    init_type='normal', input_nc=3, isTrain=False, loadSize=64, max_dataset_size=float("inf"), model=model,
                    n_layers_D=3, name=experiment_name, ndf=64, netD='basic', netG='resnet_fpn', ngf=64,
                    no_dropout=True, no_flip=False, norm='instance', ntest=float("inf"), num_test=751, num_threads=8, output_nc=3,
                    phase='test', resize_or_crop='resize_and_crop', results_dir='./results/', serial_batches=False,
                    suffix='', verbose=False)

    for e in epochs_to_test:
        opt_01.epoch = e
        test.test(opt_01)
