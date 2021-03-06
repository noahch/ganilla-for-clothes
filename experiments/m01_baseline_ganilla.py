from argparse import Namespace
import train
import numpy as np
import torch
np.random.seed(42)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

experiment_name = 'm01_baseline_ganilla_1k'
dataset = '1K'
model = 'm01'

opt = Namespace(batch_size=1, beta1=0.5, checkpoints_dir='./checkpoints/', continue_train=False, dataroot='./../datasets/'+dataset,
          dataset_mode='unaligned', direction='AtoB', display_env='main', display_freq=50, display_id=1,
          display_ncols=4, display_port=8097, display_server='http://localhost', display_winsize=256, epoch='latest',
          epoch_count=1, fineSize=64, fpn_weights=[1.0, 1.0, 1.0, 1.0], gpu_ids=[0], init_gain=0.02, init_type='normal',
          input_nc=3, isTrain=True, lambda_A=10.0, lambda_B=10.0, lambda_identity=0.5, loadSize=70, lr=0.0002,
          lr_decay_iters=50, lr_policy='lambda', max_dataset_size=float("inf"), model=model, n_layers_D=3,
          name=experiment_name, ndf=64, netD='basic', netG='resnet_fpn', ngf=64, niter=35, niter_decay=35,
          no_dropout=True, no_flip=False, no_html=False, no_lsgan=False, norm='instance', num_threads=8, output_nc=3,
          phase='train', pool_size=50, print_freq=100, resize_or_crop='resize_and_crop', save_epoch_freq=1,
          save_latest_freq=1000, serial_batches=False, suffix='', update_html_freq=1000, verbose=False)

train.train(opt)
