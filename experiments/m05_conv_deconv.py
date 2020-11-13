from argparse import Namespace
import train
import numpy as np
import torch
np.random.seed(42)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



model = 'm05'
opt_v = '03'


if opt_v == '01':
    dataset = '1K'
    experiment_name = 'm05_' + opt_v + '_conv_deconv_' + dataset
    opt_01 = Namespace(batch_size=1, beta1=0.5, checkpoints_dir='./checkpoints/', continue_train=False, dataroot='./../datasets/'+dataset,
              dataset_mode='unaligned', direction='AtoB', display_env='main', display_freq=50, display_id=1,
              display_ncols=4, display_port=8097, display_server='http://localhost', display_winsize=256, epoch='latest',
              epoch_count=1, fineSize=64, fpn_weights=[1.0, 1.0, 1.0, 1.0], gpu_ids=[0], init_gain=0.02, init_type='normal',
              input_nc=3, isTrain=True, lambda_A=10.0, lambda_B=10.0, lambda_identity=0.5, loadSize=64, lr=0.0002,
              lr_decay_iters=50, lr_policy='lambda', max_dataset_size=float("inf"), model=model, n_layers_D=3,
              name=experiment_name, ndf=16, netD='basic', netG='resnet_fpn', ngf=16, niter=35, niter_decay=35,
              no_dropout=True, no_flip=False, no_html=False, no_lsgan=False, norm='instance', num_threads=8, output_nc=3,
              phase='train', pool_size=50, print_freq=100, resize_or_crop='resize_and_crop', save_epoch_freq=1,
              save_latest_freq=1000, serial_batches=False, suffix='', update_html_freq=1000, verbose=False)
    train.train(opt_01)

elif opt_v == '02':
    # Use different Lambda Values
    dataset = '1K'
    experiment_name = 'm05_' + opt_v + '_conv_deconv_' + dataset
    opt_02 = Namespace(batch_size=1, beta1=0.5, checkpoints_dir='./checkpoints/', continue_train=False, dataroot='./../datasets/'+dataset,
              dataset_mode='unaligned', direction='AtoB', display_env='main', display_freq=50, display_id=1,
              display_ncols=4, display_port=8097, display_server='http://localhost', display_winsize=256, epoch='latest',
              epoch_count=1, fineSize=64, fpn_weights=[1.0, 1.0, 1.0, 1.0], gpu_ids=[0], init_gain=0.02, init_type='normal',
              input_nc=3, isTrain=True, lambda_A=20.0, lambda_B=20.0, lambda_identity=0.5, loadSize=64, lr=0.0002,
              lr_decay_iters=50, lr_policy='lambda', max_dataset_size=float("inf"), model=model, n_layers_D=3,
              name=experiment_name, ndf=16, netD='basic', netG='resnet_fpn', ngf=16, niter=35, niter_decay=35,
              no_dropout=True, no_flip=False, no_html=False, no_lsgan=False, norm='instance', num_threads=8, output_nc=3,
              phase='train', pool_size=50, print_freq=100, resize_or_crop='resize_and_crop', save_epoch_freq=1,
              save_latest_freq=1000, serial_batches=False, suffix='', update_html_freq=1000, verbose=False)
    train.train(opt_02)

elif opt_v == '03':
    # 10 K set
    dataset = '10K'
    experiment_name = 'm05_' + opt_v + '_conv_deconv_' + dataset
    opt_02 = Namespace(batch_size=1, beta1=0.5, checkpoints_dir='./checkpoints/', continue_train=False, dataroot='./../datasets/'+dataset,
              dataset_mode='unaligned', direction='AtoB', display_env='main', display_freq=50, display_id=1,
              display_ncols=4, display_port=8097, display_server='http://localhost', display_winsize=256, epoch='latest',
              epoch_count=1, fineSize=64, fpn_weights=[1.0, 1.0, 1.0, 1.0], gpu_ids=[0], init_gain=0.02, init_type='normal',
              input_nc=3, isTrain=True, lambda_A=20.0, lambda_B=20.0, lambda_identity=0.5, loadSize=64, lr=0.0002,
              lr_decay_iters=50, lr_policy='lambda', max_dataset_size=float("inf"), model=model, n_layers_D=3,
              name=experiment_name, ndf=16, netD='basic', netG='resnet_fpn', ngf=16, niter=17, niter_decay=17,
              no_dropout=True, no_flip=False, no_html=False, no_lsgan=False, norm='instance', num_threads=8, output_nc=3,
              phase='train', pool_size=50, print_freq=100, resize_or_crop='resize_and_crop', save_epoch_freq=1,
              save_latest_freq=1000, serial_batches=False, suffix='', update_html_freq=1000, verbose=False)
    train.train(opt_02)