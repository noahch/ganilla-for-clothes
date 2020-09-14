import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

import torch.nn.functional as F

import math
import torch.utils.model_zoo as model_zoo
import sys


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


class M03Model(BaseModel):
    def name(self):
        return 'M03Model'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = self.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                    not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, depth=18,
                                    fpn_weights=opt.fpn_weights)
        self.netG_B = self.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                    not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, depth=18,
                                    fpn_weights=opt.fpn_weights)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = self.define_D(opt.output_nc, opt.ndf, opt.netD,
                                        opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                        self.gpu_ids)
            self.netD_B = self.define_D(opt.input_nc, opt.ndf, opt.netD,
                                        opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                        self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        # print("FakeB")
        self.fake_B = self.netG_A(self.real_A)
        # print("RecA")
        self.rec_A = self.netG_B(self.fake_B)

        # print("FakeA")
        self.fake_A = self.netG_B(self.real_B)
        # print("RecB")
        self.rec_B = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A(self.real_B)
            # print(self.idt_A.size())
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward

        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

    def define_G(self, input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal',
                 init_gain=0.02, gpu_ids=[], depth=18, fpn_weights=[1.0, 1.0, 1.0, 1.0]):

        norm_layer = get_norm_layer(norm_type=norm)
        net = AblationModel2(ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)

        return init_net(net, init_type, init_gain, gpu_ids)

    def define_D(self, input_nc, ndf, netD,
                 n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
        norm_layer = get_norm_layer(norm_type=norm)

        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)

        return init_net(net, init_type, init_gain, gpu_ids)

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class AblationModel2(nn.Module):
    def __init__(self, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        self.inplanes = 64
        self.ngf = ngf
        super(AblationModel2, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.input_nc = 3
        self.output_nc = 3

        fpn_sizes = []

        init_part = [nn.ReflectionPad2d(3),
                 nn.Conv2d(self.input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        self.init_part = nn.Sequential(*init_part)
        fpn_sizes.append(ngf)

        n_downsampling = 2
        # for i in range(n_downsampling):
        #     mult = 2 ** i
        mult = 1
        down1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                            stride=2, padding=1, bias=False),
                  nn.InstanceNorm2d(ngf * mult * 2),
                  nn.ReLU(True)]

        self.down1 = nn.Sequential(*down1)

        fpn_sizes.append(ngf * mult * 2)

        mult = 2
        down2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                            stride=2, padding=1, bias=False),
                  nn.InstanceNorm2d(ngf * mult * 2),
                  nn.ReLU(True)]
        self.down2 = nn.Sequential(*down2)

        # fpn_sizes.append(ngf * mult * 2)

        flat_part = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            flat_part += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]
        self.flat_part = nn.Sequential(*flat_part)

        fpn_sizes.append(ngf * mult)

        self.fpn = PyramidFeatures_v3(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        # print("FPN SIZES: {}" .format(fpn_sizes))
        final_part = [nn.ReflectionPad2d(3)]
        final_part += [nn.Tanh()]
        final_part += [nn.Conv2d(self.ngf, 3, kernel_size=7, padding=0)]
        self.final_part = nn.Sequential(*final_part)

    def forward(self, inputs):

        img_batch = inputs
        # print("INPUT SIZE: {}".format(inputs.size()))
        x = self.init_part(img_batch)
        # print("INIT PART SIZE: {}".format(x.size()))
        x1 = self.down1(x)
        # print("DOWN1 SIZE: {}".format(x1.size()))
        x2 = self.down2(x1)
        # print("DOWN2 SIZE: {}".format(x2.size()))
        x3 = self.flat_part(x2)
        # print("FLAT SIZE: {}".format(x3.size()))

        out = self.fpn([x, x1, x3])
        # print("FPN SIZE: {}".format(out.size()))
        out = self.final_part(out)
        # print("FINAL SIZE: {}".format(out.size()))
        return out



# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out



class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class PyramidFeatures_v3(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=32):
        super(PyramidFeatures_v3, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        #self.rp1 = nn.ReflectionPad2d(1)
        #self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=0)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        #self.rp2 = nn.ReflectionPad2d(1)
        #self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=0)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.rp3 = nn.ReflectionPad2d(1)
        # self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=0)

        #self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P2_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.rp4 = nn.ReflectionPad2d(1)
        # self.P2_2 = nn.Conv2d(feature_size, feature_size/2, kernel_size=3, stride=1, padding=0)

        self.P1_1 = nn.Conv2d(feature_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P1_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.rp5 = nn.ReflectionPad2d(1)
        self.P1_2 = nn.Conv2d(feature_size, feature_size//2, kernel_size=3, stride=1, padding=0)

    def forward(self, inputs):

        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        # print("P5_Upsamled: {}".format(P5_upsampled_x.size()))

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        # print("P5_Upsamled: {}".format(P4_upsampled_x.size()))

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)

        P1_x = self.P1_1(P3_upsampled_x)
        P1_upsampled_x = self.P1_upsampled(P1_x)
        # P2_x = self.rp5(P1_upsampled_x)
        # P2_x = self.P1_2(P2_x)
        P2_x = self.rp5(P4_upsampled_x)
        P2_x = self.P1_2(P2_x)

        return P2_x
