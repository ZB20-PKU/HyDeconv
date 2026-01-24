import os
import torch
import numpy as np
from src_BF.basic import *
from src.ImageLib import ImageLib


class Recon:
    def __init__(self, args, otf):
        self.device = args.device

        self.savedir = args.savedir

        self.imgsize = args.imgsize

        self.fc = args.fc
        self.recon_order = 1#args.recon_order
        self.recon_gain = 2#args.recon_gain
        self.num_angle = args.SIM_Pattern_orientation_number
        self.num_phase = args.SIM_Pattern_phase_number
        self.apodization_type = 'cos'#args.apodization_type
        # self.triangle_apod_gamma = args.triangle_apod_gamma
        # self.triangle_apod_scale = args.triangle_apod_scale
        # self.NotchFilter_params = args.NotchFilter_params

        # 坐标格点
        self.mid = self.num_phase // 2
        self.N_recon = self.imgsize[0] * self.recon_gain
        self.gridx, self.gridy = calcGrid(self.N_recon, device=self.device)

        # 掩膜
        self.mask = createMask(self.N_recon, self.fc, device=self.device)

        # 重建的阶数
        if self.recon_order > self.mid or self.recon_order < 0:
            self.recon_order = self.mid

        # otf
        self.otf = padZero(otf, [self.N_recon, self.N_recon]) * self.mask
        self.psf = torch.abs(myifft2d(self.otf))
        self.psf = self.psf / self.psf.sum()
        # ImageLib.write(self.savedir + '/psf.tif', self.psf / self.psf.max())

        self.p = None
        self.plength = None
        self.apod = None
        self.otf_shift = torch.zeros((self.num_angle * self.num_phase, self.N_recon, self.N_recon),
                                     dtype=self.otf.dtype,
                                     device=self.device)
        self.mask_shift = torch.zeros((self.num_angle * self.num_phase, self.N_recon, self.N_recon),
                                      dtype=self.otf.dtype,
                                      device=self.device)

    # 切趾函数
    # 余弦切趾函数
    def apodization_cos(self, fc):
        R = torch.sqrt(self.gridx ** 2 + self.gridy ** 2)
        pi = torch.tensor(np.pi)
        out = torch.cos(2 * pi / fc / 4 * R)
        out[R > fc] = 0
        return out

    # 三角切趾函数
    def apodization_triangle(self, fc, gamma=0.4, scale=1.0):
        R = torch.sqrt(self.gridx ** 2 + self.gridy ** 2)
        R = 1 - scale * R / fc
        R[R < 0] = 0
        R = torch.pow(R, gamma)
        R[R < 0] = 0
        return R

    def apodization(self, plength):
        plength_mean = plength.sum() / torch.abs(
            torch.arange(-self.mid, self.mid + 1, 1)).sum() / self.num_angle
        if self.apodization_type == 'otf':
            apod = createOTF((self.N_recon, self.N_recon), fc=self.fc + plength_mean * self.recon_order,
                             device=self.device)
        elif self.apodization_type == 'cos':
            apod = self.apodization_cos(self.fc + plength_mean * self.recon_order)
        elif self.apodization_type == 'triangle':
            apod = self.apodization_triangle(self.fc + plength_mean * self.recon_order,
                                             gamma=self.triangle_apod_gamma,
                                             scale=self.triangle_apod_scale)
        else:
            print('error apodization type!!!')
            return None

        return apod

    # 陷波滤波
    def notchfilter(self):
        imgsize = self.imgsize[0]
        alpha = self.NotchFilter_params
        R2 = self.gridx ** 2 + self.gridy ** 2
        nf = 1 - torch.exp(-R2 / (alpha ** 2))
        return nf.to(self.device)

    def apply_notchfilter(self, sp):
        if self.NotchFilter_params != 0:
            nf = self.notchfilter()
            for i in range(self.num_angle * self.num_phase):
                if np.mod(i, self.num_phase) - self.mid != 0:
                    sp[i, :, :] = sp[i, :, :] * nf
        return sp

    def set_params(self, p, plength):
        self.p = p
        self.plength = plength
        self.apod = self.apodization(plength)

        for i in range(self.num_angle * self.num_phase):
            if abs(np.mod(i, self.num_angle) - self.mid) > self.recon_order:
                continue
            tmpmask = torch.abs(shift(self.mask, p[i, :], self.gridx, self.gridy))
            tmpmask[tmpmask < 0.8] = 0
            tmpmask[tmpmask != 0] = 1
            self.mask_shift[i, :, :] = tmpmask
            self.otf_shift[i, :, :] = torch.abs(shift(self.otf, p[i, :], self.gridx, self.gridy)) * tmpmask

    # 重建
    def recon(self, **kwargs):
        pass
