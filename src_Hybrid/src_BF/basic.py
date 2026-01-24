import numpy as np
import torch
import scipy.special
from src.ImageLib import ImageLib
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import shutil

if sys.platform == 'win32':
    import winsound


def normalize(stack):
    if len(stack.shape) == 2:
        return (stack - stack.min()) / (stack.max() - stack.min())
    else:
        for i in range(stack.shape[0]):
            tmp = stack[i, ::]
            stack[i, ::] = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    return stack

# (反)傅里叶变换默认移频至中心
def myfft2d(stack):
    num_dim = len(stack.shape)
    stack = torch.fft.ifftshift(stack, dim=(num_dim - 2, num_dim - 1))
    stack = torch.fft.fft2(stack)
    stack = torch.fft.fftshift(stack, dim=(num_dim - 2, num_dim - 1))
    return stack


def myifft2d(stack):
    num_dim = len(stack.shape)
    stack = torch.fft.ifftshift(stack, dim=(num_dim - 2, num_dim - 1))
    stack = torch.fft.ifft2(stack)
    stack = torch.fft.fftshift(stack, dim=(num_dim - 2, num_dim - 1))
    return stack


# 补零
def padZero(stack, outShape):
    # outShape需要与stack的尺寸对应，三维对三维，二维对二维
    inShape = stack.shape
    if stack.ndim == 2:
        orimidx = inShape[1] // 2
        orimidy = inShape[0] // 2
        tarmidx = outShape[1] // 2
        tarmidy = outShape[0] // 2
        leftx = tarmidx - orimidx
        lefty = tarmidy - orimidy
        out = torch.zeros(outShape, dtype=stack.dtype).to(stack.device)
        out[lefty:lefty + inShape[0], leftx:leftx + inShape[1]] = stack
        return out
    elif stack.ndim == 3:
        orimidx = inShape[2] // 2
        orimidy = inShape[1] // 2
        orimidz = inShape[0] // 2
        tarmidx = outShape[2] // 2
        tarmidy = outShape[1] // 2
        tarmidz = outShape[0] // 2
        leftx = (tarmidx - orimidx)
        lefty = (tarmidy - orimidy)
        leftz = (tarmidz - orimidz)
        out = torch.zeros(outShape, dtype=stack.dtype).to(stack.device)
        out[leftz:leftz + inShape[0], lefty:lefty + inShape[1], leftx:leftx + inShape[2]] = stack
        return out
    return None


# 去除补的零
def removePadZero(stack, outshape):
    inshape = stack.shape
    if stack.ndim == 2:
        orimidx = outshape[1] // 2
        orimidy = outshape[0] // 2
        curmidx = inshape[1] // 2
        curmidy = inshape[0] // 2
        leftx = curmidx - orimidx
        lefty = curmidy - orimidy
        return stack[lefty:lefty + outshape[0], leftx:leftx + outshape[1]]
    elif stack.ndim == 3:
        orimidz = outshape[0] // 2
        orimidx = outshape[2] // 2
        orimidy = outshape[1] // 2
        curmidz = inshape[0] // 2
        curmidx = inshape[2] // 2
        curmidy = inshape[1] // 2
        leftx = curmidx - orimidx
        lefty = curmidy - orimidy
        leftz = curmidz - orimidz
        return stack[leftz:leftz + outshape[0], lefty:lefty + outshape[1], leftx:leftx + outshape[2]]
    return None


# 频谱移动，arr为二维矩阵,delta=[delta_x,deltay]
def shift(arr, delta, gridx=None, gridy=None):
    if delta[0] == 0 and delta[1] == 0:
        return arr

    sy = arr.shape[0]
    sx = arr.shape[1]

    if gridx is None or gridy is None:
        mid_x = sx // 2
        mid_y = sy // 2
        gridy, gridx = torch.meshgrid(torch.arange(0, sy, 1), torch.arange(0, sx, 1), indexing='ij')
        gridy = gridy - mid_y
        gridx = gridx - mid_x

    tmp = torch.exp(1j * 2 * np.pi * delta[0] / sx * gridx + 1j * 2 * np.pi * delta[1] / sy * gridy)
    out = myifft2d(arr)
    out = myfft2d(out * tmp)
    return out


# fft之后相乘替代卷积，x/y为二维矩阵
# out='same': 输出和x尺寸相同
def myConv2(x, y, out=None):
    tmpShape = torch.tensor(x.shape) + torch.tensor(y.shape) - 1
    tmpShape = tuple(tmpShape.numpy())
    f = torch.fft.ifft2(torch.fft.fft2(x, tmpShape) * torch.fft.fft2(y, tmpShape))
    if out == 'same':
        start_y = y.shape[0] // 2
        start_x = y.shape[1] // 2
        f = f[start_y:start_y + x.shape[0], start_x:start_x + x.shape[1]]
    return f


# 通过一阶贝塞尔函数产生仿真PSF和otf
def createPSF(imgsize, NA, pixelsize, wavelength, device):
    scale = 2 * np.pi * NA * pixelsize / wavelength
    sizey, sizex = imgsize
    halfy = sizey // 2
    halfx = sizex // 2
    Y, X = torch.meshgrid(torch.arange(0, sizey, 1), torch.arange(0, sizex, 1), indexing='ij')
    X = X - halfx
    Y = Y - halfy
    R = torch.sqrt(X ** 2 + Y ** 2)
    PSF = torch.abs(2 * scipy.special.j1(scale * R + np.spacing(1)) / (scale * R + np.spacing(1))) ** 2
    PSF = PSF / PSF.sum()
    OTF = myfft2d(PSF)
    return PSF.to(device), OTF.to(device)


# 产生仿真OTF，参考HiFi-SIM
def createOTF(imgsize, fc, device, beta=1.0, center=(0, 0)):
    sizey, sizex = imgsize
    fc = fc.to(device)
    halfy = sizey // 2
    halfx = sizex // 2
    Y, X = torch.meshgrid(torch.arange(0, sizey, 1), torch.arange(0, sizex, 1), indexing='ij')
    X = X - halfx - center[1]
    Y = Y - halfy - center[0]
    R = torch.sqrt(X ** 2 + Y ** 2).to(device).to(fc.dtype)
    R[R >= fc] = fc
    b = torch.arccos(torch.abs(R / fc))
    OTF = (2 * b - torch.sin(2 * b)) / np.pi
    OTF[R == fc] = 0
    OTF = OTF * torch.pow(beta, R)
    return OTF.to(device)


# 产生圆型mask, 尺寸为[imgsize,imgsize]
def createMask(imgsize, *args, device):
    halfy = imgsize // 2
    halfx = imgsize // 2
    Y, X = torch.meshgrid(torch.arange(0, imgsize, 1), torch.arange(0, imgsize, 1), indexing='ij')
    X = X - halfx
    Y = Y - halfy
    R = torch.sqrt(X ** 2 + Y ** 2) + 1e-10
    R = R.to(device)
    if len(args) == 1:
        R[R > args[0]] = 0
        R[R != 0] = 1
        return R
    elif len(args) == 2:
        R[R < args[0]] = 0
        R[R > args[1]] = 0
        R[R != 0] = 1
        return R
    else:
        return None


# sigmoid
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


# 计算坐标格点
def calcGrid(imgsize, device):
    mid = imgsize // 2
    Y, X = torch.meshgrid(torch.arange(0, imgsize, 1), torch.arange(0, imgsize, 1), indexing='ij')
    return (X - mid).to(device), (Y - mid).to(device)


# 寻找二维tensor最大值所在位置
def argmax(data):
    shape = torch.tensor(data.shape)
    index = data.argmax()
    y = torch.floor(index / shape[1]).int()
    x = index - y * shape[1]
    return y, x


# 产生正弦pattern
def createPattern(v, mean, amp, theta, phase, gridX, gridY, device):
    pi = torch.tensor(np.pi).to(device)
    p_theta_1 = v * torch.cos(theta)
    p_theta_2 = v * torch.sin(theta)

    pattern = mean + amp * torch.cos(2 * pi * (p_theta_1 * gridX + p_theta_2 * gridY) + phase)
    return pattern


def gasuss_noise(image, mean=0, var=0.001):
    """ image: 0~1 """
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    if torch.is_tensor(image):
        out = torch.from_numpy(noise).to(image.device) + image
    else:
        out = image + noise
    out[out < 0] = 0
    out = out / out.max()
    return out


def poisson_noise(image):
    """ image: 0~1 """
    image_np = (image * 255).cpu().numpy()
    noise = np.random.poisson(image_np).astype(np.float64)
    image = image + torch.from_numpy(noise).to(image.device)
    image = image / image.max()
    return image


def createGaussianPoint(sigma):
    Y, X = torch.meshgrid(torch.arange(-sigma * 5, sigma * 5 + 1, 1),
                          torch.arange(-sigma * 5, sigma * 5 + 1, 1),
                          indexing='ij')
    R = torch.sqrt(X ** 2 + Y ** 2)
    point = torch.exp(-R ** 2 / (2 * sigma ** 2))
    return point / point.sum()


def add_bg(image, min_sigma, max_sigma, gain=0.5):
    image = image.unsqueeze(0).unsqueeze(0).float()
    bg = torch.zeros_like(image)
    num = 0
    for sigma in range(min_sigma, max_sigma):
        kernel = createGaussianPoint(sigma).unsqueeze(0).unsqueeze(0).to(image.device)
        bg += torch.nn.functional.conv2d(image, kernel, padding='same')
        num += 1
    image = image + bg * gain
    image = image.squeeze(0).squeeze(0)
    bg = bg.squeeze(0).squeeze(0) / num
    max_val = image.max()
    return image / max_val, bg * gain / max_val


def clean_folder(path):
    filelist = os.listdir(path)
    for file in filelist:
        fullpath = os.path.join(path, file)
        if os.path.isdir(fullpath):
            os.removedirs(fullpath)
        else:
            os.remove(fullpath)

def mydelete(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)

def mymkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def mymove(oldfile, newfile):
    if os.path.isfile(oldfile):
        shutil.move(oldfile, newfile)

def write_log(path, args):
    with open(path, 'w') as f:
        f.write("光学系统相关：" + "\n")
        f.write("NA = " + str(args.NA) + "\n")
        f.write("wave length = " + str(args.wavelength * 1e9) + " nm" + "\n")
        f.write("pixel size = " + str(args.pixel_size * 1e9) + " nm" + "\n")
        f.write("background = " + str(args.background) + "\n")
        f.write("otf_path = " + args.otf_path + "\n")
        f.write("----------------------------------------------------------" + "\r\n")

        f.write("数据预处理：" + "\n")
        if args.block_normalize:
            f.write("block normalize" + "\n")
        f.write("sigmoid windows function sigma = " + str(args.sigmoid_win_sigma) + "\n")
        f.write("----------------------------------------------------------" + "\r\n")

        f.write("估计参数相关：" + "\n")
        f.write("num average = " + str(args.num_average) + "\n")
        if args.use_emd:
            f.write("use EMD algorithm" + "\n")
        else:
            f.write("no EMD algorithm" + "\n")
        f.write("----------------------------------------------------------" + "\r\n")

        f.write("重建相关：" + "\n")
        f.write("recon_gain = " + str(args.recon_gain) + "\n")
        f.write("recon_order = " + str(args.recon_order) + "\n")
        f.write("notch filter params = " + str(args.NotchFilter_params) + "\n")
        f.write("recon_method: " + args.recon_method + "\n")
        if args.recon_method == 'wiener':
            f.write("apodization: " + args.apodization_type + "\n")
            if args.apodization_type == 'triangle':
                f.write("triangle_apod_gamma = " + str(args.triangle_apod_gamma) + "\n")
                f.write("triangle_apod_scale = " + str(args.triangle_apod_scale) + "\n")
        elif args.recon_method == 'TDV':
            f.write("TDV_checkpoint: " + args.checkpoint_TDV + "\n")


# 发出警报声
def ding():
    duration = 800
    freq = 2000
    if sys.platform == 'win32':
        winsound.Beep(freq, duration)
    else:
        os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
