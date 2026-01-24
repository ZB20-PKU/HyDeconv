import torch
import torch.fft
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Union, Tuple
import warnings
from src.ImageLib import ImageLib
import os
from .kernel import PSFGenerator
from src_BF.RollingBallBackgroundSubtracter import *

class Sparse_deconvolution_SIM:
    """
    迭代去卷积 - 使用PyTorch加速
    支持多种去卷积算法和GPU加速
    """
    
    def __init__(self, args, progress_callback=None, device=None, dtype=torch.float32):
        
        self.dtype = dtype
        
        self.device = torch.device(args.device)
        self.progress_callback = progress_callback
        self._stop_requested = False

        # 预分配缓冲区
        self._buffers = {}
        self._cache = {}
        self.pixel_size = args.SIM_Raw_pixel_size/2
        self.wavelength = args.SIM_emission_wavelength
        self.Sparse_NA = args.Sparse_NA
        self.iteration_num = args.Sparse_iteration_number
        # self.filedir = args.filedir
        # self.fname = args.fname
        # self.ext = args.ext
        self.Sparse_offset = args.Sparse_offset
        self.TDV = torch.from_numpy((args.tdv_result).astype(np.float32)).to(args.device)
        self.Sparse_factor = self.TDV.max()
        # ImageLib.read(os.path.join(self.filedir,
        #                    self.fname + '_3_TDV'\
        #                    + self.ext), to_tensor=True) + args.Sparse_offset
        
    def recon(self):  
        generator = PSFGenerator(device=self.device) 
        kernel =  generator.generate_psf_torch_vectorized(self.pixel_size, \
            self.wavelength, 64, self.Sparse_NA, 0)
        Tmp = self.TDV / self.Sparse_factor + self.Sparse_offset
        im_sparse = self.iterative_deconv_batch(Tmp, kernel, self.iteration_num)
        im_sparse = (im_sparse - self.Sparse_offset) * self.Sparse_factor/2
        # ImageLib.write(os.path.join(self.filedir,
        #                    self.fname + '_4_Hybrid'\
        #                    + self.ext), im_sparse.cpu().numpy(), writemode='w') 
        # print(">> Finish Hybrid deconvolution.") 
        
        if self._stop_requested:
            # 清理内存
            self.cleanup()
            return None

        return im_sparse.cpu().numpy()

    def stop(self):
        """停止重建"""
        self._stop_requested = True
    
    def cleanup(self):
        """清理内存"""
        try:
            # 清理PyTorch GPU缓存
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 清理Python变量
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'optimizer'):
                del self.optimizer
            
            # 垃圾回收
            import gc
            gc.collect()
        except:
            pass

    def _prepare_input(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """准备输入数据"""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data.astype(np.float32)).to(self.device)
        elif isinstance(data, torch.Tensor):
            return data.to(self.device, dtype=self.dtype)
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")
    
    def _prepare_kernel(self, kernel: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """准备卷积核"""
        kernel_tensor = self._prepare_input(kernel)
        # 归一化
        kernel_tensor = kernel_tensor / torch.sum(kernel_tensor)
        return kernel_tensor
    
    def psf2otf_torch(self, psf: torch.Tensor, out_size: Tuple[int, ...]) -> torch.Tensor:
        """
        PyTorch版本的PSF转OTF
        将点扩散函数转换为光学传递函数
        """
        psf_size = torch.tensor(psf.shape, device=self.device)
        out_size_tensor = torch.tensor(out_size, device=self.device)
        
        # 计算填充尺寸
        pad_size = out_size_tensor - psf_size
        
        # 填充PSF
        psf_padded = torch.nn.functional.pad(
            psf, 
            (0, int(pad_size[1].item()), 0, int(pad_size[0].item())),
            mode='constant',
            value=0
        )
        
        # 循环移位（将中心移到0,0）
        for i, dim_size in enumerate(psf_size):
            shift = -int(dim_size.item() // 2)
            psf_padded = torch.roll(psf_padded, shifts=shift, dims=i)
        
        # 计算FFT
        otf = torch.fft.fftn(psf_padded, dim=(-2, -1))
        
        # 如果虚部很小，转换为实数
        if torch.max(torch.abs(otf.imag)) / torch.max(torch.abs(otf)) < 1e-6:
            otf = otf.real
        
        return otf
    
    def _pad_data(self, data: torch.Tensor, padding: int) -> torch.Tensor:
        """
        通用数据填充函数
        修复反射填充模式的问题
        """
        if padding <= 0:
            return data
        
        # 根据数据维度选择合适的填充策略
        if data.ndim == 2:
            # 2D数据：使用复制填充
            return F.pad(data.unsqueeze(0).unsqueeze(0), 
                        (padding, padding, padding, padding), 
                        mode='replicate').squeeze(0).squeeze(0)
        elif data.ndim == 3:
            # 3D数据：对每个通道单独填充
            batch_size = data.shape[0]
            padded_slices = []
            
            for i in range(batch_size):
                padded_slice = F.pad(data[i].unsqueeze(0).unsqueeze(0),
                                   (padding, padding, padding, padding),
                                   mode='replicate').squeeze(0).squeeze(0)
                padded_slices.append(padded_slice.unsqueeze(0))
            
            return torch.cat(padded_slices, dim=0)
        else:
            # 高维数据：使用常数填充
            return F.pad(data, 
                        (padding, padding, padding, padding), 
                        mode='constant', 
                        value=data.min().item())
    
    def richardson_lucy_torch(self, data: torch.Tensor, otf: torch.Tensor, 
                             iterations: int = 10, acceleration: bool = True) -> torch.Tensor:
        """
        Richardson-Lucy去卷积（PyTorch加速版）
        
        参数:
            data: 输入图像
            otf: 光学传递函数
            iterations: 迭代次数
            acceleration: 是否使用加速
            
        返回:
            去卷积后的图像
        """

        # 将输入数据中所有小于0的数置为0
        data = torch.clamp(data, min=0.0)

        # 初始化变量
        yk = data.clone()
        xk = torch.zeros_like(data)
        vk = torch.zeros_like(data)
        
        # 预计算一些常量
        ones_tensor = torch.ones_like(data)
        otf_conj = torch.conj(otf)
        
        for iter_idx in range(iterations):
            if self._stop_requested:
                break
            # if self.progress_callback:
            #     progress = 75 + int((iter_idx + 1) * 25 / iterations)  # 75-100%
            #     self.progress_callback(progress)
            # # 保存当前xk
            xk_prev = xk
            
            # RL迭代
            # 计算: xk = yk * IFFT(conj(OTF) * FFT(data / IFFT(OTF * FFT(yk))))
            fft_yk = torch.fft.fftn(yk, dim=(-2, -1))
            conv_result = torch.fft.ifftn(otf * fft_yk, dim=(-2, -1)).real
            conv_result = torch.maximum(conv_result, torch.tensor(1e-6, device=self.device))
            
            ratio = data / conv_result
            fft_ratio = torch.fft.fftn(ratio, dim=(-2, -1))
            numerator = torch.fft.ifftn(otf_conj * fft_ratio, dim=(-2, -1)).real
            
            # 分母
            denominator = torch.fft.ifftn(
                torch.fft.fftn(ones_tensor, dim=(-2, -1)) * otf, 
                dim=(-2, -1)
            ).real
            
            xk = yk * numerator / torch.maximum(denominator, torch.tensor(1e-6, device=self.device))
            xk = torch.maximum(xk, torch.tensor(1e-6, device=self.device))
            
            # 更新vk
            vk_prev = vk
            vk = torch.maximum(xk - yk, torch.tensor(1e-6, device=self.device))
            
            if not acceleration or iter_idx == 0:
                # 不使用加速
                alpha = 0
                yk = xk.clone()
            else:
                # 计算加速参数
                vk_prev_flat = vk_prev.flatten()
                vk_flat = vk.flatten()
                
                numerator_acc = torch.sum(vk_prev_flat * vk_flat)
                denominator_acc = torch.sum(vk_prev_flat * vk_prev_flat) + 1e-10
                alpha = torch.clamp(numerator_acc / denominator_acc, min=1e-6, max=1.0)
                
                # 加速更新
                yk = xk + alpha * (xk - xk_prev)
                yk = torch.maximum(yk, torch.tensor(1e-6, device=self.device))
            
            # 处理NaN
            yk = torch.nan_to_num(yk, nan=1e-6)
            
            # 每10次迭代清理内存
            if (iter_idx + 1) % 10 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # 确保非负
        yk = torch.maximum(yk, torch.tensor(0, device=self.device))
        
        return yk
    
    def landweber_torch(self, data: torch.Tensor, otf: torch.Tensor, 
                        iterations: int = 10, step_size: float = 1.0) -> torch.Tensor:
        """
        Landweber去卷积（PyTorch加速版）
        
        参数:
            data: 输入图像
            otf: 光学传递函数
            iterations: 迭代次数
            step_size: 步长参数
            
        返回:
            去卷积后的图像
        """
        # 初始化变量
        yk = data.clone()
        xk = torch.zeros_like(data)
        xk_prev = torch.zeros_like(data)
        
        # 预计算
        otf_conj = torch.conj(otf)
        t = step_size
        
        # 初始gamma值
        gamma1 = 1.0
        
        for iter_idx in range(iterations):
            if iter_idx == 0:
                # 第一次迭代
                fft_data = torch.fft.fftn(data, dim=(-2, -1))
                update_term = torch.fft.ifftn(
                    otf_conj * (fft_data - otf * fft_data), 
                    dim=(-2, -1)
                ).real
                xk = data + t * update_term
            else:
                # 计算gamma2和beta
                gamma2 = 0.5 * torch.sqrt(4 * gamma1**2 + gamma1**4) - gamma1**2
                beta = -gamma2 * (1 - 1 / gamma1)
                
                # 更新yk
                yk_update = xk + beta * (xk - xk_prev)
                
                # Landweber迭代
                fft_yk_update = torch.fft.fftn(yk_update, dim=(-2, -1))
                fft_data = torch.fft.fftn(data, dim=(-2, -1))
                
                update_term = torch.fft.ifftn(
                    otf_conj * (fft_data - otf * fft_yk_update), 
                    dim=(-2, -1)
                ).real
                
                yk = yk_update + t * update_term
                yk = torch.maximum(yk, torch.tensor(1e-6, device=self.device))
                
                # 更新gamma
                gamma1 = gamma2
                xk_prev = xk.clone()
                xk = yk.clone()
            
            # 每10次迭代清理内存
            if (iter_idx + 1) % 10 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # 确保非负
        xk = torch.maximum(xk, torch.tensor(0, device=self.device))
        
        return xk
    
    def deblur_core_torch(self, data: torch.Tensor, kernel: torch.Tensor, 
                         iterations: int = 10, rule: int = 1) -> torch.Tensor:
        """
        核心去卷积函数（PyTorch加速版）
        
        参数:
            data: 输入图像
            kernel: 卷积核
            iterations: 迭代次数
            rule: 算法规则 (1: Richardson-Lucy, 2: Landweber)
            
        返回:
            去卷积后的图像
        """
        # 获取输入尺寸
        dx, dy = data.shape[-2], data.shape[-1]
        
        # 计算边界填充尺寸
        B = min(dx, dy) // 6
        
        # 边缘填充（使用修复后的填充函数）
        padded_data = self._pad_data(data, B)
        
        # 计算OTF
        otf = self.psf2otf_torch(kernel, padded_data.shape)
        
        # 根据规则选择算法
        if rule == 1:  # Richardson-Lucy
            result = self.richardson_lucy_torch(padded_data, otf, iterations, acceleration=True)
        elif rule == 2:  # Landweber
            result = self.landweber_torch(padded_data, otf, iterations, step_size=1.0)
        else:
            raise ValueError(f"不支持的算法规则: {rule}")
        
        # 裁剪填充区域
        if result.ndim == 2:
            result_cropped = result[B:-B, B:-B]
        else:
            result_cropped = result[:, B:-B, B:-B]
        
        return result_cropped
    
    def iterative_deconv_batch(self, data: Union[np.ndarray, torch.Tensor], 
                              kernel: Union[np.ndarray, torch.Tensor],
                              iterations: int = 10, rule: int = 1) -> torch.Tensor:
        """
        批量迭代去卷积（支持2D和3D数据）
        
        参数:
            data: 输入数据，可以是2D或3D
            kernel: 卷积核
            iterations: 迭代次数
            rule: 算法规则
            
        返回:
            去卷积后的数据
        """
        # 准备输入数据
        data_tensor = self._prepare_input(data)
        kernel_tensor = self._prepare_kernel(kernel)
        
        if data_tensor.ndim == 2:
            # 2D数据
            result = self.deblur_core_torch(data_tensor, kernel_tensor, iterations, rule)
        elif data_tensor.ndim == 3:
            # 3D数据 - 逐片处理
            batch_size = data_tensor.shape[0]
            results = []
            
            for i in range(batch_size):
                if self.progress_callback:
                    progress = 75 + int((i + 1) * 25 / batch_size)  # 75-100%
                    self.progress_callback(progress)
                    
                slice_result = self.deblur_core_torch(
                    data_tensor[i], kernel_tensor, iterations, rule
                )
                results.append(slice_result.unsqueeze(0))
            
            result = torch.cat(results, dim=0)
        else:
            raise ValueError(f"不支持的数据维度: {data_tensor.ndim}")
        
        return result
    
    def iterative_deconv_3d(self, data_3d: Union[np.ndarray, torch.Tensor], 
                           kernel_3d: Union[np.ndarray, torch.Tensor],
                           iterations: int = 10, rule: int = 1) -> torch.Tensor:
        """
        3D迭代去卷积（处理整个3D体积）
        
        参数:
            data_3d: 3D输入数据
            kernel_3d: 3D卷积核
            iterations: 迭代次数
            rule: 算法规则
            
        返回:
            去卷积后的3D数据
        """
        # 准备输入数据
        data_tensor = self._prepare_input(data_3d)
        kernel_tensor = self._prepare_kernel(kernel_3d)
        
        if data_tensor.ndim != 3:
            raise ValueError(f"需要3D数据，但得到{data_tensor.ndim}D")
        
        # 获取尺寸
        depth, height, width = data_tensor.shape
        
        # 计算边界填充尺寸
        B = min(height, width) // 6
        
        # 3D边缘填充（对每个深度切片单独填充）
        padded_slices = []
        for i in range(depth):
            slice_data = data_tensor[i]
            padded_slice = self._pad_data(slice_data, B)
            padded_slices.append(padded_slice.unsqueeze(0))
        
        padded_data = torch.cat(padded_slices, dim=0)
        
        # 计算3D OTF
        otf = self.psf2otf_torch(kernel_tensor, padded_data.shape)
        
        # 3D RL或Landweber去卷积
        if rule == 1:  # Richardson-Lucy 3D
            result = self._richardson_lucy_3d_torch(padded_data, otf, iterations)
        elif rule == 2:  # Landweber 3D
            result = self._landweber_3d_torch(padded_data, otf, iterations)
        else:
            raise ValueError(f"不支持的算法规则: {rule}")
        
        # 裁剪填充区域
        result_cropped = result[:, B:-B, B:-B]
        
        return result_cropped
    
    def _richardson_lucy_3d_torch(self, data: torch.Tensor, otf: torch.Tensor,
                                 iterations: int = 10) -> torch.Tensor:
        """
        3D Richardson-Lucy去卷积
        """
        # 初始化
        yk = data.clone()
        xk = torch.zeros_like(data)
        
        # 预计算
        ones_tensor = torch.ones_like(data)
        otf_conj = torch.conj(otf)
        
        for iter_idx in range(iterations):
            xk_prev = xk
            
            # RL迭代 (3D)
            fft_yk = torch.fft.fftn(yk, dim=(-3, -2, -1))
            conv_result = torch.fft.ifftn(otf * fft_yk, dim=(-3, -2, -1)).real
            conv_result = torch.maximum(conv_result, torch.tensor(1e-6, device=self.device))
            
            ratio = data / conv_result
            fft_ratio = torch.fft.fftn(ratio, dim=(-3, -2, -1))
            numerator = torch.fft.ifftn(otf_conj * fft_ratio, dim=(-3, -2, -1)).real
            
            denominator = torch.fft.ifftn(
                torch.fft.fftn(ones_tensor, dim=(-3, -2, -1)) * otf,
                dim=(-3, -2, -1)
            ).real
            
            xk = yk * numerator / torch.maximum(denominator, torch.tensor(1e-6, device=self.device))
            xk = torch.maximum(xk, torch.tensor(1e-6, device=self.device))
            
            # 更新yk（简单更新，无加速）
            yk = xk.clone()
            
            # 清理内存
            if (iter_idx + 1) % 5 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return torch.maximum(xk, torch.tensor(0, device=self.device))
    
    def _landweber_3d_torch(self, data: torch.Tensor, otf: torch.Tensor,
                           iterations: int = 10) -> torch.Tensor:
        """
        3D Landweber去卷积
        """
        # 初始化
        yk = data.clone()
        xk = data.clone()
        
        # 预计算
        otf_conj = torch.conj(otf)
        t = 1.0  # 步长
        
        for iter_idx in range(iterations):
            # Landweber迭代 (3D)
            fft_yk = torch.fft.fftn(yk, dim=(-3, -2, -1))
            fft_data = torch.fft.fftn(data, dim=(-3, -2, -1))
            
            update_term = torch.fft.ifftn(
                otf_conj * (fft_data - otf * fft_yk),
                dim=(-3, -2, -1)
            ).real
            
            xk = yk + t * update_term
            xk = torch.maximum(xk, torch.tensor(1e-6, device=self.device))
            
            # 简单更新（无加速）
            yk = xk.clone()
            
            # 清理内存
            if (iter_idx + 1) % 5 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return torch.maximum(xk, torch.tensor(0, device=self.device))
    
    def iterative_deconv(self, data: Union[np.ndarray, torch.Tensor],
                        kernel: Union[np.ndarray, torch.Tensor],
                        iteration: int, rule: int) -> np.ndarray:
        """
        兼容性函数，保持与原函数相同的接口
        
        参数:
            data: 输入数据
            kernel: 卷积核
            iteration: 迭代次数
            rule: 算法规则 (1: RL, 2: Landweber)
            
        返回:
            去卷积后的数据 (NumPy数组)
        """
        # 执行去卷积
        result_tensor = self.iterative_deconv_batch(data, kernel, iteration, rule)
        
        # 转换为NumPy数组
        return result_tensor.cpu().numpy()
    
    def benchmark(self, data_shape: Tuple[int, ...] = (256, 256),
                 kernel_shape: Tuple[int, ...] = (32, 32),
                 iterations: int = 20, rule: int = 1,
                 num_runs: int = 5) -> dict:
        """
        性能基准测试
        
        参数:
            data_shape: 数据形状
            kernel_shape: 卷积核形状
            iterations: 迭代次数
            rule: 算法规则
            num_runs: 运行次数
            
        返回:
            性能统计
        """
        import time
        
        print(f"\n{'='*60}")
        print("迭代去卷积性能基准测试")
        print(f"{'='*60}")
        print(f"数据形状: {data_shape}")
        print(f"卷积核形状: {kernel_shape}")
        print(f"迭代次数: {iterations}")
        print(f"算法规则: {'Richardson-Lucy' if rule == 1 else 'Landweber'}")
        print(f"设备: {self.device}")
        print(f"{'='*60}")
        
        # 生成测试数据
        torch.manual_seed(42)
        data = torch.randn(*data_shape, dtype=self.dtype, device=self.device).abs()
        kernel = torch.randn(*kernel_shape, dtype=self.dtype, device=self.device).abs()
        kernel = kernel / torch.sum(kernel)
        
        # 预热
        print("预热...")
        for _ in range(3):
            _ = self.iterative_deconv_batch(data, kernel, 5, rule)
        
        # 计时
        times = []
        for i in range(num_runs):
            start_time = time.time()
            result = self.iterative_deconv_batch(data, kernel, iterations, rule)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            print(f"运行 {i+1}/{num_runs}: {times[-1]:.3f}秒")
        
        # 统计
        stats = {
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'iterations_per_sec': iterations / np.mean(times),
            'total_iterations': iterations * num_runs
        }
        
        print(f"\n性能统计:")
        print(f"  平均时间: {stats['avg_time']:.3f}秒")
        print(f"  最慢时间: {stats['max_time']:.3f}秒")
        print(f"  最快时间: {stats['min_time']:.3f}秒")
        print(f"  标准差: {stats['std_time']:.3f}秒")
        print(f"  迭代速度: {stats['iterations_per_sec']:.2f} 迭代/秒")
        
        if self.device.type == 'cuda':
            memory_used = torch.cuda.max_memory_allocated() / 1024**2
            memory_cached = torch.cuda.max_memory_reserved() / 1024**2
            print(f"  最大GPU内存使用: {memory_used:.2f} MB")
            print(f"  最大GPU内存缓存: {memory_cached:.2f} MB")
        
        return stats

class Sparse_deconvolution_FM:
    """
    迭代去卷积 - 使用PyTorch加速
    支持多种去卷积算法和GPU加速
    """
    
    def __init__(self, args, progress_callback=None, device=None, dtype=torch.float32):
        
        self.dtype = dtype
        
        self.device = torch.device(args.device)
        self.progress_callback = progress_callback
        self._stop_requested = False

        # 预分配缓冲区
        self._buffers = {}
        self._cache = {}
        self.pixel_size = args.Raw_pixel_size
        self.wavelength = args.Emission_wavelength
        self.Sparse_NA = args.Sparse_NA
        self.iteration_num = args.Sparse_iteration_number
        # self.filedir = args.filedir
        # self.fname = args.fname
        # self.ext = args.ext
        self.Sparse_offset = args.Sparse_offset
        # self.TDV = torch.from_numpy((args.tdv_result).astype(np.float64)).to(args.device)
        self.TDV = args.tdv_result.astype(np.float32)
        self.Rolling_ball_radius = args.Rolling_ball_radius
        self.Rolling_ball_paraboloid_flag = args.Rolling_ball_paraboloid_flag
        
        # ImageLib.read(os.path.join(self.filedir,
        #                    self.fname + '_3_TDV'\
        #                    + self.ext), to_tensor=True) + args.Sparse_offset
    
    def process_tdv_background(self):
        """
        处理TDV三维矩阵的背景减除
        self.TDV是一个三维矩阵，第0维是帧数，第1-2维是图像的大小
        subtract_background_fixed只能输入二维图像
        """
        if not hasattr(self, 'TDV') or self.TDV is None:
            return None
        
        # 获取TDV的维度
        num_frames = self.TDV.shape[0]
        
        # 初始化结果数组
        tdv_processed = np.zeros_like(self.TDV)
        
        # 对每一帧进行背景减除
        for i in range(num_frames):
            # 获取当前帧
            frame = self.TDV[i, :, :]
            
            # 对当前帧进行背景减除
            background = subtract_background_fixed(
                frame,
                radius=self.Rolling_ball_radius,
                light_background=False,
                create_background=True,
                use_paraboloid=self.Rolling_ball_paraboloid_flag,
                do_presmooth=True,
                correct_corners=True
            )
            
            # 假设subtract_background_fixed返回的是减去背景后的图像
            # 如果返回的是背景，需要frame - background
            # 这里根据实际情况调整
            tdv_processed[i, :, :] = background
        
        return tdv_processed

    def recon(self):  
        # a = time.time()
        background = self.process_tdv_background()
        # b = time.time()
        # print(b-a)
        self.TDV = self.TDV - background
        self.Sparse_factor = self.TDV.max()
        generator = PSFGenerator(device=self.device) 
        kernel =  generator.generate_psf_torch_vectorized(self.pixel_size, \
            self.wavelength, 64, self.Sparse_NA, 0)
        Tmp = self.TDV / self.Sparse_factor + self.Sparse_offset
        im_sparse = self.iterative_deconv_batch(Tmp, kernel, self.iteration_num)
        im_sparse = (im_sparse - self.Sparse_offset) * self.Sparse_factor/2
        # ImageLib.write(os.path.join(self.filedir,
        #                    self.fname + '_4_Hybrid'\
        #                    + self.ext), im_sparse.cpu().numpy(), writemode='w') 
        # print(">> Finish Hybrid deconvolution.") 
        if self._stop_requested:
            # 清理内存
            self.cleanup()
            return None

        return background, im_sparse.cpu().numpy()
        
    def stop(self):
        """停止重建"""
        self._stop_requested = True
    
    def cleanup(self):
        """清理内存"""
        try:
            # 清理PyTorch GPU缓存
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 清理Python变量
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'optimizer'):
                del self.optimizer
            
            # 垃圾回收
            import gc
            gc.collect()
        except:
            pass

    def _prepare_input(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """准备输入数据"""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data.astype(np.float32)).to(self.device)
        elif isinstance(data, torch.Tensor):
            return data.to(self.device, dtype=self.dtype)
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")
    
    def _prepare_kernel(self, kernel: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """准备卷积核"""
        kernel_tensor = self._prepare_input(kernel)
        # 归一化
        kernel_tensor = kernel_tensor / torch.sum(kernel_tensor)
        return kernel_tensor
    
    def psf2otf_torch(self, psf: torch.Tensor, out_size: Tuple[int, ...]) -> torch.Tensor:
        """
        PyTorch版本的PSF转OTF
        将点扩散函数转换为光学传递函数
        """
        psf_size = torch.tensor(psf.shape, device=self.device)
        out_size_tensor = torch.tensor(out_size, device=self.device)
        
        # 计算填充尺寸
        pad_size = out_size_tensor - psf_size
        
        # 填充PSF
        psf_padded = torch.nn.functional.pad(
            psf, 
            (0, int(pad_size[1].item()), 0, int(pad_size[0].item())),
            mode='constant',
            value=0
        )
        
        # 循环移位（将中心移到0,0）
        for i, dim_size in enumerate(psf_size):
            shift = -int(dim_size.item() // 2)
            psf_padded = torch.roll(psf_padded, shifts=shift, dims=i)
        
        # 计算FFT
        otf = torch.fft.fftn(psf_padded, dim=(-2, -1))
        
        # 如果虚部很小，转换为实数
        if torch.max(torch.abs(otf.imag)) / torch.max(torch.abs(otf)) < 1e-6:
            otf = otf.real
        
        return otf
    
    def _pad_data(self, data: torch.Tensor, padding: int) -> torch.Tensor:
        """
        通用数据填充函数
        修复反射填充模式的问题
        """
        if padding <= 0:
            return data
        
        # 根据数据维度选择合适的填充策略
        if data.ndim == 2:
            # 2D数据：使用复制填充
            return F.pad(data.unsqueeze(0).unsqueeze(0), 
                        (padding, padding, padding, padding), 
                        mode='replicate').squeeze(0).squeeze(0)
        elif data.ndim == 3:
            # 3D数据：对每个通道单独填充
            batch_size = data.shape[0]
            padded_slices = []
            
            for i in range(batch_size):
                padded_slice = F.pad(data[i].unsqueeze(0).unsqueeze(0),
                                   (padding, padding, padding, padding),
                                   mode='replicate').squeeze(0).squeeze(0)
                padded_slices.append(padded_slice.unsqueeze(0))
            
            return torch.cat(padded_slices, dim=0)
        else:
            # 高维数据：使用常数填充
            return F.pad(data, 
                        (padding, padding, padding, padding), 
                        mode='constant', 
                        value=data.min().item())
    
    def richardson_lucy_torch(self, data: torch.Tensor, otf: torch.Tensor, 
                             iterations: int = 10, acceleration: bool = True) -> torch.Tensor:
        """
        Richardson-Lucy去卷积（PyTorch加速版）
        
        参数:
            data: 输入图像
            otf: 光学传递函数
            iterations: 迭代次数
            acceleration: 是否使用加速
            
        返回:
            去卷积后的图像
        """

        # 将输入数据中所有小于0的数置为0
        data = torch.clamp(data, min=0.0)

        # 初始化变量
        yk = data.clone()
        xk = torch.zeros_like(data)
        vk = torch.zeros_like(data)
        
        # 预计算一些常量
        ones_tensor = torch.ones_like(data)
        otf_conj = torch.conj(otf)
        
        for iter_idx in range(iterations):
            if self._stop_requested:
                break
            # if self.progress_callback:
            #     progress = 75 + int((iter_idx + 1) * 25 / iterations)  # 75-100%
            #     self.progress_callback(progress)
            # # 保存当前xk
            xk_prev = xk
            
            # RL迭代
            # 计算: xk = yk * IFFT(conj(OTF) * FFT(data / IFFT(OTF * FFT(yk))))
            fft_yk = torch.fft.fftn(yk, dim=(-2, -1))
            conv_result = torch.fft.ifftn(otf * fft_yk, dim=(-2, -1)).real
            conv_result = torch.maximum(conv_result, torch.tensor(1e-6, device=self.device))
            
            ratio = data / conv_result
            fft_ratio = torch.fft.fftn(ratio, dim=(-2, -1))
            numerator = torch.fft.ifftn(otf_conj * fft_ratio, dim=(-2, -1)).real
            
            # 分母
            denominator = torch.fft.ifftn(
                torch.fft.fftn(ones_tensor, dim=(-2, -1)) * otf, 
                dim=(-2, -1)
            ).real
            
            xk = yk * numerator / torch.maximum(denominator, torch.tensor(1e-6, device=self.device))
            xk = torch.maximum(xk, torch.tensor(1e-6, device=self.device))
            
            # 更新vk
            vk_prev = vk
            vk = torch.maximum(xk - yk, torch.tensor(1e-6, device=self.device))
            
            if not acceleration or iter_idx == 0:
                # 不使用加速
                alpha = 0
                yk = xk.clone()
            else:
                # 计算加速参数
                vk_prev_flat = vk_prev.flatten()
                vk_flat = vk.flatten()
                
                numerator_acc = torch.sum(vk_prev_flat * vk_flat)
                denominator_acc = torch.sum(vk_prev_flat * vk_prev_flat) + 1e-10
                alpha = torch.clamp(numerator_acc / denominator_acc, min=1e-6, max=1.0)
                
                # 加速更新
                yk = xk + alpha * (xk - xk_prev)
                yk = torch.maximum(yk, torch.tensor(1e-6, device=self.device))
            
            # 处理NaN
            yk = torch.nan_to_num(yk, nan=1e-6)
            
            # 每10次迭代清理内存
            if (iter_idx + 1) % 10 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # 确保非负
        yk = torch.maximum(yk, torch.tensor(0, device=self.device))
        
        return yk
    
    def landweber_torch(self, data: torch.Tensor, otf: torch.Tensor, 
                        iterations: int = 10, step_size: float = 1.0) -> torch.Tensor:
        """
        Landweber去卷积（PyTorch加速版）
        
        参数:
            data: 输入图像
            otf: 光学传递函数
            iterations: 迭代次数
            step_size: 步长参数
            
        返回:
            去卷积后的图像
        """
        # 初始化变量
        yk = data.clone()
        xk = torch.zeros_like(data)
        xk_prev = torch.zeros_like(data)
        
        # 预计算
        otf_conj = torch.conj(otf)
        t = step_size
        
        # 初始gamma值
        gamma1 = 1.0
        
        for iter_idx in range(iterations):
            if iter_idx == 0:
                # 第一次迭代
                fft_data = torch.fft.fftn(data, dim=(-2, -1))
                update_term = torch.fft.ifftn(
                    otf_conj * (fft_data - otf * fft_data), 
                    dim=(-2, -1)
                ).real
                xk = data + t * update_term
            else:
                # 计算gamma2和beta
                gamma2 = 0.5 * torch.sqrt(4 * gamma1**2 + gamma1**4) - gamma1**2
                beta = -gamma2 * (1 - 1 / gamma1)
                
                # 更新yk
                yk_update = xk + beta * (xk - xk_prev)
                
                # Landweber迭代
                fft_yk_update = torch.fft.fftn(yk_update, dim=(-2, -1))
                fft_data = torch.fft.fftn(data, dim=(-2, -1))
                
                update_term = torch.fft.ifftn(
                    otf_conj * (fft_data - otf * fft_yk_update), 
                    dim=(-2, -1)
                ).real
                
                yk = yk_update + t * update_term
                yk = torch.maximum(yk, torch.tensor(1e-6, device=self.device))
                
                # 更新gamma
                gamma1 = gamma2
                xk_prev = xk.clone()
                xk = yk.clone()
            
            # 每10次迭代清理内存
            if (iter_idx + 1) % 10 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # 确保非负
        xk = torch.maximum(xk, torch.tensor(0, device=self.device))
        
        return xk
    
    def deblur_core_torch(self, data: torch.Tensor, kernel: torch.Tensor, 
                         iterations: int = 10, rule: int = 1) -> torch.Tensor:
        """
        核心去卷积函数（PyTorch加速版）
        
        参数:
            data: 输入图像
            kernel: 卷积核
            iterations: 迭代次数
            rule: 算法规则 (1: Richardson-Lucy, 2: Landweber)
            
        返回:
            去卷积后的图像
        """
        # 获取输入尺寸
        dx, dy = data.shape[-2], data.shape[-1]
        
        # 计算边界填充尺寸
        B = min(dx, dy) // 6
        
        # 边缘填充（使用修复后的填充函数）
        padded_data = self._pad_data(data, B)
        
        # 计算OTF
        otf = self.psf2otf_torch(kernel, padded_data.shape)
        
        # 根据规则选择算法
        if rule == 1:  # Richardson-Lucy
            result = self.richardson_lucy_torch(padded_data, otf, iterations, acceleration=True)
        elif rule == 2:  # Landweber
            result = self.landweber_torch(padded_data, otf, iterations, step_size=1.0)
        else:
            raise ValueError(f"不支持的算法规则: {rule}")
        
        # 裁剪填充区域
        if result.ndim == 2:
            result_cropped = result[B:-B, B:-B]
        else:
            result_cropped = result[:, B:-B, B:-B]
        
        return result_cropped
    
    def iterative_deconv_batch(self, data: Union[np.ndarray, torch.Tensor], 
                              kernel: Union[np.ndarray, torch.Tensor],
                              iterations: int = 10, rule: int = 1) -> torch.Tensor:
        """
        批量迭代去卷积（支持2D和3D数据）
        
        参数:
            data: 输入数据，可以是2D或3D
            kernel: 卷积核
            iterations: 迭代次数
            rule: 算法规则
            
        返回:
            去卷积后的数据
        """
        # 准备输入数据
        data_tensor = self._prepare_input(data)
        kernel_tensor = self._prepare_kernel(kernel)
        
        if data_tensor.ndim == 2:
            # 2D数据
            result = self.deblur_core_torch(data_tensor, kernel_tensor, iterations, rule)
        elif data_tensor.ndim == 3:
            # 3D数据 - 逐片处理
            batch_size = data_tensor.shape[0]
            results = []
            
            for i in range(batch_size):
                if self.progress_callback:
                    progress = 66 + int((i + 1) * 34 / batch_size)  # 66-100%
                    self.progress_callback(progress)
                    
                slice_result = self.deblur_core_torch(
                    data_tensor[i], kernel_tensor, iterations, rule
                )
                results.append(slice_result.unsqueeze(0))
            
            result = torch.cat(results, dim=0)
        else:
            raise ValueError(f"不支持的数据维度: {data_tensor.ndim}")
        
        return result
    
    def iterative_deconv_3d(self, data_3d: Union[np.ndarray, torch.Tensor], 
                           kernel_3d: Union[np.ndarray, torch.Tensor],
                           iterations: int = 10, rule: int = 1) -> torch.Tensor:
        """
        3D迭代去卷积（处理整个3D体积）
        
        参数:
            data_3d: 3D输入数据
            kernel_3d: 3D卷积核
            iterations: 迭代次数
            rule: 算法规则
            
        返回:
            去卷积后的3D数据
        """
        # 准备输入数据
        data_tensor = self._prepare_input(data_3d)
        kernel_tensor = self._prepare_kernel(kernel_3d)
        
        if data_tensor.ndim != 3:
            raise ValueError(f"需要3D数据，但得到{data_tensor.ndim}D")
        
        # 获取尺寸
        depth, height, width = data_tensor.shape
        
        # 计算边界填充尺寸
        B = min(height, width) // 6
        
        # 3D边缘填充（对每个深度切片单独填充）
        padded_slices = []
        for i in range(depth):
            slice_data = data_tensor[i]
            padded_slice = self._pad_data(slice_data, B)
            padded_slices.append(padded_slice.unsqueeze(0))
        
        padded_data = torch.cat(padded_slices, dim=0)
        
        # 计算3D OTF
        otf = self.psf2otf_torch(kernel_tensor, padded_data.shape)
        
        # 3D RL或Landweber去卷积
        if rule == 1:  # Richardson-Lucy 3D
            result = self._richardson_lucy_3d_torch(padded_data, otf, iterations)
        elif rule == 2:  # Landweber 3D
            result = self._landweber_3d_torch(padded_data, otf, iterations)
        else:
            raise ValueError(f"不支持的算法规则: {rule}")
        
        # 裁剪填充区域
        result_cropped = result[:, B:-B, B:-B]
        
        return result_cropped
    
    def _richardson_lucy_3d_torch(self, data: torch.Tensor, otf: torch.Tensor,
                                 iterations: int = 10) -> torch.Tensor:
        """
        3D Richardson-Lucy去卷积
        """
        # 初始化
        yk = data.clone()
        xk = torch.zeros_like(data)
        
        # 预计算
        ones_tensor = torch.ones_like(data)
        otf_conj = torch.conj(otf)
        
        for iter_idx in range(iterations):
            xk_prev = xk
            
            # RL迭代 (3D)
            fft_yk = torch.fft.fftn(yk, dim=(-3, -2, -1))
            conv_result = torch.fft.ifftn(otf * fft_yk, dim=(-3, -2, -1)).real
            conv_result = torch.maximum(conv_result, torch.tensor(1e-6, device=self.device))
            
            ratio = data / conv_result
            fft_ratio = torch.fft.fftn(ratio, dim=(-3, -2, -1))
            numerator = torch.fft.ifftn(otf_conj * fft_ratio, dim=(-3, -2, -1)).real
            
            denominator = torch.fft.ifftn(
                torch.fft.fftn(ones_tensor, dim=(-3, -2, -1)) * otf,
                dim=(-3, -2, -1)
            ).real
            
            xk = yk * numerator / torch.maximum(denominator, torch.tensor(1e-6, device=self.device))
            xk = torch.maximum(xk, torch.tensor(1e-6, device=self.device))
            
            # 更新yk（简单更新，无加速）
            yk = xk.clone()
            
            # 清理内存
            if (iter_idx + 1) % 5 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return torch.maximum(xk, torch.tensor(0, device=self.device))
    
    def _landweber_3d_torch(self, data: torch.Tensor, otf: torch.Tensor,
                           iterations: int = 10) -> torch.Tensor:
        """
        3D Landweber去卷积
        """
        # 初始化
        yk = data.clone()
        xk = data.clone()
        
        # 预计算
        otf_conj = torch.conj(otf)
        t = 1.0  # 步长
        
        for iter_idx in range(iterations):
            # Landweber迭代 (3D)
            fft_yk = torch.fft.fftn(yk, dim=(-3, -2, -1))
            fft_data = torch.fft.fftn(data, dim=(-3, -2, -1))
            
            update_term = torch.fft.ifftn(
                otf_conj * (fft_data - otf * fft_yk),
                dim=(-3, -2, -1)
            ).real
            
            xk = yk + t * update_term
            xk = torch.maximum(xk, torch.tensor(1e-6, device=self.device))
            
            # 简单更新（无加速）
            yk = xk.clone()
            
            # 清理内存
            if (iter_idx + 1) % 5 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return torch.maximum(xk, torch.tensor(0, device=self.device))
    
    def iterative_deconv(self, data: Union[np.ndarray, torch.Tensor],
                        kernel: Union[np.ndarray, torch.Tensor],
                        iteration: int, rule: int) -> np.ndarray:
        """
        兼容性函数，保持与原函数相同的接口
        
        参数:
            data: 输入数据
            kernel: 卷积核
            iteration: 迭代次数
            rule: 算法规则 (1: RL, 2: Landweber)
            
        返回:
            去卷积后的数据 (NumPy数组)
        """
        # 执行去卷积
        result_tensor = self.iterative_deconv_batch(data, kernel, iteration, rule)
        
        # 转换为NumPy数组
        return result_tensor.cpu().numpy()
    
    def benchmark(self, data_shape: Tuple[int, ...] = (256, 256),
                 kernel_shape: Tuple[int, ...] = (32, 32),
                 iterations: int = 20, rule: int = 1,
                 num_runs: int = 5) -> dict:
        """
        性能基准测试
        
        参数:
            data_shape: 数据形状
            kernel_shape: 卷积核形状
            iterations: 迭代次数
            rule: 算法规则
            num_runs: 运行次数
            
        返回:
            性能统计
        """
        import time
        
        print(f"\n{'='*60}")
        print("迭代去卷积性能基准测试")
        print(f"{'='*60}")
        print(f"数据形状: {data_shape}")
        print(f"卷积核形状: {kernel_shape}")
        print(f"迭代次数: {iterations}")
        print(f"算法规则: {'Richardson-Lucy' if rule == 1 else 'Landweber'}")
        print(f"设备: {self.device}")
        print(f"{'='*60}")
        
        # 生成测试数据
        torch.manual_seed(42)
        data = torch.randn(*data_shape, dtype=self.dtype, device=self.device).abs()
        kernel = torch.randn(*kernel_shape, dtype=self.dtype, device=self.device).abs()
        kernel = kernel / torch.sum(kernel)
        
        # 预热
        print("预热...")
        for _ in range(3):
            _ = self.iterative_deconv_batch(data, kernel, 5, rule)
        
        # 计时
        times = []
        for i in range(num_runs):
            start_time = time.time()
            result = self.iterative_deconv_batch(data, kernel, iterations, rule)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            print(f"运行 {i+1}/{num_runs}: {times[-1]:.3f}秒")
        
        # 统计
        stats = {
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'iterations_per_sec': iterations / np.mean(times),
            'total_iterations': iterations * num_runs
        }
        
        print(f"\n性能统计:")
        print(f"  平均时间: {stats['avg_time']:.3f}秒")
        print(f"  最慢时间: {stats['max_time']:.3f}秒")
        print(f"  最快时间: {stats['min_time']:.3f}秒")
        print(f"  标准差: {stats['std_time']:.3f}秒")
        print(f"  迭代速度: {stats['iterations_per_sec']:.2f} 迭代/秒")
        
        if self.device.type == 'cuda':
            memory_used = torch.cuda.max_memory_allocated() / 1024**2
            memory_cached = torch.cuda.max_memory_reserved() / 1024**2
            print(f"  最大GPU内存使用: {memory_used:.2f} MB")
            print(f"  最大GPU内存缓存: {memory_cached:.2f} MB")
        
        return stats

class Sparse_deconvolution_WFM:
    """
    迭代去卷积 - 使用PyTorch加速
    支持多种去卷积算法和GPU加速
    """
    
    def __init__(self, args, progress_callback=None, device=None, dtype=torch.float32):
        
        self.dtype = dtype
        
        self.device = torch.device(args.device)
        self.progress_callback = progress_callback
        self._stop_requested = False

        # 预分配缓冲区
        self._buffers = {}
        self._cache = {}
        self.pixel_size = args.FM_Raw_pixel_size
        self.wavelength = args.FM_emission_wavelength
        self.Sparse_NA = args.Sparse_NA
        self.iteration_num = args.Sparse_iteration_number
        # self.filedir = args.filedir
        # self.fname = args.fname
        # self.ext = args.ext
        self.Sparse_offset = args.Sparse_offset
        self.TDV = torch.from_numpy((args.tdv_result).astype(np.float32)).to(args.device)
        self.Sparse_factor = self.TDV.max()
        # ImageLib.read(os.path.join(self.filedir,
        #                    self.fname + '_3_TDV'\
        #                    + self.ext), to_tensor=True) + args.Sparse_offset
        
    def recon(self):  
        generator = PSFGenerator(device=self.device) 
        kernel =  generator.generate_psf_torch_vectorized(self.pixel_size, \
            self.wavelength, 64, self.Sparse_NA, 0)
        Tmp = self.TDV / self.Sparse_factor + self.Sparse_offset
        im_sparse = self.iterative_deconv_batch(Tmp, kernel, self.iteration_num)
        im_sparse = (im_sparse - self.Sparse_offset) * self.Sparse_factor/2
        # ImageLib.write(os.path.join(self.filedir,
        #                    self.fname + '_4_Hybrid'\
        #                    + self.ext), im_sparse.cpu().numpy(), writemode='w') 
        # print(">> Finish Hybrid deconvolution.") 
        if self._stop_requested:
            # 清理内存
            self.cleanup()
            return None
        return im_sparse.cpu().numpy()

    def stop(self):
        """停止重建"""
        self._stop_requested = True
    
    def cleanup(self):
        """清理内存"""
        try:
            # 清理PyTorch GPU缓存
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 清理Python变量
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'optimizer'):
                del self.optimizer
            
            # 垃圾回收
            import gc
            gc.collect()
        except:
            pass

    def _prepare_input(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """准备输入数据"""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data.astype(np.float32)).to(self.device)
        elif isinstance(data, torch.Tensor):
            return data.to(self.device, dtype=self.dtype)
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")
    
    def _prepare_kernel(self, kernel: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """准备卷积核"""
        kernel_tensor = self._prepare_input(kernel)
        # 归一化
        kernel_tensor = kernel_tensor / torch.sum(kernel_tensor)
        return kernel_tensor
    
    def psf2otf_torch(self, psf: torch.Tensor, out_size: Tuple[int, ...]) -> torch.Tensor:
        """
        PyTorch版本的PSF转OTF
        将点扩散函数转换为光学传递函数
        """
        psf_size = torch.tensor(psf.shape, device=self.device)
        out_size_tensor = torch.tensor(out_size, device=self.device)
        
        # 计算填充尺寸
        pad_size = out_size_tensor - psf_size
        
        # 填充PSF
        psf_padded = torch.nn.functional.pad(
            psf, 
            (0, int(pad_size[1].item()), 0, int(pad_size[0].item())),
            mode='constant',
            value=0
        )
        
        # 循环移位（将中心移到0,0）
        for i, dim_size in enumerate(psf_size):
            shift = -int(dim_size.item() // 2)
            psf_padded = torch.roll(psf_padded, shifts=shift, dims=i)
        
        # 计算FFT
        otf = torch.fft.fftn(psf_padded, dim=(-2, -1))
        
        # 如果虚部很小，转换为实数
        if torch.max(torch.abs(otf.imag)) / torch.max(torch.abs(otf)) < 1e-6:
            otf = otf.real
        
        return otf
    
    def _pad_data(self, data: torch.Tensor, padding: int) -> torch.Tensor:
        """
        通用数据填充函数
        修复反射填充模式的问题
        """
        if padding <= 0:
            return data
        
        # 根据数据维度选择合适的填充策略
        if data.ndim == 2:
            # 2D数据：使用复制填充
            return F.pad(data.unsqueeze(0).unsqueeze(0), 
                        (padding, padding, padding, padding), 
                        mode='replicate').squeeze(0).squeeze(0)
        elif data.ndim == 3:
            # 3D数据：对每个通道单独填充
            batch_size = data.shape[0]
            padded_slices = []
            
            for i in range(batch_size):
                padded_slice = F.pad(data[i].unsqueeze(0).unsqueeze(0),
                                   (padding, padding, padding, padding),
                                   mode='replicate').squeeze(0).squeeze(0)
                padded_slices.append(padded_slice.unsqueeze(0))
            
            return torch.cat(padded_slices, dim=0)
        else:
            # 高维数据：使用常数填充
            return F.pad(data, 
                        (padding, padding, padding, padding), 
                        mode='constant', 
                        value=data.min().item())
    
    def richardson_lucy_torch(self, data: torch.Tensor, otf: torch.Tensor, 
                             iterations: int = 10, acceleration: bool = True) -> torch.Tensor:
        """
        Richardson-Lucy去卷积（PyTorch加速版）
        
        参数:
            data: 输入图像
            otf: 光学传递函数
            iterations: 迭代次数
            acceleration: 是否使用加速
            
        返回:
            去卷积后的图像
        """

        # 将输入数据中所有小于0的数置为0
        data = torch.clamp(data, min=0.0)

        # 初始化变量
        yk = data.clone()
        xk = torch.zeros_like(data)
        vk = torch.zeros_like(data)
        
        # 预计算一些常量
        ones_tensor = torch.ones_like(data)
        otf_conj = torch.conj(otf)
        
        for iter_idx in range(iterations):
            if self._stop_requested:
                break
            # if self.progress_callback:
            #     progress = 75 + int((iter_idx + 1) * 25 / iterations)  # 75-100%
            #     self.progress_callback(progress)
            # # 保存当前xk
            xk_prev = xk
            
            # RL迭代
            # 计算: xk = yk * IFFT(conj(OTF) * FFT(data / IFFT(OTF * FFT(yk))))
            fft_yk = torch.fft.fftn(yk, dim=(-2, -1))
            conv_result = torch.fft.ifftn(otf * fft_yk, dim=(-2, -1)).real
            conv_result = torch.maximum(conv_result, torch.tensor(1e-6, device=self.device))
            
            ratio = data / conv_result
            fft_ratio = torch.fft.fftn(ratio, dim=(-2, -1))
            numerator = torch.fft.ifftn(otf_conj * fft_ratio, dim=(-2, -1)).real
            
            # 分母
            denominator = torch.fft.ifftn(
                torch.fft.fftn(ones_tensor, dim=(-2, -1)) * otf, 
                dim=(-2, -1)
            ).real
            
            xk = yk * numerator / torch.maximum(denominator, torch.tensor(1e-6, device=self.device))
            xk = torch.maximum(xk, torch.tensor(1e-6, device=self.device))
            
            # 更新vk
            vk_prev = vk
            vk = torch.maximum(xk - yk, torch.tensor(1e-6, device=self.device))
            
            if not acceleration or iter_idx == 0:
                # 不使用加速
                alpha = 0
                yk = xk.clone()
            else:
                # 计算加速参数
                vk_prev_flat = vk_prev.flatten()
                vk_flat = vk.flatten()
                
                numerator_acc = torch.sum(vk_prev_flat * vk_flat)
                denominator_acc = torch.sum(vk_prev_flat * vk_prev_flat) + 1e-10
                alpha = torch.clamp(numerator_acc / denominator_acc, min=1e-6, max=1.0)
                
                # 加速更新
                yk = xk + alpha * (xk - xk_prev)
                yk = torch.maximum(yk, torch.tensor(1e-6, device=self.device))
            
            # 处理NaN
            yk = torch.nan_to_num(yk, nan=1e-6)
            
            # 每10次迭代清理内存
            if (iter_idx + 1) % 10 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # 确保非负
        yk = torch.maximum(yk, torch.tensor(0, device=self.device))
        
        return yk
    
    def landweber_torch(self, data: torch.Tensor, otf: torch.Tensor, 
                        iterations: int = 10, step_size: float = 1.0) -> torch.Tensor:
        """
        Landweber去卷积（PyTorch加速版）
        
        参数:
            data: 输入图像
            otf: 光学传递函数
            iterations: 迭代次数
            step_size: 步长参数
            
        返回:
            去卷积后的图像
        """
        # 初始化变量
        yk = data.clone()
        xk = torch.zeros_like(data)
        xk_prev = torch.zeros_like(data)
        
        # 预计算
        otf_conj = torch.conj(otf)
        t = step_size
        
        # 初始gamma值
        gamma1 = 1.0
        
        for iter_idx in range(iterations):
            if iter_idx == 0:
                # 第一次迭代
                fft_data = torch.fft.fftn(data, dim=(-2, -1))
                update_term = torch.fft.ifftn(
                    otf_conj * (fft_data - otf * fft_data), 
                    dim=(-2, -1)
                ).real
                xk = data + t * update_term
            else:
                # 计算gamma2和beta
                gamma2 = 0.5 * torch.sqrt(4 * gamma1**2 + gamma1**4) - gamma1**2
                beta = -gamma2 * (1 - 1 / gamma1)
                
                # 更新yk
                yk_update = xk + beta * (xk - xk_prev)
                
                # Landweber迭代
                fft_yk_update = torch.fft.fftn(yk_update, dim=(-2, -1))
                fft_data = torch.fft.fftn(data, dim=(-2, -1))
                
                update_term = torch.fft.ifftn(
                    otf_conj * (fft_data - otf * fft_yk_update), 
                    dim=(-2, -1)
                ).real
                
                yk = yk_update + t * update_term
                yk = torch.maximum(yk, torch.tensor(1e-6, device=self.device))
                
                # 更新gamma
                gamma1 = gamma2
                xk_prev = xk.clone()
                xk = yk.clone()
            
            # 每10次迭代清理内存
            if (iter_idx + 1) % 10 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # 确保非负
        xk = torch.maximum(xk, torch.tensor(0, device=self.device))
        
        return xk
    
    def deblur_core_torch(self, data: torch.Tensor, kernel: torch.Tensor, 
                         iterations: int = 10, rule: int = 1) -> torch.Tensor:
        """
        核心去卷积函数（PyTorch加速版）
        
        参数:
            data: 输入图像
            kernel: 卷积核
            iterations: 迭代次数
            rule: 算法规则 (1: Richardson-Lucy, 2: Landweber)
            
        返回:
            去卷积后的图像
        """
        # 获取输入尺寸
        dx, dy = data.shape[-2], data.shape[-1]
        
        # 计算边界填充尺寸
        B = min(dx, dy) // 6
        
        # 边缘填充（使用修复后的填充函数）
        padded_data = self._pad_data(data, B)
        
        # 计算OTF
        otf = self.psf2otf_torch(kernel, padded_data.shape)
        
        # 根据规则选择算法
        if rule == 1:  # Richardson-Lucy
            result = self.richardson_lucy_torch(padded_data, otf, iterations, acceleration=True)
        elif rule == 2:  # Landweber
            result = self.landweber_torch(padded_data, otf, iterations, step_size=1.0)
        else:
            raise ValueError(f"不支持的算法规则: {rule}")
        
        # 裁剪填充区域
        if result.ndim == 2:
            result_cropped = result[B:-B, B:-B]
        else:
            result_cropped = result[:, B:-B, B:-B]
        
        return result_cropped
    
    def iterative_deconv_batch(self, data: Union[np.ndarray, torch.Tensor], 
                              kernel: Union[np.ndarray, torch.Tensor],
                              iterations: int = 10, rule: int = 1) -> torch.Tensor:
        """
        批量迭代去卷积（支持2D和3D数据）
        
        参数:
            data: 输入数据，可以是2D或3D
            kernel: 卷积核
            iterations: 迭代次数
            rule: 算法规则
            
        返回:
            去卷积后的数据
        """
        # 准备输入数据
        data_tensor = self._prepare_input(data)
        kernel_tensor = self._prepare_kernel(kernel)
        
        if data_tensor.ndim == 2:
            # 2D数据
            result = self.deblur_core_torch(data_tensor, kernel_tensor, iterations, rule)
        elif data_tensor.ndim == 3:
            # 3D数据 - 逐片处理
            batch_size = data_tensor.shape[0]
            results = []
            
            for i in range(batch_size):
                if self.progress_callback:
                    progress = 75 + int((i + 1) * 25 / batch_size)  # 75-100%
                    self.progress_callback(progress)
                    
                slice_result = self.deblur_core_torch(
                    data_tensor[i], kernel_tensor, iterations, rule
                )
                results.append(slice_result.unsqueeze(0))
            
            result = torch.cat(results, dim=0)
        else:
            raise ValueError(f"不支持的数据维度: {data_tensor.ndim}")
        
        return result
    
    def iterative_deconv_3d(self, data_3d: Union[np.ndarray, torch.Tensor], 
                           kernel_3d: Union[np.ndarray, torch.Tensor],
                           iterations: int = 10, rule: int = 1) -> torch.Tensor:
        """
        3D迭代去卷积（处理整个3D体积）
        
        参数:
            data_3d: 3D输入数据
            kernel_3d: 3D卷积核
            iterations: 迭代次数
            rule: 算法规则
            
        返回:
            去卷积后的3D数据
        """
        # 准备输入数据
        data_tensor = self._prepare_input(data_3d)
        kernel_tensor = self._prepare_kernel(kernel_3d)
        
        if data_tensor.ndim != 3:
            raise ValueError(f"需要3D数据，但得到{data_tensor.ndim}D")
        
        # 获取尺寸
        depth, height, width = data_tensor.shape
        
        # 计算边界填充尺寸
        B = min(height, width) // 6
        
        # 3D边缘填充（对每个深度切片单独填充）
        padded_slices = []
        for i in range(depth):
            slice_data = data_tensor[i]
            padded_slice = self._pad_data(slice_data, B)
            padded_slices.append(padded_slice.unsqueeze(0))
        
        padded_data = torch.cat(padded_slices, dim=0)
        
        # 计算3D OTF
        otf = self.psf2otf_torch(kernel_tensor, padded_data.shape)
        
        # 3D RL或Landweber去卷积
        if rule == 1:  # Richardson-Lucy 3D
            result = self._richardson_lucy_3d_torch(padded_data, otf, iterations)
        elif rule == 2:  # Landweber 3D
            result = self._landweber_3d_torch(padded_data, otf, iterations)
        else:
            raise ValueError(f"不支持的算法规则: {rule}")
        
        # 裁剪填充区域
        result_cropped = result[:, B:-B, B:-B]
        
        return result_cropped
    
    def _richardson_lucy_3d_torch(self, data: torch.Tensor, otf: torch.Tensor,
                                 iterations: int = 10) -> torch.Tensor:
        """
        3D Richardson-Lucy去卷积
        """
        # 初始化
        yk = data.clone()
        xk = torch.zeros_like(data)
        
        # 预计算
        ones_tensor = torch.ones_like(data)
        otf_conj = torch.conj(otf)
        
        for iter_idx in range(iterations):
            xk_prev = xk
            
            # RL迭代 (3D)
            fft_yk = torch.fft.fftn(yk, dim=(-3, -2, -1))
            conv_result = torch.fft.ifftn(otf * fft_yk, dim=(-3, -2, -1)).real
            conv_result = torch.maximum(conv_result, torch.tensor(1e-6, device=self.device))
            
            ratio = data / conv_result
            fft_ratio = torch.fft.fftn(ratio, dim=(-3, -2, -1))
            numerator = torch.fft.ifftn(otf_conj * fft_ratio, dim=(-3, -2, -1)).real
            
            denominator = torch.fft.ifftn(
                torch.fft.fftn(ones_tensor, dim=(-3, -2, -1)) * otf,
                dim=(-3, -2, -1)
            ).real
            
            xk = yk * numerator / torch.maximum(denominator, torch.tensor(1e-6, device=self.device))
            xk = torch.maximum(xk, torch.tensor(1e-6, device=self.device))
            
            # 更新yk（简单更新，无加速）
            yk = xk.clone()
            
            # 清理内存
            if (iter_idx + 1) % 5 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return torch.maximum(xk, torch.tensor(0, device=self.device))
    
    def _landweber_3d_torch(self, data: torch.Tensor, otf: torch.Tensor,
                           iterations: int = 10) -> torch.Tensor:
        """
        3D Landweber去卷积
        """
        # 初始化
        yk = data.clone()
        xk = data.clone()
        
        # 预计算
        otf_conj = torch.conj(otf)
        t = 1.0  # 步长
        
        for iter_idx in range(iterations):
            # Landweber迭代 (3D)
            fft_yk = torch.fft.fftn(yk, dim=(-3, -2, -1))
            fft_data = torch.fft.fftn(data, dim=(-3, -2, -1))
            
            update_term = torch.fft.ifftn(
                otf_conj * (fft_data - otf * fft_yk),
                dim=(-3, -2, -1)
            ).real
            
            xk = yk + t * update_term
            xk = torch.maximum(xk, torch.tensor(1e-6, device=self.device))
            
            # 简单更新（无加速）
            yk = xk.clone()
            
            # 清理内存
            if (iter_idx + 1) % 5 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return torch.maximum(xk, torch.tensor(0, device=self.device))
    
    def iterative_deconv(self, data: Union[np.ndarray, torch.Tensor],
                        kernel: Union[np.ndarray, torch.Tensor],
                        iteration: int, rule: int) -> np.ndarray:
        """
        兼容性函数，保持与原函数相同的接口
        
        参数:
            data: 输入数据
            kernel: 卷积核
            iteration: 迭代次数
            rule: 算法规则 (1: RL, 2: Landweber)
            
        返回:
            去卷积后的数据 (NumPy数组)
        """
        # 执行去卷积
        result_tensor = self.iterative_deconv_batch(data, kernel, iteration, rule)
        
        # 转换为NumPy数组
        return result_tensor.cpu().numpy()
    
    def benchmark(self, data_shape: Tuple[int, ...] = (256, 256),
                 kernel_shape: Tuple[int, ...] = (32, 32),
                 iterations: int = 20, rule: int = 1,
                 num_runs: int = 5) -> dict:
        """
        性能基准测试
        
        参数:
            data_shape: 数据形状
            kernel_shape: 卷积核形状
            iterations: 迭代次数
            rule: 算法规则
            num_runs: 运行次数
            
        返回:
            性能统计
        """
        import time
        
        print(f"\n{'='*60}")
        print("迭代去卷积性能基准测试")
        print(f"{'='*60}")
        print(f"数据形状: {data_shape}")
        print(f"卷积核形状: {kernel_shape}")
        print(f"迭代次数: {iterations}")
        print(f"算法规则: {'Richardson-Lucy' if rule == 1 else 'Landweber'}")
        print(f"设备: {self.device}")
        print(f"{'='*60}")
        
        # 生成测试数据
        torch.manual_seed(42)
        data = torch.randn(*data_shape, dtype=self.dtype, device=self.device).abs()
        kernel = torch.randn(*kernel_shape, dtype=self.dtype, device=self.device).abs()
        kernel = kernel / torch.sum(kernel)
        
        # 预热
        print("预热...")
        for _ in range(3):
            _ = self.iterative_deconv_batch(data, kernel, 5, rule)
        
        # 计时
        times = []
        for i in range(num_runs):
            start_time = time.time()
            result = self.iterative_deconv_batch(data, kernel, iterations, rule)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            print(f"运行 {i+1}/{num_runs}: {times[-1]:.3f}秒")
        
        # 统计
        stats = {
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'iterations_per_sec': iterations / np.mean(times),
            'total_iterations': iterations * num_runs
        }
        
        print(f"\n性能统计:")
        print(f"  平均时间: {stats['avg_time']:.3f}秒")
        print(f"  最慢时间: {stats['max_time']:.3f}秒")
        print(f"  最快时间: {stats['min_time']:.3f}秒")
        print(f"  标准差: {stats['std_time']:.3f}秒")
        print(f"  迭代速度: {stats['iterations_per_sec']:.2f} 迭代/秒")
        
        if self.device.type == 'cuda':
            memory_used = torch.cuda.max_memory_allocated() / 1024**2
            memory_cached = torch.cuda.max_memory_reserved() / 1024**2
            print(f"  最大GPU内存使用: {memory_used:.2f} MB")
            print(f"  最大GPU内存缓存: {memory_cached:.2f} MB")
        
        return stats


# 兼容性函数（保持与原接口相同）
def iterative_deconv(data, kernel, iteration, rule):
    """
    兼容性函数，保持与原接口相同
    """
    deconv = Sparse_deconvolution_SIM(
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    return deconv.iterative_deconv(data, kernel, iteration, rule)


# 辅助函数（保持兼容性）
def cart2pol(x, y):
    """
    直角坐标转极坐标
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    """
    极坐标转直角坐标
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def psf2otf(psf, outSize):
    """
    保持与原函数相同的PSF转OTF函数
    """
    import numpy as np
    psfSize = np.array(psf.shape)
    outSize = np.array(outSize)
    padSize = outSize - psfSize
    psf = np.pad(psf, ((0, int(padSize[0])), (0, int(padSize[1]))), 'constant')
    
    for i in range(len(psfSize)):
        psf = np.roll(psf, -int(psfSize[i] / 2), i)
    
    otf = np.fft.fftn(psf)
    
    # 如果虚部很小，转换为实数
    if np.max(np.abs(np.imag(otf))) / np.max(np.abs(otf)) <= 1e-6:
        otf = np.real(otf)
    
    return otf


def rliter(yk, data, otf):
    """
    保持与原函数相同的RL迭代辅助函数
    """
    import numpy as np
    return np.fft.fftn(data / np.maximum(np.fft.ifftn(otf * np.fft.fftn(yk)), 1e-6))


# 使用示例
if __name__ == "__main__":
    # 创建去卷积器
    deconv = Sparse_deconvolution_SIM(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dtype=torch.float32
    )
    
    # 生成测试数据
    np.random.seed(42)
    test_data = np.random.randn(256, 256).astype(np.float32).clip(0, None)
    test_kernel = np.ones((32, 32), dtype=np.float32)
    test_kernel = test_kernel / np.sum(test_kernel)
    
    # 执行去卷积
    print("执行Richardson-Lucy去卷积...")
    result_rl = deconv.iterative_deconv_batch(
        test_data, test_kernel, 
        iterations=20, rule=1
    )
    
    print("执行Landweber去卷积...")
    result_lw = deconv.iterative_deconv_batch(
        test_data, test_kernel,
        iterations=20, rule=2
    )
    
    print(f"\n结果形状: {result_rl.shape}")
    print(f"结果范围: [{result_rl.min():.3f}, {result_rl.max():.3f}]")
    
    # 性能测试
    stats = deconv.benchmark(
        data_shape=(256, 256),
        kernel_shape=(32, 32),
        iterations=20,
        rule=1,
        num_runs=3
    )













# import math
# import warnings
# import numpy as np
# from numpy import zeros

# try:
#     import cupy as cp
# except ImportError:
#     cupy = None

# xp = np if cp is None else cp

# if xp is not cp:
#     warnings.warn("could not import cupy... falling back to numpy & cpu.")

# def iterative_deconv(data,kernel,iteration,rule):
#     if xp is not np:
#         data = xp.asarray(data)
#         kernel = xp.asarray(kernel)

#     if data.ndim > 2:
#         data_de = xp.zeros((data.shape[0], data.shape[1],data.shape[2]), dtype = 'float32')
#         for i in range(0, data.shape[0]):
#             data_de[i, :, :] = (deblur_core(data[i, :,:], kernel, iteration, rule)).real
#     else:
#         data_de = (deblur_core(data, kernel, iteration, rule)).real

#     if xp is not np:
#         data_de = xp.asnumpy(data_de)

#     return data_de

# def deblur_core(data, kernel, iteration, rule):

#     #data = cp.asnumpy(data)
#     kernel = xp.array(kernel)
#     kernel = kernel / sum(sum(kernel))
#     kernel_initial = kernel
#     [dx,dy] = data.shape

#     B = math.floor(min(dx,dy)/6)
#     data = xp.pad(data, [int(B),int(B)], 'edge')
#     yk = data
#     xk = zeros((data.shape[0], data.shape[1]), dtype = 'float32')
#     vk = zeros((data.shape[0], data.shape[1]), dtype = 'float32')
#     otf = psf2otf(kernel_initial, data.shape)

#     if rule == 2: 
#     #LandWeber deconv
#         t = 1
#         gamma1 = 1
#         for i in range(0,iteration):

#             if i == 0:
#                 xk_update = data

#                 xk = data + t*xp.fft.ifftn(xp.conj(otf)) * (xp.fft.fftn(data) - (otf *xp.fft.fftn(data)))
#             else:
#                 gamma2 = 1/2*(4 * gamma1*gamma1 + gamma1**4)**(1/2) - gamma1**2
#                 beta = -gamma2 *(1 - 1 / gamma1)
#                 yk_update = xk + beta * (xk - xk_update)
#                 yk = yk_update + t * xp.fft.ifftn(xp.conj(otf) * (xp.fft.fftn(data) - (otf * xp.fft.fftn(yk_update))))
#                 yk = xp.maximum(yk, 1e-6, dtype = 'float32')
#                 gamma1 = gamma2
#                 xk_update = xk
#                 xk = yk

#     elif rule == 1:
#     #Richardson-Lucy deconv

#         for iter in range(0, iteration):

#             xk_update = xk
#             rliter1 = rliter(yk, data, otf)

#             xk = yk * ((xp.fft.ifftn(xp.conj(otf) * rliter1)).real) / ( (xp.fft.ifftn(xp.fft.fftn(xp.ones(data.shape)) * otf)).real)

#             xk = xp.maximum(xk, 1e-6, dtype = 'float32')

#             vk_update = vk

#             vk =xp.maximum(xk - yk, 1e-6 , dtype = 'float32')

#             if iter == 0:
#                 alpha = 0
#                 yk = xk
#                 yk = xp.maximum(yk, 1e-6,dtype = 'float32')
#                 yk = xp.array(yk)

#             else:

#                 alpha = sum(sum(vk_update * vk))/(sum(sum(vk_update * vk_update)) + 1e-10)
#                 alpha = xp.maximum(xp.minimum(alpha, 1), 1e-6, dtype = 'float32')
#                # start = time.clock()
#                 yk = xk + alpha * (xk - xk_update)
#                 yk = xp.maximum(yk, 1e-6, dtype = 'float32')
#                 yk[xp.isnan(yk)] = 1e-6
#                 #end = time.clock()
#                # print(start, end)
#                 #K=np.isnan(yk)

#     yk[yk < 0] = 0
#     yk = xp.array(yk, dtype = 'float32')
#     data_decon = yk[B + 0:yk.shape[0] - B, B + 0: yk.shape[1] - B]

#     return data_decon

# def cart2pol(x, y):
#     rho = xp.sqrt(x ** 2 + y ** 2)
#     phi = xp.arctan2(y, x)
#     return (rho, phi)

# def pol2cart(rho, phi):
#     x = rho * xp.cos(phi)
#     y = rho * xp.sin(phi)
#     return (x, y)

# def psf2otf(psf, outSize):
#     psfSize = xp.array(psf.shape)
#     outSize = xp.array(outSize)
#     padSize = xp.array(outSize - psfSize)
#     psf = xp.pad(psf, ((0, int(padSize[0])), (0, int(padSize[1]))), 'constant')
#     for i in range(len(psfSize)):
#         psf = xp.roll(psf, -int(psfSize[i] / 2), i)
#     otf = xp.fft.fftn(psf)
#     # nElem = np.prod(psfSize)    
#     nElem = 1
#     for dim in psfSize:
#         nElem *= dim
#     nOps = 0
#     for k in range(len(psfSize)):
#         nffts = nElem / psfSize[k]
#         nOps = nOps + psfSize[k] * xp.log2(psfSize[k]) * nffts
#     if xp.max(xp.abs(xp.imag(otf))) / xp.max(xp.abs(otf)) <= nOps * xp.finfo(xp.float32).eps:
#         otf = xp.real(otf)
#     return otf

# def rliter(yk,data,otf):
#     rliter = xp.fft.fftn(data / xp.maximum(xp.fft.ifftn(otf * xp.fft.fftn(yk)), 1e-6))
#     return rliter

