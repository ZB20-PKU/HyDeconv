"""
Hessian算子频域计算模块
所有版本共用
"""

import torch
import torch.fft
from typing import Tuple


class HessianOperations:
    """
    Hessian算子频域计算
    包含FFT规范化修复，确保与NumPy/CuPy版本等价
    """
    
    def __init__(self, device=None, dtype=torch.float32):
        self.dtype = dtype
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
    
    def _fft_with_normalization(self, kernel: torch.Tensor, gsize: Tuple[int, int, int]) -> torch.Tensor:
        """
        带有规范化的FFT计算
        模拟NumPy/CuPy的隐式1/N规范化
        """
        fft_result = torch.fft.fftn(kernel, s=gsize)
        # norm_factor = torch.prod(torch.tensor(gsize, dtype=torch.float32))
        # fft_result = fft_result / norm_factor
        return fft_result
    
    def operation_xx(self, gsize: Tuple[int, int, int]) -> torch.Tensor:
        """xx方向二阶导数的频域能量谱"""
        delta_xx = torch.tensor([[[1.0, -2.0, 1.0]]], 
                                dtype=self.dtype, device=self.device)
        delta_fft = self._fft_with_normalization(delta_xx, gsize)
        xxfft = (delta_fft * torch.conj(delta_fft)).real
        return xxfft
    
    def operation_xy(self, gsize: Tuple[int, int, int]) -> torch.Tensor:
        """xy方向混合导数的频域能量谱"""
        delta_xy = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]], 
                                dtype=self.dtype, device=self.device)
        delta_fft = self._fft_with_normalization(delta_xy, gsize)
        xyfft = (delta_fft * torch.conj(delta_fft)).real
        return xyfft
    
    def operation_xz(self, gsize: Tuple[int, int, int]) -> torch.Tensor:
        """xz方向混合导数的频域能量谱"""
        delta_xz = torch.tensor([[[1.0, -1.0]], [[-1.0, 1.0]]], 
                                dtype=self.dtype, device=self.device)
        delta_fft = self._fft_with_normalization(delta_xz, gsize)
        xzfft = (delta_fft * torch.conj(delta_fft)).real
        return xzfft
    
    def operation_yy(self, gsize: Tuple[int, int, int]) -> torch.Tensor:
        """yy方向二阶导数的频域能量谱"""
        delta_yy = torch.tensor([[[1.0]], [[-2.0]], [[1.0]]], 
                                dtype=self.dtype, device=self.device)
        delta_fft = self._fft_with_normalization(delta_yy, gsize)
        yyfft = (delta_fft * torch.conj(delta_fft)).real
        return yyfft
    
    def operation_yz(self, gsize: Tuple[int, int, int]) -> torch.Tensor:
        """yz方向混合导数的频域能量谱"""
        delta_yz = torch.tensor([[[1.0], [-1.0]], [[-1.0], [1.0]]], 
                                dtype=self.dtype, device=self.device)
        delta_fft = self._fft_with_normalization(delta_yz, gsize)
        yzfft = (delta_fft * torch.conj(delta_fft)).real
        return yzfft
    
    def operation_zz(self, gsize: Tuple[int, int, int]) -> torch.Tensor:
        """zz方向二阶导数的频域能量谱"""
        delta_zz = torch.tensor([[[1.0]], [[-2.0]], [[1.0]]], 
                                dtype=self.dtype, device=self.device)
        delta_fft = self._fft_with_normalization(delta_zz, gsize)
        zzfft = (delta_fft * torch.conj(delta_fft)).real
        return zzfft
    
    def compute_all_operations(self, gsize: Tuple[int, int, int]) -> dict:
        """一次性计算所有算子（用于调试）"""
        return {
            'xx': self.operation_xx(gsize),
            'xy': self.operation_xy(gsize),
            'xz': self.operation_xz(gsize),
            'yy': self.operation_yy(gsize),
            'yz': self.operation_yz(gsize),
            'zz': self.operation_zz(gsize)
        }


# class HessianOperations:
#     """
#     Hessian算子频域计算
#     包含FFT规范化修复，确保与NumPy/CuPy版本等价
#     """
    
#     def __init__(self, device=None, dtype=torch.float32):
#         self.dtype = dtype
#         if device is None:
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         else:
#             self.device = torch.device(device)
    
#     def _fft_with_normalization(self, kernel: torch.Tensor, gsize: Tuple[int, int, int]) -> torch.Tensor:
#         """
#         带有规范化的FFT计算
#         模拟NumPy/CuPy的隐式1/N规范化
#         """
#         fft_result = torch.fft.fftn(kernel, s=gsize)
#         # norm_factor = torch.prod(torch.tensor(gsize, dtype=torch.float32))
#         # fft_result = fft_result / norm_factor
#         return fft_result
    
#     def operation_xx(self, gsize: Tuple[int, int, int]) -> torch.Tensor:
#         """xx方向二阶导数的频域能量谱"""
#         delta_xx = torch.tensor([[[1.0, -2.0, 1.0]]], 
#                                 dtype=self.dtype, device=self.device)
#         # delta_fft = self._fft_with_normalization(delta_xx, gsize)
#         delta_fft = self._fft_with_normalization(delta_xx, gsize)
#         norm_factor = torch.prod(torch.tensor(gsize, dtype=self.dtype, device=self.device))
#         xxfft = (delta_fft * torch.conj(delta_fft)).real / (norm_factor ** 2)
#         return xxfft
    
#     def operation_xy(self, gsize: Tuple[int, int, int]) -> torch.Tensor:
#         """xy方向混合导数的频域能量谱"""
#         delta_xy = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]], 
#                                 dtype=self.dtype, device=self.device)
#         delta_fft = self._fft_with_normalization(delta_xy, gsize)
#         norm_factor = torch.prod(torch.tensor(gsize, dtype=self.dtype, device=self.device))
#         xyfft = (delta_fft * torch.conj(delta_fft)).real/ (norm_factor ** 2)
#         return xyfft
    
#     def operation_xz(self, gsize: Tuple[int, int, int]) -> torch.Tensor:
#         """xz方向混合导数的频域能量谱"""
#         delta_xz = torch.tensor([[[1.0, -1.0]], [[-1.0, 1.0]]], 
#                                 dtype=self.dtype, device=self.device)
#         delta_fft = self._fft_with_normalization(delta_xz, gsize)
#         norm_factor = torch.prod(torch.tensor(gsize, dtype=self.dtype, device=self.device))
#         xzfft = (delta_fft * torch.conj(delta_fft)).real/ (norm_factor ** 2)
#         return xzfft
    
#     def operation_yy(self, gsize: Tuple[int, int, int]) -> torch.Tensor:
#         """yy方向二阶导数的频域能量谱"""
#         delta_yy = torch.tensor([[[1.0]], [[-2.0]], [[1.0]]], 
#                                 dtype=self.dtype, device=self.device)
#         delta_fft = self._fft_with_normalization(delta_yy, gsize)
#         norm_factor = torch.prod(torch.tensor(gsize, dtype=self.dtype, device=self.device))
#         yyfft = (delta_fft * torch.conj(delta_fft)).real/ (norm_factor ** 2)
#         return yyfft
    
#     def operation_yz(self, gsize: Tuple[int, int, int]) -> torch.Tensor:
#         """yz方向混合导数的频域能量谱"""
#         delta_yz = torch.tensor([[[1.0], [-1.0]], [[-1.0], [1.0]]], 
#                                 dtype=self.dtype, device=self.device)
#         delta_fft = self._fft_with_normalization(delta_yz, gsize)
#         norm_factor = torch.prod(torch.tensor(gsize, dtype=self.dtype, device=self.device))
#         yzfft = (delta_fft * torch.conj(delta_fft)).real/ (norm_factor ** 2)
#         return yzfft
    
#     def operation_zz(self, gsize: Tuple[int, int, int]) -> torch.Tensor:
#         """zz方向二阶导数的频域能量谱"""
#         delta_zz = torch.tensor([[[1.0]], [[-2.0]], [[1.0]]], 
#                                 dtype=self.dtype, device=self.device)
#         delta_fft = self._fft_with_normalization(delta_zz, gsize)
#         norm_factor = torch.prod(torch.tensor(gsize, dtype=self.dtype, device=self.device))
#         zzfft = (delta_fft * torch.conj(delta_fft)).real/ (norm_factor ** 2)
#         return zzfft
    
#     def compute_all_operations(self, gsize: Tuple[int, int, int]) -> dict:
#         """一次性计算所有算子（用于调试）"""
#         return {
#             'xx': self.operation_xx(gsize),
#             'xy': self.operation_xy(gsize),
#             'xz': self.operation_xz(gsize),
#             'yy': self.operation_yy(gsize),
#             'yz': self.operation_yz(gsize),
#             'zz': self.operation_zz(gsize)
#         }