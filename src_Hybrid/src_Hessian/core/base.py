"""
稀疏Hessian去卷积 - 基类与接口定义
"""

import torch
import torch.fft
import numpy as np
import gc
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Dict, Any


class BaseSparseHessian(ABC):
    """
    稀疏Hessian去卷积基类
    定义统一的接口和公共方法
    """
    
    def __init__(self, 
                 device: Optional[str] = None,
                 dtype: torch.dtype = torch.float32,
                 verbose: bool = True):
        """
        初始化基类
        
        Args:
            device: 计算设备 ('cuda', 'cpu', 或None自动选择)
            dtype: 数据类型
            verbose: 是否显示详细信息
        """
        self.dtype = dtype
        self.verbose = verbose
        
        # 设备设置
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        if verbose:
            print(f"初始化 {self.__class__.__name__}")
            print(f"  设备: {self.device}")
            print(f"  数据类型: {self.dtype}")
    
    def _prepare_input(self, f: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        准备输入张量
        
        Args:
            f: 输入图像 (NumPy数组或PyTorch张量)
            
        Returns:
            准备好的PyTorch张量
        """
        if isinstance(f, np.ndarray):
            tensor = torch.from_numpy(f.astype(np.float32))
        elif isinstance(f, torch.Tensor):
            tensor = f
        else:
            raise TypeError(f"不支持的输入类型: {type(f)}")
        
        # 确保在正确设备和数据类型上
        if tensor.device != self.device:
            tensor = tensor.to(self.device)
        if tensor.dtype != self.dtype:
            tensor = tensor.to(self.dtype)
            
        return tensor
    
    def _process_dimensions_(self, f: torch.Tensor, contiz: float) -> Tuple[torch.Tensor, float, int]:
        """
        处理输入维度（与原始算法逻辑一致）
        
        Returns:
            f_processed: 处理后的图像
            contiz_val: 处理后的contiz值
            flage: 标志位 (1表示2D输入, 0表示3D输入)
        """
        # 处理contiz参数
        if self.device.type == 'cpu':
            contiz_val = np.sqrt(contiz)
        else:
            contiz_val = torch.sqrt(torch.tensor(contiz, dtype=self.dtype, device=self.device))
        
        # 归一化
        # factor = f.max()
        # f1 = f / factor
        f1 = f
        flage = 0
        
        # 维度处理
        f_flag = len(f1.shape)
        if f_flag == 2:
            # 2D图像扩展为3D
            contiz_val = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            flage = 1
            f_processed = torch.zeros((3, f1.shape[0], f1.shape[1]), 
                                     dtype=self.dtype, device=self.device)
            for i in range(3):
                f_processed[i, :, :] = f1
                
        elif f_flag > 2:
            if f1.shape[0] < 3:
                contiz_val = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                f_processed = torch.zeros((3, f1.shape[1], f1.shape[2]), 
                                         dtype=self.dtype, device=self.device)
                f_processed[0:f1.shape[0], :, :] = f1
                for i in range(f1.shape[0], 3):
                    f_processed[i, :, :] = f_processed[1, :, :]
            else:
                f_processed = f1
        else:
            raise ValueError(f"不支持的维度: {f_flag}")
        
        return f_processed, contiz_val, flage
    
    def _process_dimensions(self, f: torch.Tensor, contiz: float) -> Tuple[torch.Tensor, float, int]:
        """
        处理输入维度（与原始算法逻辑一致）
        
        Returns:
            f_processed: 处理后的图像
            contiz_val: 处理后的contiz值
            flage: 标志位 (1表示2D输入, 0表示3D输入)
        """
        # 处理contiz参数
        if self.device.type == 'cpu':
            contiz_val = np.sqrt(contiz)
        else:
            contiz_val = torch.sqrt(torch.tensor(contiz, dtype=self.dtype, device=self.device))
        
        f1 = f
        original_dims = len(f1.shape)
        
        # 维度处理
        if original_dims == 2:
            # 2D图像扩展为3D
            contiz_val = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            flage = 1
            f_processed = torch.zeros((3, f1.shape[0], f1.shape[1]), 
                                    dtype=self.dtype, device=self.device)
            for i in range(3):
                f_processed[i, :, :] = f1
            
            # 记录原始形状，便于后续恢复
            self._original_shape = (1, f1.shape[0], f1.shape[1])
            
        elif original_dims == 3:
            # 3D图像处理
            original_frames = f1.shape[0]
            flage = 0
            
            if original_frames == 1:
                # 单帧3D输入的特殊处理
                # 这里输入形状是 (1, H, W)，需要扩展为3帧
                contiz_val = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                f_processed = torch.zeros((3, f1.shape[1], f1.shape[2]), 
                                        dtype=self.dtype, device=self.device)
                for i in range(3):
                    f_processed[i, :, :] = f1[0]
                
                # 记录原始形状
                self._original_shape = (1, f1.shape[1], f1.shape[2])
                
            elif original_frames == 2:
                # 2帧输入的特殊处理
                contiz_val = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                f_processed = torch.zeros((3, f1.shape[1], f1.shape[2]), 
                                        dtype=self.dtype, device=self.device)
                f_processed[0, :, :] = f1[0]
                f_processed[1, :, :] = f1[1]
                f_processed[2, :, :] = f1[1]  # 复制第二帧作为第三帧
                
                # 记录原始形状
                self._original_shape = (2, f1.shape[1], f1.shape[2])
                
            elif original_frames < 3:
                # 其他小于3帧的情况
                contiz_val = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                f_processed = torch.zeros((3, f1.shape[1], f1.shape[2]), 
                                        dtype=self.dtype, device=self.device)
                f_processed[:original_frames, :, :] = f1
                for i in range(original_frames, 3):
                    f_processed[i, :, :] = f_processed[original_frames-1, :, :]
                
                # 记录原始形状
                self._original_shape = (original_frames, f1.shape[1], f1.shape[2])
                
            else:
                # 3帧或以上的正常处理
                f_processed = f1
                
                # 记录原始形状
                self._original_shape = f1.shape
        else:
            raise ValueError(f"不支持的维度: {original_dims}")
        
        return f_processed, contiz_val, flage

    @abstractmethod
    def Hessian(self, 
                      f: Union[np.ndarray, torch.Tensor],
                      iteration_num: int = 100,
                      fidelity: float = 150,
                      sparsity: float = 10,
                      contiz: float = 0.5,
                      mu: float = 1) -> torch.Tensor:
        """
        稀疏Hessian去卷积主函数（抽象方法）
        
        Args:
            f: 输入图像
            iteration_num: 迭代次数
            fidelity: 保真度权重
            sparsity: 稀疏性权重
            contiz: Z轴连续性参数
            mu: 参数
            
        Returns:
            去卷积结果
        """
        pass
    
    def save_result(self, 
                   tensor: torch.Tensor, 
                   filename: str,
                   scale_to_uint8: bool = False):
        """
        保存结果张量为图像文件
        
        Args:
            tensor: PyTorch张量
            filename: 保存的文件名
            scale_to_uint8: 是否缩放到0-255范围
        """
        # 转换为NumPy数组
        if tensor.is_cuda:
            numpy_array = tensor.detach().cpu().numpy()
        else:
            numpy_array = tensor.detach().numpy()
        
        # 可选：缩放到0-255范围
        if scale_to_uint8:
            numpy_array = (numpy_array - numpy_array.min()) / (numpy_array.max() - numpy_array.min()) * 255
            numpy_array = numpy_array.astype(np.uint8)
        
        # 保存图像（这里使用PIL，您可以根据需要修改）
        try:
            from PIL import Image
            if numpy_array.ndim == 2:
                Image.fromarray(numpy_array).save(filename)
            elif numpy_array.ndim == 3:
                # 保存为多页TIFF或分别保存各通道
                for i in range(numpy_array.shape[0]):
                    Image.fromarray(numpy_array[i]).save(f"{filename}_channel_{i}.tif")
        except ImportError:
            # 如果没有PIL，保存为.npy文件
            np.save(filename.replace('.tif', '.npy'), numpy_array)
        
        if self.verbose:
            print(f"结果已保存: {filename} (形状: {numpy_array.shape})")
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        info = {
            'device': str(self.device),
            'dtype': str(self.dtype),
            'class_name': self.__class__.__name__
        }
        
        if self.device.type == 'cuda':
            info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info['gpu_name'] = torch.cuda.get_device_name(self.device)
                info['memory_allocated'] = torch.cuda.memory_allocated(self.device) / 1024**2
                info['memory_reserved'] = torch.cuda.memory_reserved(self.device) / 1024**2
        
        return info