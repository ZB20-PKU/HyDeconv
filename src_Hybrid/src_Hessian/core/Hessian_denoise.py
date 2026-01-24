"""
精确的稀疏Hessian去卷积 - 优化加速版
保持与原始NumPy/CuPy代码完全等价，但显著加速
"""

import torch
import torch.fft
import numpy as np
import gc
from typing import Optional, Union, Tuple
from .base import BaseSparseHessian
from .operations import HessianOperations
from .iterations import OriginalIterations
import time
import os
from src.ImageLib import ImageLib

class Hessian_denoise_SIM(BaseSparseHessian):
    
    def __init__(self, args, progress_callback=None,
                 device: Optional[str] = None,
                 dtype: torch.dtype = torch.float32,
                 verbose: bool = False):
        super().__init__(device, dtype, verbose)
        
        # 初始化操作和迭代模块
        self.hessian_ops = HessianOperations(device=args.device, dtype=self.dtype)
        self.sparse_iter = OriginalIterations(device=args.device, dtype=self.dtype)
        
        # 预分配的缓冲区
        self._buffers = {}
        self.device = torch.device(args.device)
        # 性能统计
        self.iteration_times = []
        self.memory_clear_count = 0

        #   Hessian parameters
        # self.BF_SIM = args.BF_SIM
        self.iteration_num = args.Hessian_iteration_number
        self.fidelity = args.Hessian_fidelity
        self.contiz = args.Hessian_Z_continuity
        self.filedir = args.filedir
        self.fname = args.fname
        self.progress_callback = progress_callback
        self._stop_requested = False
        # self.suffix_savefilename = args.suffix_savefilename
        self.ext = args.ext
        # self.BF_SIM = ImageLib.read(os.path.join(self.filedir,
        #                     self.fname + '_1_BF_SIM'\
        #                     + self.ext), to_tensor=True)
        self.BF_SIM = torch.from_numpy(args.bf_result.astype(np.float32))
        self.BF_factor = self.BF_SIM.max()
        args.BF_factor = self.BF_factor
        # self.num_frames, self.imgsize_ori = ImageLib.getInfo(\
        #                     os.path.join(self.filedir,
        #                     self.fname + '_1_BF_SIM'\
        #                     + self.ext))
        # self.num_frames, self.imgsize_ori = ImageLib.getInfo(\
        #                     os.path.join(self.filedir,
        #                     self.fname + '_1_BF_SIM'\
        #                     + self.ext))
        self.verbose = args.debug
        self.Hessian_Z_rolling_window_size = args.Hessian_Z_rolling_window_size
    
    def _init_buffers(self, shape: Tuple[int, ...]):
        """初始化预分配缓冲区"""
        self._buffers = {
            'Lxx': torch.zeros(shape, dtype=self.dtype, device=self.device),
            'Lyy': torch.zeros(shape, dtype=self.dtype, device=self.device),
            'Lzz': torch.zeros(shape, dtype=self.dtype, device=self.device),
            'Lxy': torch.zeros(shape, dtype=self.dtype, device=self.device),
            'Lxz': torch.zeros(shape, dtype=self.dtype, device=self.device),
            'Lyz': torch.zeros(shape, dtype=self.dtype, device=self.device),
            # 'Lsparse': torch.zeros(shape, dtype=self.dtype, device=self.device),
            'g_fft': torch.zeros(shape, dtype=torch.complex64 if self.dtype == torch.float32 else torch.complex128,
                               device=self.device),
        }
    
    def _clear_buffers(self):
        """清理缓冲区"""
        self._buffers.clear()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        # end_time = time.time()
        # times.append(end_time - start_time)
        # print(f"运行: {times[-1]:.2f}秒")

    def recon(self, window_size: int = 50, overlap: int = 5, 
          progress_start: int = 25, progress_end: int = 50):
        """
        滚动窗口重建函数
        
        参数:
        ----------
        window_size : int
            每次处理的帧数，默认为50
        overlap : int
            窗口之间的重叠帧数，默认为5
        progress_start : int
            进度开始百分比，默认为25
        progress_end : int
            进度结束百分比，默认为50
            
        返回:
        ----------
        ndarray
            重建后的图像（在CPU上的numpy数组）
        """
        window_size = self.Hessian_Z_rolling_window_size
        if window_size < 20:
            window_size = 20
        # print(window_size)
        # 验证进度范围
        if progress_start < 0 or progress_end > 100 or progress_start >= progress_end:
            raise ValueError(f"进度范围无效: {progress_start}-{progress_end}，必须满足 0 ≤ start < end ≤ 100")
        
        # 获取数据基本信息
        self.BF_SIM = self.BF_SIM / self.BF_factor
        num_frames = self.BF_SIM.shape[0]
        
        # 修复：检查输入形状并保存
        self._original_input_shape = self.BF_SIM.shape
    

        # 如果帧数不超过窗口大小，直接处理
        if num_frames <= window_size:
            if self.verbose:
                print(f"帧数 ({num_frames}) <= 窗口大小 ({window_size})，直接处理全部数据")
            
            if self.progress_callback:
                self.progress_callback(progress_start)
            
            # 处理主数据
            img = self.Hessian(self.BF_SIM, self.iteration_num, self.fidelity, self.contiz)
            
            # 对前5帧进行翻转处理
            if num_frames >= 5:
                first_five = img[:5]
                flipped_first_five = torch.flip(first_five, dims=[0])
                processed_flipped = self.Hessian(flipped_first_five, 
                                                self.iteration_num, 
                                                self.fidelity, 
                                                self.contiz, 
                                                flag=False)
                processed_original_order = torch.flip(processed_flipped, dims=[0])
                img[:2] = processed_original_order[:2]
            
            if self.progress_callback:
                self.progress_callback(progress_end)
            
            # 转换为CPU上的numpy数组
            result = img * self.BF_factor
            return result.cpu().numpy() if isinstance(result, torch.Tensor) else result
        
        # 滚动窗口处理
        if self.verbose:
            print(f"使用滚动窗口处理: {num_frames}帧, 窗口大小={window_size}, 重叠={overlap}")
            print(f"进度范围: {progress_start}%-{progress_end}%")
        
        # 计算步长和窗口数量
        step = window_size - 2 * overlap  # 保留的中间部分
        num_windows = (num_frames - 2 * overlap + step - 1) // step  # 向上取整
        
        # 初始化结果数组
        result = torch.zeros_like(self.BF_SIM, device=self.device)
        window_counts = torch.zeros(num_frames, device=self.device)
        
        # 进度范围计算
        progress_range = progress_end - progress_start
        progress_per_window = progress_range / num_windows
        
        # 报告处理开始
        if self.progress_callback:
            self.progress_callback(progress_start)
        
        # 处理第一个窗口（前window_size帧）
        if self.verbose:
            print(f"处理窗口 1/{num_windows}: 帧 0-{window_size-1}")
        
        # 对第一个窗口应用翻转处理
        first_window = self.BF_SIM[:window_size]
        flipped_first_five = torch.flip(first_window[:5], dims=[0])
        processed_flipped = self.Hessian(flipped_first_five, 
                                        self.iteration_num, 
                                        self.fidelity, 
                                        self.contiz, 
                                        flag=False)
        processed_original_order = torch.flip(processed_flipped, dims=[0])
        
        # 处理整个第一个窗口
        img_first = self.Hessian(first_window, 
                                self.iteration_num, 
                                self.fidelity, 
                                self.contiz, 
                                flag=False)
        img_first[:2] = processed_original_order[:2]
        
        # 保留中间部分
        keep_start = 0
        keep_end = window_size - overlap
        
        result[:keep_end] += img_first[:keep_end]
        window_counts[:keep_end] += 1
        
        # 更新进度
        if self.progress_callback:
            current_progress = progress_start + progress_per_window
            self.progress_callback(int(current_progress))
        
        # 处理中间窗口
        for w in range(1, num_windows - 1):
            start_idx = w * step - overlap
            end_idx = start_idx + window_size
            
            if end_idx > num_frames:
                end_idx = num_frames
                start_idx = end_idx - window_size
            
            if self.verbose:
                print(f"处理窗口 {w+1}/{num_windows}: 帧 {start_idx}-{end_idx-1}")
            
            # 处理当前窗口
            window_data = self.BF_SIM[start_idx:end_idx]
            img_window = self.Hessian(window_data, 
                                    self.iteration_num, 
                                    self.fidelity, 
                                    self.contiz, 
                                    flag=False)
            
            # 保留中间部分
            result[start_idx + overlap:end_idx - overlap] += img_window[overlap:-overlap]
            window_counts[start_idx + overlap:end_idx - overlap] += 1
            
            # 更新进度
            if self.progress_callback:
                current_progress = progress_start + (w + 1) * progress_per_window
                self.progress_callback(int(min(progress_end, current_progress)))
            
            # 清理内存
            del window_data, img_window
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # 处理最后一个窗口
        if self.verbose:
            print(f"处理窗口 {num_windows}/{num_windows}: 最后{window_size}帧")
        
        last_start = num_frames - window_size
        last_window = self.BF_SIM[last_start:]
        img_last = self.Hessian(last_window, 
                                self.iteration_num, 
                                self.fidelity, 
                                self.contiz, 
                                flag=False)
        
        # 保留结尾部分
        result[last_start + overlap:] += img_last[overlap:]
        window_counts[last_start + overlap:] += 1
        
        # 更新进度到结束点
        if self.progress_callback:
            self.progress_callback(progress_end)
        
        # 平均处理重叠区域
        mask = window_counts > 0
        result[mask] = result[mask] / window_counts[mask].view(-1, 1, 1)
        
        # 处理可能未被覆盖的帧（可选，根据实际需要保留或移除）
        zero_mask = window_counts == 0
        if zero_mask.any():
            if self.verbose:
                zero_count = zero_mask.sum().item()
                zero_percentage = zero_count / num_frames * 100
                print(f"注意: {zero_count}帧({zero_percentage:.2f}%)未被覆盖")
            
            # 简单插值处理未被覆盖的帧
            for i in range(num_frames):
                if zero_mask[i]:
                    # 寻找最近的非零帧
                    left_idx = i - 1
                    right_idx = i + 1
                    
                    # 向左寻找
                    while left_idx >= 0 and zero_mask[left_idx]:
                        left_idx -= 1
                    
                    # 向右寻找
                    while right_idx < num_frames and zero_mask[right_idx]:
                        right_idx += 1
                    
                    # 如果两边都找到，线性插值
                    if left_idx >= 0 and right_idx < num_frames:
                        left_dist = i - left_idx
                        right_dist = right_idx - i
                        total_dist = left_dist + right_dist
                        
                        result[i] = (right_dist / total_dist) * result[left_idx] + \
                                (left_dist / total_dist) * result[right_idx]
                    # 如果只有左边找到
                    elif left_idx >= 0:
                        result[i] = result[left_idx]
                    # 如果只有右边找到
                    elif right_idx < num_frames:
                        result[i] = result[right_idx]
        
        # 最终处理：转换为CPU上的numpy数组
        result = result * self.BF_factor
        
        # 确保数据转换到CPU和numpy格式
        if isinstance(result, torch.Tensor):
            result = result.cpu().numpy()
        
        # 确保数据类型正确
        result = result.astype(np.float32)
        
        if self.verbose:
            print(f"处理完成，结果形状: {result.shape}, 数据类型: {result.dtype}")

        # 在处理完成后，如果需要，可以恢复原始形状
        if hasattr(self, '_original_input_shape'):
            original_shape = self._original_input_shape
            
            # 如果原始输入是2D，但结果是3D，需要调整
            if len(original_shape) == 2 and len(result.shape) == 3:
                # 2D输入应该返回2D结果
                if result.shape[0] == 1:
                    result = result[0]  # 去除帧维度
                elif result.shape[0] == 3:
                    result = result[1]  # 取中间帧
            
            # 清理
            del self._original_input_shape

        if self._stop_requested:
            # 清理内存
            self.cleanup()
            return None

        return result

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
    
    def Hessian(self, 
                      f: Union[np.ndarray, torch.Tensor],
                      iteration_num: int = 100,
                      fidelity: float = 150,
                      contiz: float = 0.5,
                      mu: float = 1,
                      flag: bool = True) -> torch.Tensor:
        """
        优化的稀疏Hessian去卷积
        通过预计算和内存重用加速，功能与原始代码完全一致
        """
        if self.verbose:
            print(f"开始优化稀疏去卷积 (迭代次数: {iteration_num})...")
        
        import time
        total_start_time = time.time()
        
        # 准备输入数据
        f_tensor = self._prepare_input(f)
        
        # 使用无梯度上下文
        with torch.no_grad():
            # 处理维度
            f_processed, contiz_val, flage = self._process_dimensions(f_tensor, contiz)
            imgsize = f_processed.shape
            
            # 初始化缓冲区
            self._init_buffers(imgsize)
            
            # 预计算频域算子（使用内存重用的优化版本）
            if self.verbose:
                print("  预计算Hessian频域算子...")
            
            # 预计算所有算子并重用内存
            operators = self._precompute_operators(imgsize, contiz_val)
            
            # 预计算归一化因子
            normlize = (fidelity / mu) + operators['operationfft']
            # normlize = (fidelity / mu) + (sparsity**2) + operators['operationfft']
            
            # 初始化变量（重用已有内存）
            zeros_tensor = torch.zeros(imgsize, dtype=self.dtype, device=self.device)
            bxx = zeros_tensor.clone()
            byy = zeros_tensor.clone()
            bzz = zeros_tensor.clone()
            bxy = zeros_tensor.clone()
            bxz = zeros_tensor.clone()
            byz = zeros_tensor.clone()
            # bl1 = zeros_tensor.clone()
            
            # 预计算常量
            fidelity_mu = fidelity / mu
            # sparsity_squared = sparsity ** 2
            contiz_squared = contiz_val ** 2
            
            # 初始化g_update（重用内存）
            g_update = f_processed * fidelity_mu
            
            # 主迭代循环 - 优化版
            if self.verbose:
                print(f"  开始迭代，共{iteration_num}次...")
            
            # 预分配迭代用临时变量
            temp_sum = torch.zeros_like(g_update)
            
            for iter_idx in range(iteration_num):
                if self._stop_requested:
                    break
                if flag:
                    if self.progress_callback:
                        progress = 25 + int((iter_idx + 1) * 25 / iteration_num)  # 25-50%
                        self.progress_callback(progress)

                iter_start = time.time()
                
                # 优化：使用预分配的fft缓冲区
                g_fft = torch.fft.fftn(g_update, dim=(-3, -2, -1), out=self._buffers['g_fft'])
                
                # # 第一次迭代特殊处理
                # if iter_idx == 0:
                #     g = torch.fft.ifftn(g_fft / fidelity_mu).real
                # else:
                #     g = torch.fft.ifftn(g_fft / normlize).real

                # 第一次迭代特殊处理
                if iter_idx == 0:
                    g = torch.fft.ifftn(g_fft / fidelity_mu).real
                else:
                    g = torch.fft.ifftn(g_fft / normlize).real
                
                # 重置g_update（重用内存）
                g_update.copy_(f_processed * fidelity_mu)
                # g_update = f_processed * fidelity_mu
                
                # 各方向迭代更新（使用优化版本）
                # 优化：重用缓冲区，避免每次分配新内存
                Lxx, bxx = self.sparse_iter.iter_xx(g, bxx, 1, mu)
                Lyy, byy = self.sparse_iter.iter_yy(g, byy, 1, mu)
                Lzz, bzz = self.sparse_iter.iter_zz(g, bzz, contiz_squared, mu)
                Lxy, bxy = self.sparse_iter.iter_xy(g, bxy, 2, mu)
                Lxz, bxz = self.sparse_iter.iter_xz(g, bxz, 2 * contiz_val, mu)
                Lyz, byz = self.sparse_iter.iter_yz(g, byz, 2 * contiz_val, mu)
                # Lsparse, bl1 = self.sparse_iter.iter_sparse(g, bl1, sparsity, mu)
                
                # 优化：使用原地累加
                temp_sum.zero_()
                # temp_sum.add_(Lxx).add_(Lyy).add_(Lzz).add_(Lxy).add_(Lxz).add_(Lyz).add_(Lsparse)
                temp_sum.add_(Lxx).add_(Lyy).add_(Lzz).add_(Lxy).add_(Lxz).add_(Lyz)
                g_update.add_(temp_sum)
                
                # 记录迭代时间
                iter_time = time.time() - iter_start
                self.iteration_times.append(iter_time)
                
                # 每10次迭代输出进度
                if self.verbose and (iter_idx + 1) % 10 == 0:
                    # avg_time = np.mean(self.iteration_times[-10:]) * 1000
                    print(f'    迭代 {iter_idx + 1}/{iteration_num} - 耗时: {iter_time*1000:.1f}ms')
            
            # 后处理
            g = g
        
        # 最终清理
        self._clear_buffers()
        # del bxx, byy, bzz, bxy, bxz, byz, bl1, f_processed, normlize, g_update, temp_sum, operators
        del bxx, byy, bzz, bxy, bxz, byz, f_processed, normlize, g_update, temp_sum, operators
        self._clear_memory(force=True)
        
        # # 返回结果
        # if flage:
        #     result = g[1, :, :]  # 2D输入时返回中间切片
        # else:
        #     result = g

        # 根据原始形状恢复输出
        if hasattr(self, '_original_shape'):
            original_shape = self._original_shape
            
            # 根据flage和原始形状决定输出
            if flage:  # 2D输入，返回2D结果
                result = g[1, :, :]  # 返回中间切片
                
            else:  # 3D输入
                if original_shape[0] == 1:
                    # 单帧输入，返回单帧
                    result = g[1:2, :, :]  # 返回中间帧，但保持3D形状(1, H, W)
                elif original_shape[0] == 2:
                    # 2帧输入，返回前2帧
                    result = g[:2, :, :]
                elif original_shape[0] < g.shape[0]:
                    # 原始帧数少于处理帧数，返回原始帧数
                    result = g[:original_shape[0], :, :]
                else:
                    # 正常情况，返回全部
                    result = g
            
            # 清理原始形状记录
            del self._original_shape
        else:
            # 如果没有记录原始形状，使用默认逻辑
            if flage:
                result = g[1, :, :]  # 2D输入时返回中间切片
            else:
                result = g
        
        # 性能报告
        if self.verbose:
            total_time = time.time() - total_start_time
            self._print_performance_report(total_time, iteration_num)       
        
        return result
    
    def _precompute_operators(self, imgsize: Tuple[int, ...], contiz_val: float) -> dict:
        """
        预计算并缓存所有频域算子
        显著减少重复计算
        """
        # 检查是否已缓存
        cache_key = f"{imgsize}_{contiz_val}"
        
        xxfft = self.hessian_ops.operation_xx(imgsize)
        yyfft = self.hessian_ops.operation_yy(imgsize)
        zzfft = self.hessian_ops.operation_zz(imgsize)
        xyfft = self.hessian_ops.operation_xy(imgsize)
        xzfft = self.hessian_ops.operation_xz(imgsize)
        yzfft = self.hessian_ops.operation_yz(imgsize)
        
        # 计算组合算子
        operationfft = (xxfft + yyfft + (contiz_val**2) * zzfft + 
                       2 * xyfft + 2 * contiz_val * xzfft + 2 * contiz_val * yzfft)
        
        return {
            'xxfft': xxfft,
            'yyfft': yyfft,
            'zzfft': zzfft,
            'xyfft': xyfft,
            'xzfft': xzfft,
            'yzfft': yzfft,
            'operationfft': operationfft
        }
    
    # def _optimized_iteration_step(self, g: torch.Tensor, 
    #                              bxx: torch.Tensor, byy: torch.Tensor, bzz: torch.Tensor,
    #                              bxy: torch.Tensor, bxz: torch.Tensor, byz: torch.Tensor,
    #                              bl1: torch.Tensor,
    #                              contiz_val: float, sparsity: float, mu: float) -> Tuple[torch.Tensor, ...]:
    #     """
    #     优化的迭代步骤，减少内存分配
    #     """
    #     # 重用预分配的缓冲区
    #     Lxx, bxx = self.sparse_iter.iter_xx(g, bxx, 1, mu)
    #     Lyy, byy = self.sparse_iter.iter_yy(g, byy, 1, mu)
    #     Lzz, bzz = self.sparse_iter.iter_zz(g, bzz, contiz_val**2, mu)
    #     Lxy, bxy = self.sparse_iter.iter_xy(g, bxy, 2, mu)
    #     Lxz, bxz = self.sparse_iter.iter_xz(g, bxz, 2 * contiz_val, mu)
    #     Lyz, byz = self.sparse_iter.iter_yz(g, byz, 2 * contiz_val, mu)
    #     # Lsparse, bl1 = self.sparse_iter.iter_sparse(g, bl1, sparsity, mu)
        
    #     return Lxx, Lyy, Lzz, Lxy, Lxz, Lyz, bxx, byy, bzz, bxy, bxz, byz, bl1
    #     # return Lxx, Lyy, Lzz, Lxy, Lxz, Lyz, Lsparse, bxx, byy, bzz, bxy, bxz, byz, bl1
    
    def _clear_memory(self, force: bool = False):
        """优化内存清理"""
        if force or len(self._buffers) > 0:
            self._clear_buffers()
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # 确保所有操作完成
            self.memory_clear_count += 1
    
    def _print_performance_report(self, total_time: float, iteration_num: int):
        """打印性能报告"""
        print(f"\n{'='*60}")
        print(f"{self.__class__.__name__} - 性能报告")
        print(f"{'='*60}")
        print(f"总时间: {total_time:.2f}秒")
        print(f"迭代次数: {iteration_num}")
        
        if self.iteration_times:
            avg_iter_time = np.mean(self.iteration_times) * 1000
            min_iter_time = np.min(self.iteration_times) * 1000
            max_iter_time = np.max(self.iteration_times) * 1000
            print(f"平均每迭代: {avg_iter_time:.1f}毫秒")
            print(f"最快迭代: {min_iter_time:.1f}毫秒")
            print(f"最慢迭代: {max_iter_time:.1f}毫秒")
            print(f"吞吐量: {1000/avg_iter_time:.2f} 迭代/秒")
        
        print(f"内存清理次数: {self.memory_clear_count}")
        print(f"设备: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU内存使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"GPU内存缓存: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

    def benchmark(self, f: Union[np.ndarray, torch.Tensor], 
                 warmup: int = 5, runs: int = 10,
                 **kwargs) -> dict:
        """
        性能基准测试
        """
        print(f"\n{'='*60}")
        print(f"开始性能基准测试")
        print(f"{'='*60}")
        
        # 预热
        print(f"预热 {warmup} 次...")
        for i in range(warmup):
            _ = self.sparse_hessian(f, **kwargs)
        
        # 正式测试
        times = []
        for i in range(runs):
            start_time = time.time()
            result = self.sparse_hessian(f, **kwargs)
            end_time = time.time()
            times.append(end_time - start_time)
            print(f"运行 {i+1}/{runs}: {times[-1]:.2f}秒")
        
        # 统计结果
        stats = {
            'runs': runs,
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'throughput': runs / np.sum(times),
        }
        
        print(f"\n基准测试结果:")
        print(f"  平均时间: {stats['avg_time']:.2f}秒")
        print(f"  最小时: {stats['min_time']:.2f}秒")
        print(f"  最大时间: {stats['max_time']:.2f}秒")
        print(f"  标准差: {stats['std_time']:.2f}秒")
        print(f"  吞吐量: {stats['throughput']:.2f} 次/秒")
        
        return stats


class Hessian_denoise_FM(BaseSparseHessian):
    
    def __init__(self, args, progress_callback=None,
                 device: Optional[str] = None,
                 dtype: torch.dtype = torch.float32,
                 verbose: bool = False):
        super().__init__(device, dtype, verbose)
        
        # 初始化操作和迭代模块
        self.hessian_ops = HessianOperations(device=args.device, dtype=self.dtype)
        self.sparse_iter = OriginalIterations(device=args.device, dtype=self.dtype)
        
        # 预分配的缓冲区
        self._buffers = {}
        self.device = torch.device(args.device)
        # 性能统计
        self.iteration_times = []
        self.memory_clear_count = 0

        #   Hessian parameters
        # self.BF_SIM = args.BF_SIM
        ImageLib.getInfo(args.Raw_data_path)
        args.filedir, filename = os.path.split(args.Raw_data_path)
        args.fname, args.ext = os.path.splitext(filename)
        self.iteration_num = args.Hessian_iteration_number
        self.fidelity = args.Hessian_fidelity
        self.contiz = args.Hessian_Z_continuity
        self.filedir = args.filedir
        self.fname = args.fname
        self.progress_callback = progress_callback
        self._stop_requested = False
        # self.suffix_savefilename = args.suffix_savefilename
        self.ext = args.ext
        self.BF_SIM = ImageLib.read(args.Raw_data_path, to_tensor=True)
        # self.BF_SIM = torch.from_numpy(args.bf_result.astype(np.float32))
        self.BF_factor = self.BF_SIM.max()
        args.BF_factor = self.BF_factor
        # self.num_frames, self.imgsize_ori = ImageLib.getInfo(\
        #                     os.path.join(self.filedir,
        #                     self.fname + '_1_BF_SIM'\
        #                     + self.ext))
        # self.num_frames, self.imgsize_ori = ImageLib.getInfo(\
        #                     os.path.join(self.filedir,
        #                     self.fname + '_1_BF_SIM'\
        #                     + self.ext))
        self.verbose = args.debug
        self.Hessian_Z_rolling_window_size = args.Hessian_Z_rolling_window_size
    
    def _init_buffers(self, shape: Tuple[int, ...]):
        """初始化预分配缓冲区"""
        self._buffers = {
            'Lxx': torch.zeros(shape, dtype=self.dtype, device=self.device),
            'Lyy': torch.zeros(shape, dtype=self.dtype, device=self.device),
            'Lzz': torch.zeros(shape, dtype=self.dtype, device=self.device),
            'Lxy': torch.zeros(shape, dtype=self.dtype, device=self.device),
            'Lxz': torch.zeros(shape, dtype=self.dtype, device=self.device),
            'Lyz': torch.zeros(shape, dtype=self.dtype, device=self.device),
            # 'Lsparse': torch.zeros(shape, dtype=self.dtype, device=self.device),
            'g_fft': torch.zeros(shape, dtype=torch.complex64 if self.dtype == torch.float32 else torch.complex128,
                               device=self.device),
        }
    
    def _clear_buffers(self):
        """清理缓冲区"""
        self._buffers.clear()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def recon(self, window_size: int = 50, overlap: int = 5, 
          progress_start: int = 0, progress_end: int = 33):
        """
        滚动窗口重建函数
        
        参数:
        ----------
        window_size : int
            每次处理的帧数，默认为50
        overlap : int
            窗口之间的重叠帧数，默认为5
        progress_start : int
            进度开始百分比，默认为25
        progress_end : int
            进度结束百分比，默认为50
            
        返回:
        ----------
        ndarray
            重建后的图像（在CPU上的numpy数组）
        """
        window_size = self.Hessian_Z_rolling_window_size
        if window_size < 20:
            window_size = 20
        # 验证进度范围
        if progress_start < 0 or progress_end > 100 or progress_start >= progress_end:
            raise ValueError(f"进度范围无效: {progress_start}-{progress_end}，必须满足 0 ≤ start < end ≤ 100")
        
        # 获取数据基本信息
        self.BF_SIM = self.BF_SIM / self.BF_factor
        if len(self.BF_SIM.shape) == 2:
            self.BF_SIM = self.BF_SIM.unsqueeze(0)
            
        num_frames = self.BF_SIM.shape[0]
        # print(num_frames)

        # 修复：检查输入形状并保存
        self._original_input_shape = self.BF_SIM.shape
        # print(self._original_input_shape)
        
        # 如果帧数不超过窗口大小，直接处理
        if num_frames <= window_size:
            if self.verbose:
                print(f"帧数 ({num_frames}) <= 窗口大小 ({window_size})，直接处理全部数据")
            
            if self.progress_callback:
                self.progress_callback(progress_start)
            
            # 处理主数据
            img = self.Hessian(self.BF_SIM, self.iteration_num, self.fidelity, self.contiz)
            
            # 对前5帧进行翻转处理
            if num_frames >= 5:
                first_five = img[:5]
                flipped_first_five = torch.flip(first_five, dims=[0])
                processed_flipped = self.Hessian(flipped_first_five, 
                                                self.iteration_num, 
                                                self.fidelity, 
                                                self.contiz, 
                                                flag=False)
                processed_original_order = torch.flip(processed_flipped, dims=[0])
                img[:2] = processed_original_order[:2]
            
            if self.progress_callback:
                self.progress_callback(progress_end)
            
            # 转换为CPU上的numpy数组
            result = img * self.BF_factor
            return result.cpu().numpy() if isinstance(result, torch.Tensor) else result
        
        # 滚动窗口处理
        if self.verbose:
            print(f"使用滚动窗口处理: {num_frames}帧, 窗口大小={window_size}, 重叠={overlap}")
            print(f"进度范围: {progress_start}%-{progress_end}%")
        
        # 计算步长和窗口数量
        step = window_size - 2 * overlap  # 保留的中间部分
        num_windows = (num_frames - 2 * overlap + step - 1) // step  # 向上取整
        
        # 初始化结果数组
        result = torch.zeros_like(self.BF_SIM, device=self.device)
        window_counts = torch.zeros(num_frames, device=self.device)
        
        # 进度范围计算
        progress_range = progress_end - progress_start
        progress_per_window = progress_range / num_windows
        
        # 报告处理开始
        if self.progress_callback:
            self.progress_callback(progress_start)
        
        # 处理第一个窗口（前window_size帧）
        if self.verbose:
            print(f"处理窗口 1/{num_windows}: 帧 0-{window_size-1}")
        
        # 对第一个窗口应用翻转处理
        first_window = self.BF_SIM[:window_size]
        flipped_first_five = torch.flip(first_window[:5], dims=[0])
        processed_flipped = self.Hessian(flipped_first_five, 
                                        self.iteration_num, 
                                        self.fidelity, 
                                        self.contiz, 
                                        flag=False)
        processed_original_order = torch.flip(processed_flipped, dims=[0])
        
        # 处理整个第一个窗口
        img_first = self.Hessian(first_window, 
                                self.iteration_num, 
                                self.fidelity, 
                                self.contiz, 
                                flag=False)
        img_first[:2] = processed_original_order[:2]
        
        # 保留中间部分
        keep_start = 0
        keep_end = window_size - overlap
        
        result[:keep_end] += img_first[:keep_end]
        window_counts[:keep_end] += 1
        
        # 更新进度
        if self.progress_callback:
            current_progress = progress_start + progress_per_window
            self.progress_callback(int(current_progress))
        
        # 处理中间窗口
        for w in range(1, num_windows - 1):
            start_idx = w * step - overlap
            end_idx = start_idx + window_size
            
            if end_idx > num_frames:
                end_idx = num_frames
                start_idx = end_idx - window_size
            
            if self.verbose:
                print(f"处理窗口 {w+1}/{num_windows}: 帧 {start_idx}-{end_idx-1}")
            
            # 处理当前窗口
            window_data = self.BF_SIM[start_idx:end_idx]
            img_window = self.Hessian(window_data, 
                                    self.iteration_num, 
                                    self.fidelity, 
                                    self.contiz, 
                                    flag=False)
            
            # 保留中间部分
            result[start_idx + overlap:end_idx - overlap] += img_window[overlap:-overlap]
            window_counts[start_idx + overlap:end_idx - overlap] += 1
            
            # 更新进度
            if self.progress_callback:
                current_progress = progress_start + (w + 1) * progress_per_window
                self.progress_callback(int(min(progress_end, current_progress)))
            
            # 清理内存
            del window_data, img_window
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # 处理最后一个窗口
        if self.verbose:
            print(f"处理窗口 {num_windows}/{num_windows}: 最后{window_size}帧")
        
        last_start = num_frames - window_size
        last_window = self.BF_SIM[last_start:]
        img_last = self.Hessian(last_window, 
                                self.iteration_num, 
                                self.fidelity, 
                                self.contiz, 
                                flag=False)
        
        # 保留结尾部分
        result[last_start + overlap:] += img_last[overlap:]
        window_counts[last_start + overlap:] += 1
        
        # 更新进度到结束点
        if self.progress_callback:
            self.progress_callback(progress_end)
        
        # 平均处理重叠区域
        mask = window_counts > 0
        result[mask] = result[mask] / window_counts[mask].view(-1, 1, 1)
        
        # 处理可能未被覆盖的帧（可选，根据实际需要保留或移除）
        zero_mask = window_counts == 0
        if zero_mask.any():
            if self.verbose:
                zero_count = zero_mask.sum().item()
                zero_percentage = zero_count / num_frames * 100
                print(f"注意: {zero_count}帧({zero_percentage:.2f}%)未被覆盖")
            
            # 简单插值处理未被覆盖的帧
            for i in range(num_frames):
                if zero_mask[i]:
                    # 寻找最近的非零帧
                    left_idx = i - 1
                    right_idx = i + 1
                    
                    # 向左寻找
                    while left_idx >= 0 and zero_mask[left_idx]:
                        left_idx -= 1
                    
                    # 向右寻找
                    while right_idx < num_frames and zero_mask[right_idx]:
                        right_idx += 1
                    
                    # 如果两边都找到，线性插值
                    if left_idx >= 0 and right_idx < num_frames:
                        left_dist = i - left_idx
                        right_dist = right_idx - i
                        total_dist = left_dist + right_dist
                        
                        result[i] = (right_dist / total_dist) * result[left_idx] + \
                                (left_dist / total_dist) * result[right_idx]
                    # 如果只有左边找到
                    elif left_idx >= 0:
                        result[i] = result[left_idx]
                    # 如果只有右边找到
                    elif right_idx < num_frames:
                        result[i] = result[right_idx]
        
        # 最终处理：转换为CPU上的numpy数组
        result = result * self.BF_factor
        
        # 确保数据转换到CPU和numpy格式
        if isinstance(result, torch.Tensor):
            result = result.cpu().numpy()
        
        # 确保数据类型正确
        result = result.astype(np.float32)

        # 在处理完成后，如果需要，可以恢复原始形状
        if hasattr(self, '_original_input_shape'):
            original_shape = self._original_input_shape
            
            # 如果原始输入是2D，但结果是3D，需要调整
            if len(original_shape) == 2 and len(result.shape) == 3:
                # 2D输入应该返回2D结果
                if result.shape[0] == 1:
                    result = result[0]  # 去除帧维度
                elif result.shape[0] == 3:
                    result = result[1]  # 取中间帧
            
            # 清理
            del self._original_input_shape
            
        if self._stop_requested:
            # 清理内存
            self.cleanup()
            return None
        
        if self.verbose:
            print(f"处理完成，结果形状: {result.shape}, 数据类型: {result.dtype}")
        
        return result


        # end_time = time.time()
        # times.append(end_time - start_time)
        # print(f"运行: {times[-1]:.2f}秒")

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

    def Hessian(self, 
                      f: Union[np.ndarray, torch.Tensor],
                      iteration_num: int = 100,
                      fidelity: float = 150,
                      contiz: float = 0.5,
                      mu: float = 1,
                      flag: bool = True) -> torch.Tensor:
        """
        优化的稀疏Hessian去卷积
        通过预计算和内存重用加速，功能与原始代码完全一致
        """
        if self.verbose:
            print(f"开始优化稀疏去卷积 (迭代次数: {iteration_num})...")
        
        import time
        total_start_time = time.time()
        
        # 准备输入数据
        f_tensor = self._prepare_input(f)
        
        # 使用无梯度上下文
        with torch.no_grad():
            # 处理维度
            f_processed, contiz_val, flage = self._process_dimensions(f_tensor, contiz)
            imgsize = f_processed.shape
            
            # 初始化缓冲区
            self._init_buffers(imgsize)
            
            # 预计算频域算子（使用内存重用的优化版本）
            if self.verbose:
                print("  预计算Hessian频域算子...")
            
            # 预计算所有算子并重用内存
            operators = self._precompute_operators(imgsize, contiz_val)
            
            # 预计算归一化因子
            normlize = (fidelity / mu) + operators['operationfft']
            # normlize = (fidelity / mu) + (sparsity**2) + operators['operationfft']
            
            # 初始化变量（重用已有内存）
            zeros_tensor = torch.zeros(imgsize, dtype=self.dtype, device=self.device)
            bxx = zeros_tensor.clone()
            byy = zeros_tensor.clone()
            bzz = zeros_tensor.clone()
            bxy = zeros_tensor.clone()
            bxz = zeros_tensor.clone()
            byz = zeros_tensor.clone()
            # bl1 = zeros_tensor.clone()
            
            # 预计算常量
            fidelity_mu = fidelity / mu
            # sparsity_squared = sparsity ** 2
            contiz_squared = contiz_val ** 2
            
            # 初始化g_update（重用内存）
            g_update = f_processed * fidelity_mu
            
            # 主迭代循环 - 优化版
            if self.verbose:
                print(f"  开始迭代，共{iteration_num}次...")
            
            # 预分配迭代用临时变量
            temp_sum = torch.zeros_like(g_update)
            
            for iter_idx in range(iteration_num):
                if self._stop_requested:
                    break
                if flag:
                    if self.progress_callback:
                        progress = 0 + int((iter_idx + 1) * 33 / iteration_num)  # 1-33%
                        self.progress_callback(progress)

                iter_start = time.time()
                
                # 优化：使用预分配的fft缓冲区
                g_fft = torch.fft.fftn(g_update, dim=(-3, -2, -1), out=self._buffers['g_fft'])
                
                # # 第一次迭代特殊处理
                # if iter_idx == 0:
                #     g = torch.fft.ifftn(g_fft / fidelity_mu).real
                # else:
                #     g = torch.fft.ifftn(g_fft / normlize).real

                # 第一次迭代特殊处理
                if iter_idx == 0:
                    g = torch.fft.ifftn(g_fft / fidelity_mu).real
                else:
                    g = torch.fft.ifftn(g_fft / normlize).real
                
                # 重置g_update（重用内存）
                g_update.copy_(f_processed * fidelity_mu)
                # g_update = f_processed * fidelity_mu
                
                # 各方向迭代更新（使用优化版本）
                # 优化：重用缓冲区，避免每次分配新内存
                Lxx, bxx = self.sparse_iter.iter_xx(g, bxx, 1, mu)
                Lyy, byy = self.sparse_iter.iter_yy(g, byy, 1, mu)
                Lzz, bzz = self.sparse_iter.iter_zz(g, bzz, contiz_squared, mu)
                Lxy, bxy = self.sparse_iter.iter_xy(g, bxy, 2, mu)
                Lxz, bxz = self.sparse_iter.iter_xz(g, bxz, 2 * contiz_val, mu)
                Lyz, byz = self.sparse_iter.iter_yz(g, byz, 2 * contiz_val, mu)
                # Lsparse, bl1 = self.sparse_iter.iter_sparse(g, bl1, sparsity, mu)
                
                # 优化：使用原地累加
                temp_sum.zero_()
                # temp_sum.add_(Lxx).add_(Lyy).add_(Lzz).add_(Lxy).add_(Lxz).add_(Lyz).add_(Lsparse)
                temp_sum.add_(Lxx).add_(Lyy).add_(Lzz).add_(Lxy).add_(Lxz).add_(Lyz)
                g_update.add_(temp_sum)
                
                # 记录迭代时间
                iter_time = time.time() - iter_start
                self.iteration_times.append(iter_time)
                
                # 每10次迭代输出进度
                if self.verbose and (iter_idx + 1) % 10 == 0:
                    # avg_time = np.mean(self.iteration_times[-10:]) * 1000
                    print(f'    迭代 {iter_idx + 1}/{iteration_num} - 耗时: {iter_time*1000:.1f}ms')
            
            # 后处理
            g = g
        
        # 最终清理
        self._clear_buffers()
        # del bxx, byy, bzz, bxy, bxz, byz, bl1, f_processed, normlize, g_update, temp_sum, operators
        del bxx, byy, bzz, bxy, bxz, byz, f_processed, normlize, g_update, temp_sum, operators
        self._clear_memory(force=True)
        
        # # 返回结果
        # if flage:
        #     result = g[1, :, :]  # 2D输入时返回中间切片
        # else:
        #     result = g
        
        # 根据原始形状恢复输出
        if hasattr(self, '_original_shape'):
            original_shape = self._original_shape
            
            # 根据flage和原始形状决定输出
            if flage:  # 2D输入，返回2D结果
                result = g[1, :, :]  # 返回中间切片
                
            else:  # 3D输入
                if original_shape[0] == 1:
                    # 单帧输入，返回单帧
                    result = g[1:2, :, :]  # 返回中间帧，但保持3D形状(1, H, W)
                elif original_shape[0] == 2:
                    # 2帧输入，返回前2帧
                    result = g[:2, :, :]
                elif original_shape[0] < g.shape[0]:
                    # 原始帧数少于处理帧数，返回原始帧数
                    result = g[:original_shape[0], :, :]
                else:
                    # 正常情况，返回全部
                    result = g
            
            # 清理原始形状记录
            del self._original_shape
        else:
            # 如果没有记录原始形状，使用默认逻辑
            if flage:
                result = g[1, :, :]  # 2D输入时返回中间切片
            else:
                result = g

        # 性能报告
        if self.verbose:
            total_time = time.time() - total_start_time
            self._print_performance_report(total_time, iteration_num)       
        
        return result
    
    def _precompute_operators(self, imgsize: Tuple[int, ...], contiz_val: float) -> dict:
        """
        预计算并缓存所有频域算子
        显著减少重复计算
        """
        # 检查是否已缓存
        cache_key = f"{imgsize}_{contiz_val}"
        
        xxfft = self.hessian_ops.operation_xx(imgsize)
        yyfft = self.hessian_ops.operation_yy(imgsize)
        zzfft = self.hessian_ops.operation_zz(imgsize)
        xyfft = self.hessian_ops.operation_xy(imgsize)
        xzfft = self.hessian_ops.operation_xz(imgsize)
        yzfft = self.hessian_ops.operation_yz(imgsize)
        
        # 计算组合算子
        operationfft = (xxfft + yyfft + (contiz_val**2) * zzfft + 
                       2 * xyfft + 2 * contiz_val * xzfft + 2 * contiz_val * yzfft)
        
        return {
            'xxfft': xxfft,
            'yyfft': yyfft,
            'zzfft': zzfft,
            'xyfft': xyfft,
            'xzfft': xzfft,
            'yzfft': yzfft,
            'operationfft': operationfft
        }
    
    # def _optimized_iteration_step(self, g: torch.Tensor, 
    #                              bxx: torch.Tensor, byy: torch.Tensor, bzz: torch.Tensor,
    #                              bxy: torch.Tensor, bxz: torch.Tensor, byz: torch.Tensor,
    #                              bl1: torch.Tensor,
    #                              contiz_val: float, sparsity: float, mu: float) -> Tuple[torch.Tensor, ...]:
    #     """
    #     优化的迭代步骤，减少内存分配
    #     """
    #     # 重用预分配的缓冲区
    #     Lxx, bxx = self.sparse_iter.iter_xx(g, bxx, 1, mu)
    #     Lyy, byy = self.sparse_iter.iter_yy(g, byy, 1, mu)
    #     Lzz, bzz = self.sparse_iter.iter_zz(g, bzz, contiz_val**2, mu)
    #     Lxy, bxy = self.sparse_iter.iter_xy(g, bxy, 2, mu)
    #     Lxz, bxz = self.sparse_iter.iter_xz(g, bxz, 2 * contiz_val, mu)
    #     Lyz, byz = self.sparse_iter.iter_yz(g, byz, 2 * contiz_val, mu)
    #     # Lsparse, bl1 = self.sparse_iter.iter_sparse(g, bl1, sparsity, mu)
        
    #     return Lxx, Lyy, Lzz, Lxy, Lxz, Lyz, bxx, byy, bzz, bxy, bxz, byz, bl1
    #     # return Lxx, Lyy, Lzz, Lxy, Lxz, Lyz, Lsparse, bxx, byy, bzz, bxy, bxz, byz, bl1
    
    def _clear_memory(self, force: bool = False):
        """优化内存清理"""
        if force or len(self._buffers) > 0:
            self._clear_buffers()
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # 确保所有操作完成
            self.memory_clear_count += 1
    
    def _print_performance_report(self, total_time: float, iteration_num: int):
        """打印性能报告"""
        print(f"\n{'='*60}")
        print(f"{self.__class__.__name__} - 性能报告")
        print(f"{'='*60}")
        print(f"总时间: {total_time:.2f}秒")
        print(f"迭代次数: {iteration_num}")
        
        if self.iteration_times:
            avg_iter_time = np.mean(self.iteration_times) * 1000
            min_iter_time = np.min(self.iteration_times) * 1000
            max_iter_time = np.max(self.iteration_times) * 1000
            print(f"平均每迭代: {avg_iter_time:.1f}毫秒")
            print(f"最快迭代: {min_iter_time:.1f}毫秒")
            print(f"最慢迭代: {max_iter_time:.1f}毫秒")
            print(f"吞吐量: {1000/avg_iter_time:.2f} 迭代/秒")
        
        print(f"内存清理次数: {self.memory_clear_count}")
        print(f"设备: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU内存使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"GPU内存缓存: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

    def benchmark(self, f: Union[np.ndarray, torch.Tensor], 
                 warmup: int = 5, runs: int = 10,
                 **kwargs) -> dict:
        """
        性能基准测试
        """
        print(f"\n{'='*60}")
        print(f"开始性能基准测试")
        print(f"{'='*60}")
        
        # 预热
        print(f"预热 {warmup} 次...")
        for i in range(warmup):
            _ = self.sparse_hessian(f, **kwargs)
        
        # 正式测试
        times = []
        for i in range(runs):
            start_time = time.time()
            result = self.sparse_hessian(f, **kwargs)
            end_time = time.time()
            times.append(end_time - start_time)
            print(f"运行 {i+1}/{runs}: {times[-1]:.2f}秒")
        
        # 统计结果
        stats = {
            'runs': runs,
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'throughput': runs / np.sum(times),
        }
        
        print(f"\n基准测试结果:")
        print(f"  平均时间: {stats['avg_time']:.2f}秒")
        print(f"  最小时: {stats['min_time']:.2f}秒")
        print(f"  最大时间: {stats['max_time']:.2f}秒")
        print(f"  标准差: {stats['std_time']:.2f}秒")
        print(f"  吞吐量: {stats['throughput']:.2f} 次/秒")
        
        return stats


class Hessian_denoise_WFM(BaseSparseHessian):
    
    def __init__(self, args, progress_callback=None,
                 device: Optional[str] = None,
                 dtype: torch.dtype = torch.float32,
                 verbose: bool = False):
        super().__init__(device, dtype, verbose)
        
        # 初始化操作和迭代模块
        self.hessian_ops = HessianOperations(device=args.device, dtype=self.dtype)
        self.sparse_iter = OriginalIterations(device=args.device, dtype=self.dtype)
        
        # 预分配的缓冲区
        self._buffers = {}
        self.device = torch.device(args.device)
        # 性能统计
        self.iteration_times = []
        self.memory_clear_count = 0

        #   Hessian parameters
        # self.BF_SIM = args.BF_SIM
        ImageLib.getInfo(args.WFM_raw_data_path)
        args.filedir, filename = os.path.split(args.WFM_raw_data_path)
        args.fname, args.ext = os.path.splitext(filename)
        self.iteration_num = args.Hessian_iteration_number
        self.fidelity = args.Hessian_fidelity
        self.contiz = args.Hessian_Z_continuity
        # self.filedir = args.filedir
        # self.fname = args.fname
        self.progress_callback = progress_callback
        self._stop_requested = False
        # self.suffix_savefilename = args.suffix_savefilename
        # self.ext = args.ext
        # self.BF_SIM = ImageLib.read(os.path.join(self.filedir,
        #                     self.fname + '_1_BF_SIM'\
        #                     + self.ext), to_tensor=True)
        self.BF_SIM = torch.from_numpy(args.bf_result.astype(np.float32))
        self.BF_factor = self.BF_SIM.max()
        args.BF_factor = self.BF_factor
        # self.num_frames, self.imgsize_ori = ImageLib.getInfo(\
        #                     os.path.join(self.filedir,
        #                     self.fname + '_1_BF_SIM'\
        #                     + self.ext))
        # self.num_frames, self.imgsize_ori = ImageLib.getInfo(\
        #                     os.path.join(self.filedir,
        #                     self.fname + '_1_BF_SIM'\
        #                     + self.ext))
        self.verbose = args.debug
        self.Hessian_Z_rolling_window_size = args.Hessian_Z_rolling_window_size
        
    
    def _init_buffers(self, shape: Tuple[int, ...]):
        """初始化预分配缓冲区"""
        self._buffers = {
            'Lxx': torch.zeros(shape, dtype=self.dtype, device=self.device),
            'Lyy': torch.zeros(shape, dtype=self.dtype, device=self.device),
            'Lzz': torch.zeros(shape, dtype=self.dtype, device=self.device),
            'Lxy': torch.zeros(shape, dtype=self.dtype, device=self.device),
            'Lxz': torch.zeros(shape, dtype=self.dtype, device=self.device),
            'Lyz': torch.zeros(shape, dtype=self.dtype, device=self.device),
            # 'Lsparse': torch.zeros(shape, dtype=self.dtype, device=self.device),
            'g_fft': torch.zeros(shape, dtype=torch.complex64 if self.dtype == torch.float32 else torch.complex128,
                               device=self.device),
        }
    
    def _clear_buffers(self):
        """清理缓冲区"""
        self._buffers.clear()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def recon(self, window_size: int = 50, overlap: int = 5, 
          progress_start: int = 25, progress_end: int = 50):
        """
        滚动窗口重建函数
        
        参数:
        ----------
        window_size : int
            每次处理的帧数，默认为50
        overlap : int
            窗口之间的重叠帧数，默认为5
        progress_start : int
            进度开始百分比，默认为25
        progress_end : int
            进度结束百分比，默认为50
            
        返回:
        ----------
        ndarray
            重建后的图像（在CPU上的numpy数组）
        """
        window_size = self.Hessian_Z_rolling_window_size
        if window_size < 20:
            window_size = 20
        # 验证进度范围
        if progress_start < 0 or progress_end > 100 or progress_start >= progress_end:
            raise ValueError(f"进度范围无效: {progress_start}-{progress_end}，必须满足 0 ≤ start < end ≤ 100")
        
        # 获取数据基本信息
        self.BF_SIM = self.BF_SIM / self.BF_factor
        num_frames = self.BF_SIM.shape[0]
        self._original_input_shape = self.BF_SIM.shape
        # print(self._original_input_shape)
        
        # 如果帧数不超过窗口大小，直接处理
        if num_frames <= window_size:
            if self.verbose:
                print(f"帧数 ({num_frames}) <= 窗口大小 ({window_size})，直接处理全部数据")
            
            if self.progress_callback:
                self.progress_callback(progress_start)
            
            # 处理主数据
            img = self.Hessian(self.BF_SIM, self.iteration_num, self.fidelity, self.contiz)
            
            # 对前5帧进行翻转处理
            if num_frames >= 5:
                first_five = img[:5]
                flipped_first_five = torch.flip(first_five, dims=[0])
                processed_flipped = self.Hessian(flipped_first_five, 
                                                self.iteration_num, 
                                                self.fidelity, 
                                                self.contiz, 
                                                flag=False)
                processed_original_order = torch.flip(processed_flipped, dims=[0])
                img[:2] = processed_original_order[:2]
            
            if self.progress_callback:
                self.progress_callback(progress_end)
            
            # 转换为CPU上的numpy数组
            result = img * self.BF_factor
            return result.cpu().numpy() if isinstance(result, torch.Tensor) else result
        
        # 滚动窗口处理
        if self.verbose:
            print(f"使用滚动窗口处理: {num_frames}帧, 窗口大小={window_size}, 重叠={overlap}")
            print(f"进度范围: {progress_start}%-{progress_end}%")
        
        # 计算步长和窗口数量
        step = window_size - 2 * overlap  # 保留的中间部分
        num_windows = (num_frames - 2 * overlap + step - 1) // step  # 向上取整
        
        # 初始化结果数组
        result = torch.zeros_like(self.BF_SIM, device=self.device)
        window_counts = torch.zeros(num_frames, device=self.device)
        
        # 进度范围计算
        progress_range = progress_end - progress_start
        progress_per_window = progress_range / num_windows
        
        # 报告处理开始
        if self.progress_callback:
            self.progress_callback(progress_start)
        
        # 处理第一个窗口（前window_size帧）
        if self.verbose:
            print(f"处理窗口 1/{num_windows}: 帧 0-{window_size-1}")
        
        # 对第一个窗口应用翻转处理
        first_window = self.BF_SIM[:window_size]
        flipped_first_five = torch.flip(first_window[:5], dims=[0])
        processed_flipped = self.Hessian(flipped_first_five, 
                                        self.iteration_num, 
                                        self.fidelity, 
                                        self.contiz, 
                                        flag=False)
        processed_original_order = torch.flip(processed_flipped, dims=[0])
        
        # 处理整个第一个窗口
        img_first = self.Hessian(first_window, 
                                self.iteration_num, 
                                self.fidelity, 
                                self.contiz, 
                                flag=False)
        img_first[:2] = processed_original_order[:2]
        
        # 保留中间部分
        keep_start = 0
        keep_end = window_size - overlap
        
        result[:keep_end] += img_first[:keep_end]
        window_counts[:keep_end] += 1
        
        # 更新进度
        if self.progress_callback:
            current_progress = progress_start + progress_per_window
            self.progress_callback(int(current_progress))
        
        # 处理中间窗口
        for w in range(1, num_windows - 1):
            start_idx = w * step - overlap
            end_idx = start_idx + window_size
            
            if end_idx > num_frames:
                end_idx = num_frames
                start_idx = end_idx - window_size
            
            if self.verbose:
                print(f"处理窗口 {w+1}/{num_windows}: 帧 {start_idx}-{end_idx-1}")
            
            # 处理当前窗口
            window_data = self.BF_SIM[start_idx:end_idx]
            img_window = self.Hessian(window_data, 
                                    self.iteration_num, 
                                    self.fidelity, 
                                    self.contiz, 
                                    flag=False)
            
            # 保留中间部分
            result[start_idx + overlap:end_idx - overlap] += img_window[overlap:-overlap]
            window_counts[start_idx + overlap:end_idx - overlap] += 1
            
            # 更新进度
            if self.progress_callback:
                current_progress = progress_start + (w + 1) * progress_per_window
                self.progress_callback(int(min(progress_end, current_progress)))
            
            # 清理内存
            del window_data, img_window
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # 处理最后一个窗口
        if self.verbose:
            print(f"处理窗口 {num_windows}/{num_windows}: 最后{window_size}帧")
        
        last_start = num_frames - window_size
        last_window = self.BF_SIM[last_start:]
        img_last = self.Hessian(last_window, 
                                self.iteration_num, 
                                self.fidelity, 
                                self.contiz, 
                                flag=False)
        
        # 保留结尾部分
        result[last_start + overlap:] += img_last[overlap:]
        window_counts[last_start + overlap:] += 1
        
        # 更新进度到结束点
        if self.progress_callback:
            self.progress_callback(progress_end)
        
        # 平均处理重叠区域
        mask = window_counts > 0
        result[mask] = result[mask] / window_counts[mask].view(-1, 1, 1)
        
        # 处理可能未被覆盖的帧（可选，根据实际需要保留或移除）
        zero_mask = window_counts == 0
        if zero_mask.any():
            if self.verbose:
                zero_count = zero_mask.sum().item()
                zero_percentage = zero_count / num_frames * 100
                print(f"注意: {zero_count}帧({zero_percentage:.2f}%)未被覆盖")
            
            # 简单插值处理未被覆盖的帧
            for i in range(num_frames):
                if zero_mask[i]:
                    # 寻找最近的非零帧
                    left_idx = i - 1
                    right_idx = i + 1
                    
                    # 向左寻找
                    while left_idx >= 0 and zero_mask[left_idx]:
                        left_idx -= 1
                    
                    # 向右寻找
                    while right_idx < num_frames and zero_mask[right_idx]:
                        right_idx += 1
                    
                    # 如果两边都找到，线性插值
                    if left_idx >= 0 and right_idx < num_frames:
                        left_dist = i - left_idx
                        right_dist = right_idx - i
                        total_dist = left_dist + right_dist
                        
                        result[i] = (right_dist / total_dist) * result[left_idx] + \
                                (left_dist / total_dist) * result[right_idx]
                    # 如果只有左边找到
                    elif left_idx >= 0:
                        result[i] = result[left_idx]
                    # 如果只有右边找到
                    elif right_idx < num_frames:
                        result[i] = result[right_idx]
        
        # 最终处理：转换为CPU上的numpy数组
        result = result * self.BF_factor
        
        # 确保数据转换到CPU和numpy格式
        if isinstance(result, torch.Tensor):
            result = result.cpu().numpy()
        
        # 确保数据类型正确
        result = result.astype(np.float32)

        # 在处理完成后，如果需要，可以恢复原始形状
        if hasattr(self, '_original_input_shape'):
            original_shape = self._original_input_shape
            
            # 如果原始输入是2D，但结果是3D，需要调整
            if len(original_shape) == 2 and len(result.shape) == 3:
                # 2D输入应该返回2D结果
                if result.shape[0] == 1:
                    result = result[0]  # 去除帧维度
                elif result.shape[0] == 3:
                    result = result[1]  # 取中间帧
            
            # 清理
            del self._original_input_shape

        if self.verbose:
            print(f"处理完成，结果形状: {result.shape}, 数据类型: {result.dtype}")
        if self._stop_requested:
            # 清理内存
            self.cleanup()
            return None
        return result

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
        # end_time = time.time()
        # times.append(end_time - start_time)
        # print(f"运行: {times[-1]:.2f}秒")

    def Hessian(self, 
                      f: Union[np.ndarray, torch.Tensor],
                      iteration_num: int = 100,
                      fidelity: float = 150,
                      contiz: float = 0.5,
                      mu: float = 1,
                      flag: bool = True) -> torch.Tensor:
        """
        优化的稀疏Hessian去卷积
        通过预计算和内存重用加速，功能与原始代码完全一致
        """
        if self.verbose:
            print(f"开始优化稀疏去卷积 (迭代次数: {iteration_num})...")
        
        import time
        total_start_time = time.time()
        
        # 准备输入数据
        f_tensor = self._prepare_input(f)
        
        # 使用无梯度上下文
        with torch.no_grad():
            # 处理维度
            f_processed, contiz_val, flage = self._process_dimensions(f_tensor, contiz)
            imgsize = f_processed.shape
            
            # 初始化缓冲区
            self._init_buffers(imgsize)
            
            # 预计算频域算子（使用内存重用的优化版本）
            if self.verbose:
                print("  预计算Hessian频域算子...")
            
            # 预计算所有算子并重用内存
            operators = self._precompute_operators(imgsize, contiz_val)
            
            # 预计算归一化因子
            normlize = (fidelity / mu) + operators['operationfft']
            # normlize = (fidelity / mu) + (sparsity**2) + operators['operationfft']
            
            # 初始化变量（重用已有内存）
            zeros_tensor = torch.zeros(imgsize, dtype=self.dtype, device=self.device)
            bxx = zeros_tensor.clone()
            byy = zeros_tensor.clone()
            bzz = zeros_tensor.clone()
            bxy = zeros_tensor.clone()
            bxz = zeros_tensor.clone()
            byz = zeros_tensor.clone()
            # bl1 = zeros_tensor.clone()
            
            # 预计算常量
            fidelity_mu = fidelity / mu
            # sparsity_squared = sparsity ** 2
            contiz_squared = contiz_val ** 2
            
            # 初始化g_update（重用内存）
            g_update = f_processed * fidelity_mu
            
            # 主迭代循环 - 优化版
            if self.verbose:
                print(f"  开始迭代，共{iteration_num}次...")
            
            # 预分配迭代用临时变量
            temp_sum = torch.zeros_like(g_update)
            
            for iter_idx in range(iteration_num):
                if self._stop_requested:
                    break
                if flag:
                    if self.progress_callback:
                        progress = 25 + int((iter_idx + 1) * 25 / iteration_num)  # 25-50%
                        self.progress_callback(progress)

                iter_start = time.time()
                
                # 优化：使用预分配的fft缓冲区
                g_fft = torch.fft.fftn(g_update, dim=(-3, -2, -1), out=self._buffers['g_fft'])
                
                # # 第一次迭代特殊处理
                # if iter_idx == 0:
                #     g = torch.fft.ifftn(g_fft / fidelity_mu).real
                # else:
                #     g = torch.fft.ifftn(g_fft / normlize).real

                # 第一次迭代特殊处理
                if iter_idx == 0:
                    g = torch.fft.ifftn(g_fft / fidelity_mu).real
                else:
                    g = torch.fft.ifftn(g_fft / normlize).real
                
                # 重置g_update（重用内存）
                g_update.copy_(f_processed * fidelity_mu)
                # g_update = f_processed * fidelity_mu
                
                # 各方向迭代更新（使用优化版本）
                # 优化：重用缓冲区，避免每次分配新内存
                Lxx, bxx = self.sparse_iter.iter_xx(g, bxx, 1, mu)
                Lyy, byy = self.sparse_iter.iter_yy(g, byy, 1, mu)
                Lzz, bzz = self.sparse_iter.iter_zz(g, bzz, contiz_squared, mu)
                Lxy, bxy = self.sparse_iter.iter_xy(g, bxy, 2, mu)
                Lxz, bxz = self.sparse_iter.iter_xz(g, bxz, 2 * contiz_val, mu)
                Lyz, byz = self.sparse_iter.iter_yz(g, byz, 2 * contiz_val, mu)
                # Lsparse, bl1 = self.sparse_iter.iter_sparse(g, bl1, sparsity, mu)
                
                # 优化：使用原地累加
                temp_sum.zero_()
                # temp_sum.add_(Lxx).add_(Lyy).add_(Lzz).add_(Lxy).add_(Lxz).add_(Lyz).add_(Lsparse)
                temp_sum.add_(Lxx).add_(Lyy).add_(Lzz).add_(Lxy).add_(Lxz).add_(Lyz)
                g_update.add_(temp_sum)
                
                # 记录迭代时间
                iter_time = time.time() - iter_start
                self.iteration_times.append(iter_time)
                
                # 每10次迭代输出进度
                if self.verbose and (iter_idx + 1) % 10 == 0:
                    # avg_time = np.mean(self.iteration_times[-10:]) * 1000
                    print(f'    迭代 {iter_idx + 1}/{iteration_num} - 耗时: {iter_time*1000:.1f}ms')
            
            # 后处理
            g = g
        
        # 最终清理
        self._clear_buffers()
        # del bxx, byy, bzz, bxy, bxz, byz, bl1, f_processed, normlize, g_update, temp_sum, operators
        del bxx, byy, bzz, bxy, bxz, byz, f_processed, normlize, g_update, temp_sum, operators
        self._clear_memory(force=True)
        
        # # 返回结果
        # if flage:
        #     result = g[1, :, :]  # 2D输入时返回中间切片
        # else:
        #     result = g
        
        # 根据原始形状恢复输出
        if hasattr(self, '_original_shape'):
            original_shape = self._original_shape
            
            # 根据flage和原始形状决定输出
            if flage:  # 2D输入，返回2D结果
                result = g[1, :, :]  # 返回中间切片
                
            else:  # 3D输入
                if original_shape[0] == 1:
                    # 单帧输入，返回单帧
                    result = g[1:2, :, :]  # 返回中间帧，但保持3D形状(1, H, W)
                elif original_shape[0] == 2:
                    # 2帧输入，返回前2帧
                    result = g[:2, :, :]
                elif original_shape[0] < g.shape[0]:
                    # 原始帧数少于处理帧数，返回原始帧数
                    result = g[:original_shape[0], :, :]
                else:
                    # 正常情况，返回全部
                    result = g
            
            # 清理原始形状记录
            del self._original_shape
        else:
            # 如果没有记录原始形状，使用默认逻辑
            if flage:
                result = g[1, :, :]  # 2D输入时返回中间切片
            else:
                result = g

        # 性能报告
        if self.verbose:
            total_time = time.time() - total_start_time
            self._print_performance_report(total_time, iteration_num)       
        
        return result
    
    def _precompute_operators(self, imgsize: Tuple[int, ...], contiz_val: float) -> dict:
        """
        预计算并缓存所有频域算子
        显著减少重复计算
        """
        # 检查是否已缓存
        cache_key = f"{imgsize}_{contiz_val}"
        
        xxfft = self.hessian_ops.operation_xx(imgsize)
        yyfft = self.hessian_ops.operation_yy(imgsize)
        zzfft = self.hessian_ops.operation_zz(imgsize)
        xyfft = self.hessian_ops.operation_xy(imgsize)
        xzfft = self.hessian_ops.operation_xz(imgsize)
        yzfft = self.hessian_ops.operation_yz(imgsize)
        
        # 计算组合算子
        operationfft = (xxfft + yyfft + (contiz_val**2) * zzfft + 
                       2 * xyfft + 2 * contiz_val * xzfft + 2 * contiz_val * yzfft)
        
        return {
            'xxfft': xxfft,
            'yyfft': yyfft,
            'zzfft': zzfft,
            'xyfft': xyfft,
            'xzfft': xzfft,
            'yzfft': yzfft,
            'operationfft': operationfft
        }
    
    # def _optimized_iteration_step(self, g: torch.Tensor, 
    #                              bxx: torch.Tensor, byy: torch.Tensor, bzz: torch.Tensor,
    #                              bxy: torch.Tensor, bxz: torch.Tensor, byz: torch.Tensor,
    #                              bl1: torch.Tensor,
    #                              contiz_val: float, sparsity: float, mu: float) -> Tuple[torch.Tensor, ...]:
    #     """
    #     优化的迭代步骤，减少内存分配
    #     """
    #     # 重用预分配的缓冲区
    #     Lxx, bxx = self.sparse_iter.iter_xx(g, bxx, 1, mu)
    #     Lyy, byy = self.sparse_iter.iter_yy(g, byy, 1, mu)
    #     Lzz, bzz = self.sparse_iter.iter_zz(g, bzz, contiz_val**2, mu)
    #     Lxy, bxy = self.sparse_iter.iter_xy(g, bxy, 2, mu)
    #     Lxz, bxz = self.sparse_iter.iter_xz(g, bxz, 2 * contiz_val, mu)
    #     Lyz, byz = self.sparse_iter.iter_yz(g, byz, 2 * contiz_val, mu)
    #     # Lsparse, bl1 = self.sparse_iter.iter_sparse(g, bl1, sparsity, mu)
        
    #     return Lxx, Lyy, Lzz, Lxy, Lxz, Lyz, bxx, byy, bzz, bxy, bxz, byz, bl1
    #     # return Lxx, Lyy, Lzz, Lxy, Lxz, Lyz, Lsparse, bxx, byy, bzz, bxy, bxz, byz, bl1
    
    def _clear_memory(self, force: bool = False):
        """优化内存清理"""
        if force or len(self._buffers) > 0:
            self._clear_buffers()
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # 确保所有操作完成
            self.memory_clear_count += 1
    
    def _print_performance_report(self, total_time: float, iteration_num: int):
        """打印性能报告"""
        print(f"\n{'='*60}")
        print(f"{self.__class__.__name__} - 性能报告")
        print(f"{'='*60}")
        print(f"总时间: {total_time:.2f}秒")
        print(f"迭代次数: {iteration_num}")
        
        if self.iteration_times:
            avg_iter_time = np.mean(self.iteration_times) * 1000
            min_iter_time = np.min(self.iteration_times) * 1000
            max_iter_time = np.max(self.iteration_times) * 1000
            print(f"平均每迭代: {avg_iter_time:.1f}毫秒")
            print(f"最快迭代: {min_iter_time:.1f}毫秒")
            print(f"最慢迭代: {max_iter_time:.1f}毫秒")
            print(f"吞吐量: {1000/avg_iter_time:.2f} 迭代/秒")
        
        print(f"内存清理次数: {self.memory_clear_count}")
        print(f"设备: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU内存使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"GPU内存缓存: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

    def benchmark(self, f: Union[np.ndarray, torch.Tensor], 
                 warmup: int = 5, runs: int = 10,
                 **kwargs) -> dict:
        """
        性能基准测试
        """
        print(f"\n{'='*60}")
        print(f"开始性能基准测试")
        print(f"{'='*60}")
        
        # 预热
        print(f"预热 {warmup} 次...")
        for i in range(warmup):
            _ = self.sparse_hessian(f, **kwargs)
        
        # 正式测试
        times = []
        for i in range(runs):
            start_time = time.time()
            result = self.sparse_hessian(f, **kwargs)
            end_time = time.time()
            times.append(end_time - start_time)
            print(f"运行 {i+1}/{runs}: {times[-1]:.2f}秒")
        
        # 统计结果
        stats = {
            'runs': runs,
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'throughput': runs / np.sum(times),
        }
        
        print(f"\n基准测试结果:")
        print(f"  平均时间: {stats['avg_time']:.2f}秒")
        print(f"  最小时: {stats['min_time']:.2f}秒")
        print(f"  最大时间: {stats['max_time']:.2f}秒")
        print(f"  标准差: {stats['std_time']:.2f}秒")
        print(f"  吞吐量: {stats['throughput']:.2f} 次/秒")
        
        return stats
