import sys
import os
import json
from pathlib import Path
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
import tifffile
from src.default_config import default_config_Hybrid_SIM
from src_BF.BF_SIM import BF_SIM
from src_Hessian.core.Hessian_denoise import Hessian_denoise_SIM
from src_TDV.TDV_denoise import TDV_denoise_SIM
from src_Sparse.iterative_deconv.Sparse_deconvolution import Sparse_deconvolution_SIM
import matplotlib
matplotlib.use('Qt5Agg')  # 使用Qt5作为matplotlib后端

# 获取软件根目录路径
def get_root_directory_():
    """获取软件根目录"""
    # 尝试从当前文件路径向上查找
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # print(current_dir)
    return current_dir

def get_root_directory():
    """获取当前文件的上级目录"""
    # 获取当前文件的绝对路径，然后取其上级目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    return parent_dir

# 修改后的Hybrid_SIM类，支持进度回调
class Hybrid_SIM_:    
    def recon(self, args, progress_callback=None):
        # 创建四个模块，并传入进度回调
        BF = BF_SIM(args, lambda p: progress_callback(p) if progress_callback else None)
        args.bf_result = BF.recon()
        
        if args.Hessian_iteration_number > 0:
            Hessian = Hessian_denoise_SIM(args, lambda p: progress_callback(p) if progress_callback else None)
            args.hessian_result = Hessian.recon()
        else:
            args.hessian_result = args.bf_result
        
        if args.TDV_iteration_number > 0:
            TDV = TDV_denoise_SIM(args, lambda p: progress_callback(p) if progress_callback else None)
            args.tdv_result = TDV.recon()
        else:
            args.tdv_result = args.hessian_result

        if args.Sparse_iteration_number > 0:
            Sparse = Sparse_deconvolution_SIM(args, lambda p: progress_callback(p) if progress_callback else None)
            args.sparse_result = Sparse.recon()
        else:
            args.sparse_result = args.tdv_result

        # 返回所有结果
        return {
            'bf_result': args.bf_result,
            'bf_hessian_result': args.hessian_result,
            'bf_hessian_tdv_result': args.tdv_result,
            'hybrid_result': args.sparse_result
        }

class Hybrid_SIM:
    def __init__(self):
        self._stop_requested = False
        self.bf_sim = None
        self.hessian = None
        self.tdv = None
        self.sparse = None
    
    def recon(self, args, progress_callback=None):
        # 检查是否已请求停止
        if self._stop_requested:
            return {}
        
        # 创建四个模块，并传入进度回调
        self.bf_sim = BF_SIM(args, lambda p: progress_callback(p) if progress_callback else None)
        if self._stop_requested:
            return {}
        
        args.bf_result = self.bf_sim.recon()
        
        if args.Hessian_iteration_number > 0 and not self._stop_requested:
            self.hessian = Hessian_denoise_SIM(args, lambda p: progress_callback(p) if progress_callback else None)
            if self._stop_requested:
                return {'bf_result': args.bf_result}
            
            args.hessian_result = self.hessian.recon()
        else:
            args.hessian_result = args.bf_result
        
        if args.TDV_iteration_number > 0 and not self._stop_requested:
            self.tdv = TDV_denoise_SIM(args, lambda p: progress_callback(p) if progress_callback else None)
            if self._stop_requested:
                return {
                    'bf_result': args.bf_result,
                    'bf_hessian_result': args.hessian_result
                }
            
            args.tdv_result = self.tdv.recon()
        else:
            args.tdv_result = args.hessian_result

        if args.Sparse_iteration_number > 0 and not self._stop_requested:
            self.sparse = Sparse_deconvolution_SIM(args, lambda p: progress_callback(p) if progress_callback else None)
            if self._stop_requested:
                return {
                    'bf_result': args.bf_result,
                    'bf_hessian_result': args.hessian_result,
                    'bf_hessian_tdv_result': args.tdv_result
                }
            
            args.sparse_result = self.sparse.recon()
        else:
            args.sparse_result = args.tdv_result

        # 返回所有结果
        return {
            'bf_result': args.bf_result,
            'bf_hessian_result': args.hessian_result,
            'bf_hessian_tdv_result': args.tdv_result,
            'hybrid_result': args.sparse_result
        }
    
    def stop(self):
        """停止重建过程"""
        self._stop_requested = True
        
        # 停止所有模块
        if self.sparse:
            try:
                self.sparse.stop()
            except:
                pass
        
        if self.tdv:
            try:
                self.tdv.stop()
            except:
                pass
        
        if self.hessian:
            try:
                self.hessian.stop()
            except:
                pass
        
        if self.bf_sim:
            try:
                self.bf_sim.stop()
            except:
                pass

class ReconstructionThread(QThread):
    """重建线程, 避免阻塞GUI"""
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._is_running = True
        self._stop_requested = False  # 添加停止请求标志
        self.hybrid_sim = None  # 存储Hybrid_SIM实例
        
    def run_(self):
        try:
            # 创建Hybrid_SIM实例
            hybrid_sim = Hybrid_SIM()
            
            # 执行重建，传入进度回调函数
            results = hybrid_sim.recon(self.args, self.update_progress)
            if self._is_running:
                self.result_signal.emit(results)
        except Exception as e:
            if self._is_running:
                self.error_signal.emit(str(e))
        finally:
            if self._is_running:
                self.finished_signal.emit()
    
    def run(self):
        try:
            # 创建Hybrid_SIM实例
            self.hybrid_sim = Hybrid_SIM()
            
            # 执行重建，传入进度回调函数
            results = self.hybrid_sim.recon(self.args, self.update_progress)
            if self._is_running and not self._stop_requested:
                self.result_signal.emit(results)
        except Exception as e:
            if self._is_running and not self._stop_requested:
                self.error_signal.emit(str(e))
        finally:
            if self._is_running:
                self.finished_signal.emit()

    def update_progress_(self, value):
        """更新进度"""
        if self._is_running:
            self.progress_signal.emit(value)
    
    def update_progress(self, value):
        """更新进度"""
        if self._is_running and not self._stop_requested:
            self.progress_signal.emit(value)

    def stop_(self):
        """停止线程"""
        self._is_running = False

    def stop(self):
        """停止线程"""
        self._stop_requested = True
        self._is_running = False
        
        # 停止Hybrid_SIM重建
        if self.hybrid_sim:
            try:
                self.hybrid_sim.stop()
            except:
                pass

    def is_stop_requested(self):
        """检查是否已请求停止"""
        return self._stop_requested           


class ParameterWidget(QWidget):
    """单个参数控件"""
    def __init__(self, param_name, param_type, default_value, tooltip="", unit="", parent=None):
        super().__init__(parent)
        self.param_name = param_name
        self.param_type = param_type
        self.default_value = default_value
        self.unit = unit
        
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 修改：替换标签文本中的特定缩写
        label_text = param_name.replace('_', ' ').title()
        # 替换特定缩写
        label_text = label_text.replace('Na', 'NA')
        label_text = label_text.replace('Bf', 'BF')
        label_text = label_text.replace('Psf', 'PSF')
        label_text = label_text.replace('Otf', 'OTF')
        label_text = label_text.replace('Tdv', 'TDV')
        label_text = label_text.replace('Sim', 'SIM')
        
        # 添加单位到标签
        if unit:
            label_text += f" ({unit})"
            
        self.label = QLabel(label_text)
        self.label.setToolTip(tooltip)
        self.label.setFixedWidth(200)
        self.label.setFont(QFont("Arial", 9))
        
        if param_type == "file":
            self.widget = QLineEdit(str(default_value))
            self.widget.setFont(QFont("Arial", 9))
            browse_btn = QPushButton("Browse")
            browse_btn.setFont(QFont("Arial", 9))
            browse_btn.clicked.connect(self.browse_file)
            layout.addWidget(self.label)
            layout.addWidget(self.widget)
            layout.addWidget(browse_btn)
        elif param_type == "float":
            self.widget = QDoubleSpinBox()
            self.widget.setRange(0, 10000)
            
            # 特殊处理：对于波长和像素尺寸，显示为nm单位的整数值
            if param_name in ['SIM_excitation_wavelength', 'SIM_emission_wavelength', 'SIM_Raw_pixel_size']:
                # 默认值是以米为单位的，转换为nm显示
                nm_value = default_value * 1e9
                self.widget.setValue(float(nm_value))
                self.widget.setDecimals(2)  # 显示为整数
            else:
                self.widget.setValue(float(default_value))
                self.widget.setDecimals(2)  # 默认保留2位小数
            
            self.widget.setFont(QFont("Arial", 9))
            layout.addWidget(self.label)
            layout.addWidget(self.widget)
        elif param_type == "int":
            self.widget = QSpinBox()
            self.widget.setRange(0, 10000)
            self.widget.setValue(int(default_value))
            self.widget.setFont(QFont("Arial", 9))
            layout.addWidget(self.label)
            layout.addWidget(self.widget)
        elif param_type == "str":
            self.widget = QLineEdit(str(default_value))
            self.widget.setFont(QFont("Arial", 9))
            layout.addWidget(self.label)
            layout.addWidget(self.widget)
        elif param_type == "bool":
            self.widget = QCheckBox()
            self.widget.setChecked(bool(default_value))
            layout.addWidget(self.label)
            layout.addWidget(self.widget)
        
        reset_btn = QPushButton("Reset")
        reset_btn.setFixedWidth(60)
        reset_btn.setFont(QFont("Arial", 9))
        reset_btn.clicked.connect(self.reset_value)
        layout.addWidget(reset_btn)
        
        self.setLayout(layout)
    
    def browse_file(self):
        # 获取软件根目录
        root_dir = get_root_directory()
        # 根据参数名称设置默认文件夹
        if self.param_name == 'SIM_BF_PSF_path':
            default_dir = os.path.join(root_dir, 'src_Hybrid/src_Optics/BF_PSF')
            file_filter = "TIFF Files (*.tif *.tiff);;All Files (*)"
        elif self.param_name == 'SIM_Recon_OTF_path':
            default_dir = os.path.join(root_dir, 'src_Hybrid/src_Optics')
            file_filter = "TIFF Files (*.tif *.tiff);;All Files (*)"
        elif self.param_name == 'TDV_model_path':
            default_dir = os.path.join(root_dir, 'src_Hybrid/src_model')
            file_filter = "Model Files (*.pth);;All Files (*)"
        else:
            # 对于其他文件，使用当前设置的路径的目录或根目录
            current_path = self.widget.text()
            if current_path and os.path.exists(current_path):
                default_dir = os.path.dirname(current_path)
            else:
                default_dir = root_dir
            file_filter = "All Files (*);;TIFF Files (*.tif *.tiff)"
        
        # 确保默认目录存在，如果不存在则使用根目录
        if not os.path.exists(default_dir):
            default_dir = root_dir
        
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select File", default_dir, file_filter
        )
        
        if filename:
            # 尝试将绝对路径转换为相对于软件根目录的相对路径
            try:
                # 规范化路径（确保使用相同的路径分隔符）
                filename_norm = os.path.normpath(filename)
                root_dir_norm = os.path.normpath(root_dir)
                
                # 使用大小写不敏感的比较（Windows系统需要）
                # 在Windows上，os.path.normpath不会改变驱动器字母的大小写
                # 所以我们需要使用lower()或casefold()进行大小写不敏感的比较
                filename_lower = filename_norm.lower()
                root_dir_lower = root_dir_norm.lower()
                
                # 检查文件路径是否以根目录开头（忽略大小写）
                if filename_lower.startswith(root_dir_lower):
                    # 计算相对路径
                    rel_path = os.path.relpath(filename_norm, root_dir_norm)
                    # 将路径分隔符转换为正斜杠（跨平台兼容性）
                    rel_path = rel_path.replace('\\', '/')
                    # 确保以./开头（相对路径）
                    if not rel_path.startswith('./') and not rel_path.startswith('../'):
                        rel_path = './' + rel_path
                    self.widget.setText(rel_path)
                else:
                    # 如果文件不在根目录下，保持绝对路径
                    self.widget.setText(filename)
            except Exception as e:
                # 如果转换失败，使用原始路径
                print(f"Error converting path to relative: {e}")
                self.widget.setText(filename)

    def browse_file_(self):
        # 获取软件根目录
        root_dir = get_root_directory()
        print(root_dir)
        # 根据参数名称设置默认文件夹
        if self.param_name == 'SIM_BF_PSF_path':
            default_dir = os.path.join(root_dir, 'src_Hybrid/src_Optics/BF_PSF')
            file_filter = "TIFF Files (*.tif *.tiff);;All Files (*)"
        elif self.param_name == 'SIM_Recon_OTF_path':
            default_dir = os.path.join(root_dir, 'src_Hybrid/src_Optics')
            file_filter = "TIFF Files (*.tif *.tiff);;All Files (*)"
        elif self.param_name == 'TDV_model_path':
            default_dir = os.path.join(root_dir, 'src_Hybrid/src_model')
            file_filter = "Model Files (*.pth);;All Files (*)"
        else:
            # 对于其他文件，使用当前设置的路径的目录或根目录
            current_path = self.widget.text()
            if current_path and os.path.exists(current_path):
                default_dir = os.path.dirname(current_path)
            else:
                default_dir = root_dir
            file_filter = "All Files (*);;TIFF Files (*.tif *.tiff)"
        
        # 确保默认目录存在，如果不存在则使用根目录
        if not os.path.exists(default_dir):
            default_dir = root_dir
        
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select File", default_dir, file_filter
        )
        print(filename)
        if filename:
            self.widget.setText(filename)
    
    def reset_value(self):
        if self.param_type == "file" or self.param_type == "str":
            self.widget.setText(str(self.default_value))
        elif self.param_type == "float":
            # 特殊处理：对于波长和像素尺寸，显示为nm单位的整数值
            if self.param_name in ['SIM_excitation_wavelength', 'SIM_emission_wavelength', 'SIM_Raw_pixel_size']:
                # 默认值是以米为单位的，转换为nm显示
                nm_value = self.default_value * 1e9
                self.widget.setValue(float(nm_value))
            else:
                self.widget.setValue(float(self.default_value))
        elif self.param_type == "int":
            self.widget.setValue(int(self.default_value))
        elif self.param_type == "bool":
            self.widget.setChecked(bool(self.default_value))
    
    def get_value(self):
        if self.param_type == "file" or self.param_type == "str":
            return str(self.widget.text())
        elif self.param_type == "float":
            value = float(self.widget.value())
            # 特殊处理：对于波长和像素尺寸，将nm转换为米
            if self.param_name in ['SIM_excitation_wavelength', 'SIM_emission_wavelength', 'SIM_Raw_pixel_size']:
                return value * 1e-9  # 转换为米
            return value
        elif self.param_type == "int":
            return int(self.widget.value())
        elif self.param_type == "bool":
            return bool(self.widget.isChecked())
    
    def set_value(self, value):
        """设置参数值"""
        if self.param_type == "file" or self.param_type == "str":
            self.widget.setText(str(value))
        elif self.param_type == "float":
            # 特殊处理：对于波长和像素尺寸，将米转换为nm显示
            if self.param_name in ['SIM_excitation_wavelength', 'SIM_emission_wavelength', 'SIM_Raw_pixel_size']:
                # 确保value是以米为单位的
                if isinstance(value, float) and value < 1e-6:  # 假设小于1微米的是以米为单位的
                    nm_value = value * 1e9
                else:
                    nm_value = float(value)  # 假设已经是nm单位
                self.widget.setValue(float(nm_value))
            else:
                self.widget.setValue(float(value))
        elif self.param_type == "int":
            self.widget.setValue(int(value))
        elif self.param_type == "bool":
            self.widget.setChecked(bool(value))
    
    def set_enabled(self, enabled):
        """启用/禁用控件"""
        self.label.setEnabled(enabled)
        self.widget.setEnabled(enabled)
        # 查找并禁用Reset按钮
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i)
            widget = item.widget()
            if widget and isinstance(widget, QPushButton) and widget.text() == "Reset":
                widget.setEnabled(enabled)
            elif widget and isinstance(widget, QPushButton) and widget.text() == "Browse":
                widget.setEnabled(enabled)

class ModuleWidget(QGroupBox):
    """单个模块的控件组"""
    def __init__(self, title, parameters, parent=None):
        # 修改：替换标题文本中的特定缩写
        title = title.replace('Bf', 'BF')
        title = title.replace('Tdv', 'TDV')
        
        super().__init__(title, parent)
        self.parameters = {}
        
        # 设置标题字体
        font = QFont("Arial", 10, QFont.Bold)
        self.setFont(font)
        
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        for param_name, param_config in parameters.items():
            param_widget = ParameterWidget(
                param_name, 
                param_config['type'],
                param_config['default'],
                param_config.get('tooltip', ''),
                param_config.get('unit', '')
            )
            self.parameters[param_name] = param_widget
            layout.addWidget(param_widget)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def get_values(self):
        return {name: widget.get_value() for name, widget in self.parameters.items()}
    
    def set_enabled(self, enabled):
        """启用/禁用模块内所有控件"""
        for param_widget in self.parameters.values():
            param_widget.set_enabled(enabled)

class ImageDisplayWidget(QLabel):
    """图像显示控件，支持3D数据和鼠标交互，对整个3D体积归一化，使用复制像素插值"""
    # 添加缩放变化信号
    zoom_changed = pyqtSignal(float, QPoint)  # 添加鼠标位置参数
    pan_changed = pyqtSignal(QPoint)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_image = None  # 3D数据
        self.original_qimage = None  # QImage对象
        self.current_frame = 0  # 当前显示的帧索引
        self.total_frames = 0   # 总帧数
        
        # 添加用于全局归一化的变量
        self.global_min = None
        self.global_max = None
        
        # 添加显示区间变量
        self.display_min = None
        self.display_max = None
        
        self.zoom_factor = 1.0
        self.initial_zoom_factor = 1.0  # 初始缩放因子
        self.min_zoom_factor = 1.0  # 最小缩放因子
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        self.last_pan_pos = QPoint(0, 0)
        self.last_valid_pan_offset = QPoint(0, 0)  # 上一次有效的平移偏移
        self.interaction_enabled = True  # 交互功能是否启用
        
        # 设置固定大小策略
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(300, 300)  # 固定显示区域大小
        self.setStyleSheet("""
            QLabel {
                border: 1px solid #ccc;
                background-color: #f0f0f0;
                padding: 2px;
            }
        """)
        
        # 启用鼠标跟踪
        self.setMouseTracking(True)
    
    def set_image(self, image_array, frame_idx=0, normalize=True, use_display_range=False):
        """设置图像数据（支持2D和3D数据），对整个3D体积归一化"""
        if image_array is None:
            self.clear()
            self.original_image = None
            self.original_qimage = None
            self.global_min = None
            self.global_max = None
            self.display_min = None
            self.display_max = None
            return
        
        self.original_image = image_array
        
        # 确定数据维度
        if len(image_array.shape) == 3:  # 3D数据
            self.total_frames = image_array.shape[0]
            if frame_idx >= self.total_frames:
                frame_idx = self.total_frames - 1
            self.current_frame = frame_idx
            frame_data = image_array[frame_idx]
        else:  # 2D数据
            self.total_frames = 1
            self.current_frame = 0
            frame_data = image_array
        
        # 计算全局最小值和最大值（对整个3D体积进行归一化）
        if normalize and self.original_image is not None:
            # 计算整个3D体积的最小值和最大值
            if self.global_min is None or self.global_max is None:
                if len(self.original_image.shape) == 3:  # 3D数据
                    self.global_min = self.original_image.min()
                    self.global_max = self.original_image.max()
                else:  # 2D数据
                    self.global_min = self.original_image.min()
                    self.global_max = self.original_image.max()
            
            # 初始化显示区间
            if self.display_min is None:
                self.display_min = self.global_min
            if self.display_max is None:
                self.display_max = self.global_max
            
            # 使用显示区间进行归一化
            if use_display_range:
                min_val = self.display_min
                max_val = self.display_max
            else:
                min_val = self.global_min
                max_val = self.global_max
            
            if max_val - min_val > 1e-10:
                # 归一化当前帧，使用显示区间
                img_normalized = (frame_data - min_val) / (max_val - min_val)
            else:
                img_normalized = np.zeros_like(frame_data)
            
            # 确保值在[0, 1]范围内
            img_normalized = np.clip(img_normalized, 0, 1)
            img_8bit = (img_normalized * 255).astype(np.uint8)
        else:
            img_8bit = frame_data.astype(np.uint8)
        
        # 转换为QImage
        height, width = img_8bit.shape
        if len(img_8bit.shape) == 2:  # 灰度图
            bytes_per_line = width
            qimage = QImage(img_8bit.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            # 复制QImage数据以防止原始数据被释放
            self.original_qimage = qimage.copy()
        else:  # RGB图
            bytes_per_line = 3 * width
            qimage = QImage(img_8bit.data, width, height, bytes_per_line, QImage.Format_RGB888)
            # 复制QImage数据以防止原始数据被释放
            self.original_qimage = qimage.copy()
        
        # 计算合适的初始缩放因子，使图像完整显示在显示区域内
        self.calculate_initial_zoom()
        
        self.update_display()
    
    def calculate_initial_zoom(self):
        """计算合适的初始缩放因子，使图像最长边与显示窗口的边界重合"""
        if self.original_qimage is None or self.original_qimage.isNull():
            return
        
        # 获取图像和显示区域的尺寸
        image_width = self.original_qimage.width()
        image_height = self.original_qimage.height()
        display_width = self.width()
        display_height = self.height()
        
        # 计算使图像最长边与窗口边重合的缩放因子
        width_ratio = display_width / image_width
        height_ratio = display_height / image_height
        
        # 取较小的比例，确保图像完整显示在显示区域内
        self.initial_zoom_factor = min(width_ratio, height_ratio)
        
        # 设置最小缩放因子为初始缩放因子
        # 这样当图像缩到最小时，其最长边与窗口重合，不能再缩小了
        self.min_zoom_factor = self.initial_zoom_factor
        
        # 设置初始缩放因子
        self.zoom_factor = self.initial_zoom_factor
        
        # 重置平移偏移
        self.pan_offset = QPoint(0, 0)
    
    def update_display(self):
        """更新显示，使用复制像素插值（最近邻插值）"""
        # 如果没有图像数据，直接返回
        if self.original_qimage is None or self.original_qimage.isNull():
            return
        
        # 确保缩放因子不小于最小值
        if self.zoom_factor < self.min_zoom_factor:
            self.zoom_factor = self.min_zoom_factor
        
        # 限制平移偏移，确保图像不会移出显示区域
        self.limit_pan_offset()
        
        # 创建缩放后的图像（使用最近邻插值）
        scaled_width = int(self.original_qimage.width() * self.zoom_factor)
        scaled_height = int(self.original_qimage.height() * self.zoom_factor)
        
        # 使用最近邻插值（复制像素）- Qt.FastTransformation
        scaled_qimage = self.original_qimage.scaled(
            scaled_width, scaled_height,
            Qt.IgnoreAspectRatio,  # 不保持宽高比，按指定尺寸缩放
            Qt.FastTransformation  # 使用复制像素插值（最近邻插值）
        )
        
        # 创建带偏移的图像
        result_image = QImage(self.size(), QImage.Format_ARGB32)
        result_image.fill(Qt.transparent)
        
        painter = QPainter(result_image)
        
        # 计算绘制位置（考虑偏移）
        x_offset = (self.width() - scaled_qimage.width()) // 2 + self.pan_offset.x()
        y_offset = (self.height() - scaled_qimage.height()) // 2 + self.pan_offset.y()
        
        painter.drawImage(x_offset, y_offset, scaled_qimage)
        painter.end()
        
        self.setPixmap(QPixmap.fromImage(result_image))
    
    def limit_pan_offset(self):
        """限制平移偏移，确保图像不会移出显示区域"""
        if self.original_qimage is None or self.original_qimage.isNull():
            return
        
        # 计算缩放后的图像尺寸
        scaled_width = int(self.original_qimage.width() * self.zoom_factor)
        scaled_height = int(self.original_qimage.height() * self.zoom_factor)
        
        # 计算显示区域的中心点
        center_x = self.width() // 2
        center_y = self.height() // 2
        
        # 计算图像在显示区域中的边界
        left_bound = center_x - scaled_width // 2
        right_bound = center_x + scaled_width // 2
        top_bound = center_y - scaled_height // 2
        bottom_bound = center_y + scaled_height // 2
        
        # 计算最大允许的平移偏移
        # 图像不能移出显示区域
        max_x_offset = 0
        max_y_offset = 0
        
        if scaled_width > self.width():
            # 图像宽度大于显示区域，允许平移
            max_x_offset = (scaled_width - self.width()) // 2
        else:
            # 图像宽度小于或等于显示区域，不允许水平平移
            max_x_offset = 0
        
        if scaled_height > self.height():
            # 图像高度大于显示区域，允许平移
            max_y_offset = (scaled_height - self.height()) // 2
        else:
            # 图像高度小于或等于显示区域，不允许垂直平移
            max_y_offset = 0
        
        # 限制平移偏移在允许范围内
        self.pan_offset.setX(max(-max_x_offset, min(max_x_offset, self.pan_offset.x())))
        self.pan_offset.setY(max(-max_y_offset, min(max_y_offset, self.pan_offset.y())))
    
    def set_display_range(self, display_min, display_max):
        """设置显示区间"""
        if self.original_image is None:
            return
        
        # 确保显示区间在全局范围内
        display_min = max(display_min, self.global_min)
        display_max = min(display_max, self.global_max)
        
        # 确保最小值小于最大值
        if display_min >= display_max:
            return
        
        self.display_min = display_min
        self.display_max = display_max
        
        # 重新显示当前帧，使用显示区间
        self.apply_display_range_to_current_frame()
    
    def apply_display_range_to_current_frame(self):
        """将显示区间应用到当前帧"""
        if self.original_image is None:
            return
        
        # 获取当前帧数据
        if len(self.original_image.shape) == 3:
            frame_data = self.original_image[self.current_frame]
        else:
            frame_data = self.original_image
        
        # 使用显示区间进行归一化
        if self.display_max - self.display_min > 1e-10:
            img_normalized = (frame_data - self.display_min) / (self.display_max - self.display_min)
        else:
            img_normalized = np.zeros_like(frame_data)
        
        # 确保值在[0, 1]范围内
        img_normalized = np.clip(img_normalized, 0, 1)
        img_8bit = (img_normalized * 255).astype(np.uint8)
        
        # 转换为QImage
        height, width = img_8bit.shape
        if len(img_8bit.shape) == 2:  # 灰度图
            bytes_per_line = width
            qimage = QImage(img_8bit.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            # 复制QImage数据以防止原始数据被释放
            self.original_qimage = qimage.copy()
        else:  # RGB图
            bytes_per_line = 3 * width
            qimage = QImage(img_8bit.data, width, height, bytes_per_line, QImage.Format_RGB888)
            # 复制QImage数据以防止原始数据被释放
            self.original_qimage = qimage.copy()
        
        self.update_display()
    
    def get_global_range(self):
        """获取全局范围"""
        return self.global_min, self.global_max
    
    def set_zoom_and_pan(self, zoom_factor, pan_offset):
        """设置缩放和平移（用于联动）"""
        if not self.interaction_enabled:
            return
        self.zoom_factor = max(self.min_zoom_factor, zoom_factor)
        self.pan_offset = QPoint(pan_offset)
        self.update_display()
    
    def zoom_with_mouse_center(self, factor, mouse_pos):
        """以鼠标位置为中心缩放"""
        if not self.interaction_enabled:
            return
        if self.original_qimage is None or self.original_qimage.isNull():
            return
        
        # 保存旧的缩放因子和平移偏移
        old_zoom = self.zoom_factor
        old_pan = QPoint(self.pan_offset)
        
        # 计算新的缩放因子
        new_zoom = old_zoom * factor
        
        # 确保不小于最小缩放因子
        if new_zoom < self.min_zoom_factor:
            new_zoom = self.min_zoom_factor
            factor = new_zoom / old_zoom
        
        # 如果缩放因子没有变化，直接返回
        if abs(factor - 1.0) < 1e-6:
            return
        
        # 计算鼠标在图像控件中的相对位置
        rel_pos = mouse_pos
        
        # 计算图像显示位置
        scaled_width = int(self.original_qimage.width() * old_zoom)
        scaled_height = int(self.original_qimage.height() * old_zoom)
        
        # 图像在控件中的显示区域
        image_rect = QRect(
            (self.width() - scaled_width) // 2 + old_pan.x(),
            (self.height() - scaled_height) // 2 + old_pan.y(),
            scaled_width,
            scaled_height
        )
        
        # 如果鼠标不在图像区域内，以图像中心为中心缩放
        if not image_rect.contains(rel_pos):
            # 以图像中心为中心缩放
            self.zoom_factor = new_zoom
            # 发射缩放变化信号
            self.zoom_changed.emit(new_zoom, self.pan_offset)
        else:
            # 计算鼠标相对于图像的位置（0-1之间）
            rel_x_in_image = (rel_pos.x() - image_rect.left()) / scaled_width
            rel_y_in_image = (rel_pos.y() - image_rect.top()) / scaled_height
            
            # 更新缩放因子
            self.zoom_factor = new_zoom
            
            # 计算新的平移偏移，使鼠标位置在缩放后保持不变
            new_scaled_width = int(self.original_qimage.width() * new_zoom)
            new_scaled_height = int(self.original_qimage.height() * new_zoom)
            
            # 计算新的图像显示位置
            new_image_left = rel_pos.x() - rel_x_in_image * new_scaled_width
            new_image_top = rel_pos.y() - rel_y_in_image * new_scaled_height
            
            # 计算新的平移偏移
            new_pan_x = new_image_left - (self.width() - new_scaled_width) // 2
            new_pan_y = new_image_top - (self.height() - new_scaled_height) // 2
            
            self.pan_offset = QPoint(int(new_pan_x), int(new_pan_y))
            
            # 限制平移偏移
            self.limit_pan_offset()
            
            # 发射缩放变化信号（包含鼠标位置）
            self.zoom_changed.emit(new_zoom, self.pan_offset)
        
        # 更新显示
        self.update_display()
    
    def zoom_in_with_mouse(self, mouse_pos):
        """以鼠标位置为中心放大图像"""
        self.zoom_with_mouse_center(1.2, mouse_pos)
    
    def zoom_out_with_mouse(self, mouse_pos):
        """以鼠标位置为中心缩小图像"""
        self.zoom_with_mouse_center(1/1.2, mouse_pos)
    
    def reset_view(self):
        """重置视图到初始状态"""
        if not self.interaction_enabled:
            return
        if self.original_qimage is None or self.original_qimage.isNull():
            return
        
        old_zoom = self.zoom_factor
        # 重置到初始缩放因子
        self.zoom_factor = self.initial_zoom_factor
        self.pan_offset = QPoint(0, 0)
        
        # 发射变化信号
        if old_zoom != self.zoom_factor:
            self.zoom_changed.emit(self.zoom_factor, self.pan_offset)
        self.pan_changed.emit(self.pan_offset)
        
        self.update_display()
    
    def clear_display(self):
        """清空显示"""
        self.clear()
        self.original_image = None
        self.original_qimage = None
        self.global_min = None
        self.global_max = None
        self.display_min = None
        self.display_max = None
        self.current_frame = 0
        self.total_frames = 0
        self.zoom_factor = 1.0
        self.initial_zoom_factor = 1.0
        self.min_zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        self.last_pan_pos = QPoint(0, 0)
        self.last_valid_pan_offset = QPoint(0, 0)
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if not self.interaction_enabled:
            return
        if event.button() == Qt.LeftButton and self.zoom_factor > self.min_zoom_factor:
            self.is_panning = True
            self.last_pan_pos = event.pos()
            self.last_valid_pan_offset = QPoint(self.pan_offset)
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if not self.interaction_enabled:
            self.setCursor(Qt.ArrowCursor)
            return
        if self.is_panning:
            delta = event.pos() - self.last_pan_pos
            new_pan_offset = self.last_valid_pan_offset + delta
            
            # 临时设置平移偏移以检查限制
            temp_pan_offset = QPoint(self.pan_offset)
            self.pan_offset = QPoint(new_pan_offset)
            self.limit_pan_offset()
            
            # 如果平移有效，更新显示并发射信号
            if self.pan_offset != temp_pan_offset:
                self.update_display()
                self.pan_changed.emit(self.pan_offset)
            
            # 恢复有效偏移
            self.last_valid_pan_offset = QPoint(self.pan_offset)
            self.last_pan_pos = event.pos()
        
        # 如果放大状态，显示可拖动手型光标
        if self.zoom_factor > self.min_zoom_factor and not self.is_panning:
            self.setCursor(Qt.OpenHandCursor)
        elif self.zoom_factor <= self.min_zoom_factor:
            self.setCursor(Qt.ArrowCursor)
            
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if not self.interaction_enabled:
            return
        if event.button() == Qt.LeftButton and self.is_panning:
            self.is_panning = False
            if self.zoom_factor > self.min_zoom_factor:
                self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)
    
    def wheelEvent(self, event):
        """鼠标滚轮事件"""
        if not self.interaction_enabled:
            return
        # 如果没有图像数据，直接返回
        if self.original_qimage is None or self.original_qimage.isNull():
            return
        
        # 获取鼠标位置
        mouse_pos = event.pos()
        
        # 缩放图像
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_in_with_mouse(mouse_pos)
        else:
            self.zoom_out_with_mouse(mouse_pos)
        
        super().wheelEvent(event)
    
    def set_interaction_enabled(self, enabled):
        """设置交互功能是否启用"""
        self.interaction_enabled = enabled
        if not enabled:
            self.setCursor(Qt.ArrowCursor)
            self.is_panning = False
        elif self.zoom_factor > self.min_zoom_factor:
            self.setCursor(Qt.OpenHandCursor)

class ImageDisplayContainer(QWidget):
    """图像显示容器，包含标题和图像显示控件"""
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        
        # 设置固定大小策略
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)  # 减少边距
        layout.setSpacing(5)
        
        # 标题
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 11, QFont.Bold))
        title_label.setFixedHeight(20)  # 固定标题高度
        title_label.setStyleSheet("margin: 0px;")
        layout.addWidget(title_label)
        
        # 图像显示控件
        self.image_widget = ImageDisplayWidget()
        layout.addWidget(self.image_widget)
        
        self.setLayout(layout)
        
        # 计算并设置固定大小
        # 标题高度(20) + 图像显示控件高度(300) + 布局边距和间距
        total_height = 20 + 300 + 5 + 5 + 5 + 5
        total_width = 300 + 5 + 5
        self.setFixedSize(total_width, total_height)
    
    def set_image(self, image_array, frame_idx=0, normalize=True, use_display_range=False):
        """设置图像数据"""
        self.image_widget.set_image(image_array, frame_idx, normalize, use_display_range)
    
    def set_display_range(self, display_min, display_max):
        """设置显示区间"""
        self.image_widget.set_display_range(display_min, display_max)
    
    def get_global_range(self):
        """获取全局范围"""
        return self.image_widget.get_global_range()
    
    def set_interaction_enabled(self, enabled):
        """设置交互功能是否启用"""
        self.image_widget.set_interaction_enabled(enabled)
    
    def clear_display(self):
        """清空显示"""
        self.image_widget.clear_display()

class SIMReconstructionGUI(QMainWindow):
    # 进度更新信号
    progress_update_signal = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self.args = default_config_Hybrid_SIM()  # 使用模拟的Args类
        self.results = {}
        self.image_displays = []  # 存储所有图像显示控件
        self.current_frame = 0
        self.reconstruction_thread = None  # 重建线程
        self.global_min_val = 0  # 全局最小值
        self.global_max_val = 1  # 全局最大值
        self.display_min_val = 0  # 显示区间最小值
        self.display_max_val = 1  # 显示区间最大值
        
        # 视频播放相关变量
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.next_video_frame)
        self.is_video_playing = False
        self.video_speed = 100  # 默认播放速度，单位ms（10fps）
        
        # 添加标志来追踪是否由视频播放触发的帧变化
        self._is_video_changing_frame = False
        
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Hybrid-SIM Reconstruction")
        # 修改：设置窗口大小为1200x900
        self.setGeometry(100, 100, 1200, 900)
        self.setMinimumSize(1200, 900)
        self.setMaximumSize(1200, 900)
        
        # 设置主窗口字体
        self.setFont(QFont("Arial", 9))
        
        # 主窗口布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 左侧参数面板
        left_panel = QWidget()
        left_panel.setFixedWidth(500)
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)  # 增加间距
        
        # 标题
        title_label = QLabel("Hybrid-SIM Reconstruction Parameters")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setStyleSheet("padding: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title_label)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setFont(QFont("Arial", 9))
        
        # BF-SIM 模块 - 根据要求调整参数
        bf_sim_params = {
            'SIM_raw_data_path': {
                'type': 'file',
                'default': './Demo_Recon_SIM.tif',
                'tooltip': 'Path to SIM raw data TIFF file'
            },
            'SIM_excitation_NA': {
                'type': 'float',
                'default': 1.4,
                'tooltip': 'SIM excitation numerical aperture',
                'unit': ''
            },
            'SIM_emission_NA': {
                'type': 'float',
                'default': 1.4,
                'tooltip': 'SIM emission numerical aperture',
                'unit': ''
            },
            'SIM_excitation_wavelength': {
                'type': 'float',
                'default': 488e-9,
                'tooltip': 'Excitation wavelength',
                'unit': 'nm'
            },
            'SIM_emission_wavelength': {
                'type': 'float',
                'default': 525e-9,
                'tooltip': 'Emission wavelength',
                'unit': 'nm'
            },
            'SIM_Raw_pixel_size': {
                'type': 'float',
                'default': 43.33e-9,
                'tooltip': 'Raw pixel size',
                'unit': 'nm'
            },
            'SIM_Recon_OTF_path': {
                'type': 'file',
                'default': 'Simulation',
                'tooltip': 'Path to Wiener OTF TIFF file'
            }
        }
        self.bf_sim_widget = ModuleWidget("BF-SIM Parameters", bf_sim_params)
        self.tab_widget.addTab(self.bf_sim_widget, "BF-SIM")
        
        # Hessian Denoise 模块
        hessian_params = {
            'Hessian_fidelity': {
                'type': 'float',
                'default': 100,
                'tooltip': 'Hessian fidelity parameter',
                'unit': ''
            },
            'Hessian_Z_continuity': {
                'type': 'float',
                'default': 2,
                'tooltip': 'Hessian z/t continuity parameter',
                'unit': ''
            },
            'Hessian_iteration_number': {
                'type': 'int',
                'default': 100,
                'tooltip': 'Number of Hessian iterations'
            }
        }
        self.hessian_widget = ModuleWidget("Hessian Denoise Parameters", hessian_params)
        self.tab_widget.addTab(self.hessian_widget, "Hessian Denoise")
        
        # TDV Denoise 模块
        tdv_params = {
            'TDV_model_path': {
                'type': 'file',
                'default': './src_Hybrid/src_model/TDV_150XSIM_Actin.pth',
                'tooltip': 'Path to TDV model file'
            },
            'TDV_fidelity': {
                'type': 'float',
                'default': 100,
                'tooltip': 'TDV fidelity parameter',
                'unit': ''
            },
            'TDV_offset': {
                'type': 'float',
                'default': 0.1,
                'tooltip': 'TDV offset parameter',
                'unit': ''
            },
            'TDV_weight': {
                'type': 'float',
                'default': 1,
                'tooltip': 'TDV weight parameter',
                'unit': ''
            },
            'TDV_iteration_number': {
                'type': 'int',
                'default': 30,
                'tooltip': 'Number of TDV iterations'
            }
        }
        self.tdv_widget = ModuleWidget("TDV Denoise Parameters", tdv_params)
        self.tab_widget.addTab(self.tdv_widget, "TDV Denoise")
        
        # Sparse Deconvolution 模块
        sparse_params = {            
            'Sparse_NA': {
                'type': 'float',
                'default': 3,
                'tooltip': 'Sparse deconvolution NA parameter',
                'unit': ''
            },
            'Sparse_offset': {
                'type': 'float',
                'default': 0.2,
                'tooltip': 'Sparse deconvolution offset parameter',
                'unit': ''
            },
            'Sparse_iteration_number': {
                'type': 'int',
                'default': 25,
                'tooltip': 'Number of sparse deconvolution iterations'
            }
        }
        self.sparse_widget = ModuleWidget("Sparse Deconvolution Parameters", sparse_params)
        self.tab_widget.addTab(self.sparse_widget, "Sparse Deconvolution")
        
        left_layout.addWidget(self.tab_widget)
        
        # 设备选择
        device_layout = QHBoxLayout()
        device_label = QLabel("Device:")
        device_label.setFont(QFont("Arial", 9))
        
        # 自动检测可用设备
        self.device_combo = QComboBox()
        self.device_combo.setFont(QFont("Arial", 9))
        
        # 检测可用的设备
        available_devices = self.detect_available_devices()
        for device in available_devices:
            self.device_combo.addItem(device)
        
        # 修改：设置默认设备为cuda:0
        if 'cuda:0' in available_devices:
            self.device_combo.setCurrentText('cuda:0')
        elif available_devices:
            self.device_combo.setCurrentText(available_devices[0])
        
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)
        
        # 添加设备信息标签
        self.device_info_label = QLabel(f"Detected: {len(available_devices)} device(s)")
        self.device_info_label.setFont(QFont("Arial", 8))
        self.device_info_label.setStyleSheet("color: #666;")
        device_layout.addWidget(self.device_info_label)
        
        left_layout.addLayout(device_layout)
        
        # 左侧底部按钮区域 - 重新设计布局
        left_bottom_widget = QWidget()
        left_bottom_layout = QVBoxLayout()
        left_bottom_layout.setSpacing(10)
        
        # 修改：参数操作按钮在上方
        # 参数操作按钮 - 三个按钮在一行
        param_buttons_layout = QHBoxLayout()
        param_buttons_layout.setSpacing(8)
        
        self.load_defaults_btn = QPushButton("Default Parameters")
        self.load_defaults_btn.setFont(QFont("Arial", 9))
        self.load_defaults_btn.setMinimumHeight(35)
        self.load_defaults_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.load_defaults_btn.clicked.connect(self.load_default_parameters)
        
        self.load_previous_btn = QPushButton("Load Parameters")
        self.load_previous_btn.setFont(QFont("Arial", 9))
        self.load_previous_btn.setMinimumHeight(35)
        self.load_previous_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.load_previous_btn.clicked.connect(self.load_previous_parameters)
        
        self.export_params_btn = QPushButton("Export Parameters")
        self.export_params_btn.setFont(QFont("Arial", 9))
        self.export_params_btn.setMinimumHeight(35)
        self.export_params_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.export_params_btn.clicked.connect(self.export_current_parameters)
        
        param_buttons_layout.addWidget(self.load_defaults_btn)
        param_buttons_layout.addWidget(self.load_previous_btn)
        param_buttons_layout.addWidget(self.export_params_btn)
        
        left_bottom_layout.addLayout(param_buttons_layout)
        
        # 运行重建按钮 - 在参数按钮下方
        self.run_btn = QPushButton("Run Reconstruction")
        self.run_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.run_btn.setMinimumHeight(50)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;  /* 绿色 */
                color: white;
                padding: 12px;
                border-radius: 6px;
                border: 2px solid #388E3C;
            }
            QPushButton:hover {
                background-color: #45a049;
                border-color: #2E7D32;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                border-color: #999999;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
                padding-top: 13px;
                padding-bottom: 11px;
            }
        """)
        self.run_btn.clicked.connect(self.run_reconstruction)
        left_bottom_layout.addWidget(self.run_btn)

        # 停止重建按钮
        self.stop_recon_btn = QPushButton("Stop Reconstruction")
        self.stop_recon_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.stop_recon_btn.setMinimumHeight(50)
        self.stop_recon_btn.setEnabled(False)  # 初始不可用
        self.stop_recon_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;  /* 红色 */
                color: white;
                padding: 12px;
                border-radius: 6px;
                border: 2px solid #d32f2f;
            }
            QPushButton:hover {
                background-color: #d32f2f;
                border-color: #b71c1c;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                border-color: #999999;
            }
            QPushButton:pressed {
                background-color: #c62828;
                padding-top: 13px;
                padding-bottom: 11px;
            }
        """)
        self.stop_recon_btn.clicked.connect(self.stop_reconstruction)
        left_bottom_layout.addWidget(self.stop_recon_btn)
        
        left_bottom_widget.setLayout(left_bottom_layout)
        left_layout.addWidget(left_bottom_widget)
        
        # 进度条和状态标签
        self.progress_bar = QProgressBar()
        # 初始设置为空灰色
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFont(QFont("Arial", 9))
        # 修改：改进的进度条样式
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 3px;
                text-align: center;
                background-color: #f0f0f0;
                min-height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;  /* 绿色 */
                border-radius: 2px;
                width: 1px;  /* 确保平滑填充 */
            }
        """)
        left_layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("Ready !")
        self.status_label.setFont(QFont("Arial", 9))
        self.status_label.setFixedHeight(20)  # 固定高度
        left_layout.addWidget(self.status_label)
        
        left_panel.setLayout(left_layout)
        main_layout.addWidget(left_panel)
        
        # 右侧结果显示面板
        right_panel = QWidget()
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(10)  # 增加间距
        
        # 标题
        results_title = QLabel("Reconstruction Results")
        results_title.setFont(QFont("Arial", 14, QFont.Bold))
        results_title.setStyleSheet("padding: 10px;")
        results_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(results_title)
        
        # 图像显示区域 - 2x2网格布局，使用FixedSizeContainer来包装
        image_grid_widget = QWidget()
        image_grid_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_grid = QGridLayout(image_grid_widget)
        image_grid.setSpacing(10)
        
        # 创建四个图像显示容器
        self.bf_display = ImageDisplayContainer("BF-SIM Reconstruction")
        self.bf_hessian_display = ImageDisplayContainer("BF+Hessian-SIM Reconstruction")
        self.bf_hessian_tdv_display = ImageDisplayContainer("BF+Hessian+TDV-SIM Reconstruction")
        self.hybrid_display = ImageDisplayContainer("Hybrid-SIM Reconstruction")
        
        # 添加到网格
        image_grid.addWidget(self.bf_display, 0, 0, Qt.AlignCenter)
        image_grid.addWidget(self.bf_hessian_display, 0, 1, Qt.AlignCenter)
        image_grid.addWidget(self.bf_hessian_tdv_display, 1, 0, Qt.AlignCenter)
        image_grid.addWidget(self.hybrid_display, 1, 1, Qt.AlignCenter)
        
        # 设置网格对齐方式
        image_grid.setAlignment(Qt.AlignCenter)
        
        # 创建一个滚动区域来包含图像网格
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(image_grid_widget)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        right_layout.addWidget(scroll_area, 1)  # 添加拉伸因子
        
        # 存储所有图像显示控件，用于联动
        self.image_displays = [
            self.bf_display.image_widget,
            self.bf_hessian_display.image_widget,
            self.bf_hessian_tdv_display.image_widget,
            self.hybrid_display.image_widget
        ]
        
        # 存储所有图像显示容器，用于禁用交互
        self.image_containers = [
            self.bf_display,
            self.bf_hessian_display,
            self.bf_hessian_tdv_display,
            self.hybrid_display
        ]
        
        # 连接缩放和平移信号以实现联动
        for display in self.image_displays:
            display.zoom_changed.connect(self.sync_zoom)
            display.pan_changed.connect(self.sync_pan)
        
        # 图像操作按钮 - 修改：按照新结构调整布局
        image_buttons_layout = QHBoxLayout()
        
        # 显示区间标签
        range_label = QLabel("Display Range:")
        range_label.setFont(QFont("Arial", 9))
        image_buttons_layout.addWidget(range_label)
        
        # 最小值滑块 - 缩短长度
        self.min_slider = QSlider(Qt.Horizontal)
        self.min_slider.setRange(0, 1000)
        self.min_slider.setValue(0)
        self.min_slider.setEnabled(False)  # 初始不可用
        self.min_slider.setMinimumWidth(140)  # 缩短滑块长度
        self.min_slider.setMaximumWidth(140)
        self.min_slider.valueChanged.connect(self.display_range_changed)
        image_buttons_layout.addWidget(self.min_slider)
        
        # 最大值滑块 - 缩短长度
        self.max_slider = QSlider(Qt.Horizontal)
        self.max_slider.setRange(0, 1000)
        self.max_slider.setValue(1000)
        self.max_slider.setEnabled(False)  # 初始不可用
        self.max_slider.setMinimumWidth(140)  # 缩短滑块长度
        self.max_slider.setMaximumWidth(140)
        self.max_slider.valueChanged.connect(self.display_range_changed)
        image_buttons_layout.addWidget(self.max_slider)
        
        # 重置显示区间按钮 - 设置与Reset View相同的高度
        self.reset_range_btn = QPushButton("Reset Range")
        self.reset_range_btn.setFont(QFont("Arial", 9))
        self.reset_range_btn.setMinimumHeight(35)  # 与Reset View高度一致
        self.reset_range_btn.setEnabled(False)  # 初始不可用
        self.reset_range_btn.clicked.connect(self.reset_display_range)
        image_buttons_layout.addWidget(self.reset_range_btn)
        
        # Reset View按钮
        self.reset_zoom_btn = QPushButton("Reset View")
        self.reset_zoom_btn.setFont(QFont("Arial", 9))
        self.reset_zoom_btn.setMinimumHeight(35)
        self.reset_zoom_btn.setEnabled(False)  # 初始不可用，因为没有图像数据
        self.reset_zoom_btn.clicked.connect(self.reset_view_all)
        image_buttons_layout.addWidget(self.reset_zoom_btn)
        
        # 添加Export Images按钮
        self.export_images_btn = QPushButton("Export Results")
        self.export_images_btn.setFont(QFont("Arial", 9))
        self.export_images_btn.setMinimumHeight(35)
        self.export_images_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;  /* 蓝色 */
                color: white;
                border-radius: 4px;
                padding: 6px;
                border: 1px solid #1976D2;
            }
            QPushButton:hover {
                background-color: #1976D2;
                border-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                border-color: #999999;
            }
        """)
        self.export_images_btn.clicked.connect(self.export_images)
        self.export_images_btn.setEnabled(False)  # 初始不可用，重建完成后才可用
        image_buttons_layout.addWidget(self.export_images_btn)
        
        image_buttons_layout.addStretch()
        
        right_layout.addLayout(image_buttons_layout)
        
        # 全局帧控制组件（右下角） - 添加视频播放/暂停按钮
        frame_control_widget = QWidget()
        frame_control_layout = QHBoxLayout()
        
        # 视频播放/暂停按钮
        self.play_pause_btn = QPushButton()
        self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_pause_btn.setToolTip("Play/Stop Video")
        self.play_pause_btn.setFixedSize(35, 35)
        self.play_pause_btn.setEnabled(False)  # 初始不可用
        self.play_pause_btn.clicked.connect(self.toggle_video_playback)
        self.play_pause_btn.setStyleSheet("""
            QPushButton {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f0f0f0;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        frame_control_layout.addWidget(self.play_pause_btn)
        
        # 全局帧控制标签
        global_frame_label = QLabel("Frame:")
        global_frame_label.setFont(QFont("Arial", 9))
        frame_control_layout.addWidget(global_frame_label)
        
        # 全局帧滑块
        self.global_frame_slider = QSlider(Qt.Horizontal)
        self.global_frame_slider.setRange(0, 0)
        self.global_frame_slider.setValue(0)
        self.global_frame_slider.setEnabled(False)
        self.global_frame_slider.valueChanged.connect(self.global_frame_changed)
        self.global_frame_slider.setMinimumHeight(20)
        frame_control_layout.addWidget(self.global_frame_slider)
        
        # 全局帧标签
        self.global_frame_label = QLabel("0/0")
        self.global_frame_label.setFont(QFont("Arial", 9))
        self.global_frame_label.setMinimumWidth(40)
        frame_control_layout.addWidget(self.global_frame_label)
        
        frame_control_widget.setLayout(frame_control_layout)
        right_layout.addWidget(frame_control_widget)
        
        right_panel.setLayout(right_layout)
        main_layout.addWidget(right_panel, 1)
        
        central_widget.setLayout(main_layout)
        
        # 连接进度更新信号
        self.progress_update_signal.connect(self.update_progress)
        
        # 加载默认参数
        self.load_default_parameters()
    
    def stop_reconstruction(self):
        """停止重建过程"""
        if self.reconstruction_thread and self.reconstruction_thread.isRunning():
            reply = QMessageBox.question(self, 'Stop Reconstruction', 
                                        'Are you sure you want to stop the reconstruction process?',
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.stop_recon_btn.setEnabled(False)
                self.stop_recon_btn.setText("Stopping ...")
                
                # 停止重建线程
                self.reconstruction_thread.stop()
                
                # 立即更新UI状态
                self.status_label.setText("Stopping reconstruction ...")
                QApplication.processEvents()
                
                # 等待一小段时间确保线程停止
                QTimer.singleShot(1000, self.handle_stop_completed)
    
    def handle_stop_completed(self):
        """处理停止完成"""
        # 恢复按钮状态
        self.stop_recon_btn.setEnabled(False)
        self.stop_recon_btn.setText("Stop Reconstruction")
        
        # 更新状态
        self.status_label.setText("Reconstruction stopped !")
        
        # 启用控件
        self.enable_controls()
        self.run_btn.setEnabled(True)
        
        # 清理内存
        self.force_cleanup()

    def update_progress(self, value):
        """更新进度条"""
        # 始终使用确定模式
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(value)
        
        # 更新状态标签
        if value == 0:
            self.status_label.setText("Ready !")
        elif value < 25:
            self.status_label.setText(f"Executing BF-SIM reconstruction ...")
        elif value < 50:
            self.status_label.setText(f"Executing Hessian Denoise ...")
        elif value < 75:
            self.status_label.setText(f"Executing TDV Denoise ...")
        elif value < 100:
            self.status_label.setText(f"Executing Sparse Deconvolution ...")
        else:
            self.status_label.setText("Reconstruction completed !")
    
    def sync_zoom(self, zoom_factor, pan_offset):
        """同步所有图像的缩放和平移"""
        # 停止视频播放
        self.stop_video_playback()
        sender = self.sender()
        for display in self.image_displays:
            if display != sender and display.original_qimage is not None:
                display.set_zoom_and_pan(zoom_factor, pan_offset)
    
    def sync_pan(self, pan_offset):
        """同步所有图像的平移"""
        # 停止视频播放
        self.stop_video_playback()
        sender = self.sender()
        for display in self.image_displays:
            if display != sender and display.original_qimage is not None:
                display.set_zoom_and_pan(display.zoom_factor, pan_offset)
    
    def detect_available_devices(self):
        """检测可用的计算设备"""
        available_devices = ['cpu']  # CPU总是可用的
        
        # 尝试检测CUDA设备
        try:
            import torch
            if torch.cuda.is_available():
                cuda_count = torch.cuda.device_count()
                for i in range(cuda_count):
                    device_name = f"cuda:{i}"
                    # 获取设备名称
                    try:
                        device_props = torch.cuda.get_device_properties(i)
                        device_name_display = f"cuda:{i} ({device_props.name}, {device_props.total_memory/1024**3:.1f}GB)"
                    except:
                        device_name_display = f"cuda:{i}"
                    available_devices.append(device_name_display)
            else:
                print("CUDA is not available")
        except ImportError:
            print("PyTorch not installed, cannot detect CUDA devices")
        except Exception as e:
            print(f"Error detecting CUDA devices: {e}")
        
        return available_devices
    
    def reset_view_all(self):
        """重置所有图像的视图"""
        # 停止视频播放
        self.stop_video_playback()
        for display in self.image_displays:
            if display.original_qimage is not None:
                display.reset_view()
    
    def global_frame_changed(self, value):
        """全局帧变化 - 更新所有图像并保持缩放、平移和显示区间状态"""
        # 如果不是由视频播放触发的帧变化，停止视频播放
        if not self._is_video_changing_frame and self.is_video_playing:
            # 如果用户手动拖动滑块而视频正在播放，停止播放
            self.stop_video_playback()
        
        self.current_frame = value
        
        # 更新所有图像显示，但保持当前的缩放、平移和显示区间状态
        for display in self.image_displays:
            if display.original_image is not None:
                # 保存当前的缩放和平移状态
                current_zoom = display.zoom_factor
                current_pan = QPoint(display.pan_offset)
                current_display_min = display.display_min
                current_display_max = display.display_max
                
                # 更新当前帧
                display.current_frame = value
                
                # 如果设置了显示区间，应用显示区间到当前帧
                if current_display_min is not None and current_display_max is not None:
                    display.apply_display_range_to_current_frame()
                else:
                    # 否则使用全局范围显示
                    if len(display.original_image.shape) == 3:
                        frame_data = display.original_image[value]
                    else:
                        frame_data = display.original_image
                    
                    # 使用全局范围进行归一化
                    if display.global_max - display.global_min > 1e-10:
                        img_normalized = (frame_data - display.global_min) / (display.global_max - display.global_min)
                    else:
                        img_normalized = np.zeros_like(frame_data)
                    
                    # 确保值在[0, 1]范围内
                    img_normalized = np.clip(img_normalized, 0, 1)
                    img_8bit = (img_normalized * 255).astype(np.uint8)
                    
                    # 转换为QImage
                    height, width = img_8bit.shape
                    if len(img_8bit.shape) == 2:  # 灰度图
                        bytes_per_line = width
                        qimage = QImage(img_8bit.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                        # 复制QImage数据以防止原始数据被释放
                        display.original_qimage = qimage.copy()
                    else:  # RGB图
                        bytes_per_line = 3 * width
                        qimage = QImage(img_8bit.data, width, height, bytes_per_line, QImage.Format_RGB888)
                        # 复制QImage数据以防止原始数据被释放
                        display.original_qimage = qimage.copy()
                
                # 恢复缩放和平移状态
                display.set_zoom_and_pan(current_zoom, current_pan)
        
        # 更新全局帧标签
        max_frames = self.global_frame_slider.maximum() + 1
        self.global_frame_label.setText(f"{value+1}/{max_frames}")
        
        # 重置视频触发标志
        self._is_video_changing_frame = False
    
    def display_range_changed(self):
        """显示区间变化"""
        # 停止视频播放
        self.stop_video_playback()
        
        if not self.results:
            return
        
        # 获取滑块值并转换为实际值
        min_val = self.min_slider.value() / 1000.0
        max_val = self.max_slider.value() / 1000.0
        
        # 确保最小值小于最大值
        if min_val >= max_val:
            # 调整最小值
            min_val = max_val - 0.001
            if min_val < 0:
                min_val = 0
            self.min_slider.setValue(int(min_val * 1000))
        
        # 计算实际显示值
        self.display_min_val = self.global_min_val + (self.global_max_val - self.global_min_val) * min_val
        self.display_max_val = self.global_min_val + (self.global_max_val - self.global_min_val) * max_val
        
        # 更新所有图像的显示区间
        for container in self.image_containers:
            container.set_display_range(self.display_min_val, self.display_max_val)
    
    def reset_display_range(self):
        """重置显示区间"""
        # 停止视频播放
        self.stop_video_playback()
        
        # 重置滑块到全局范围
        self.min_slider.setValue(0)
        self.max_slider.setValue(1000)
        
        # 重置显示值
        self.display_min_val = self.global_min_val
        self.display_max_val = self.global_max_val
        
        # 重置所有图像的显示区间
        for container in self.image_containers:
            container.set_display_range(self.display_min_val, self.display_max_val)
    
    def update_global_range(self):
        """更新全局范围"""
        if not self.results:
            return
        
        # 获取所有图像的最小值和最大值
        all_mins = []
        all_maxs = []
        
        for container in self.image_containers:
            g_min, g_max = container.get_global_range()
            if g_min is not None and g_max is not None:
                all_mins.append(g_min)
                all_maxs.append(g_max)
        
        if all_mins and all_maxs:
            self.global_min_val = min(all_mins)
            self.global_max_val = max(all_maxs)
            
            # 重置显示区间到全局范围
            self.display_min_val = self.global_min_val
            self.display_max_val = self.global_max_val
            
            # 启用显示区间控制
            self.min_slider.setEnabled(True)
            self.max_slider.setEnabled(True)
            self.reset_range_btn.setEnabled(True)
    
    def update_global_frame_control(self, max_frames):
        """更新全局帧控制"""
        if max_frames > 0:
            self.global_frame_slider.setRange(0, max_frames - 1)
            self.global_frame_slider.setEnabled(True)
            self.global_frame_label.setText(f"1/{max_frames}")
            
            # 启用视频播放按钮
            self.play_pause_btn.setEnabled(True)
        else:
            self.global_frame_slider.setRange(0, 0)
            self.global_frame_slider.setEnabled(False)
            self.global_frame_label.setText("0/0")
            
            # 禁用视频播放按钮
            self.play_pause_btn.setEnabled(False)
    
    def disable_controls_(self):
        """禁用所有控件"""
        # 禁用左侧面板控件
        self.device_combo.setEnabled(False)
        self.load_defaults_btn.setEnabled(False)
        self.load_previous_btn.setEnabled(False)
        self.export_params_btn.setEnabled(False)
        
        # 禁用参数标签页中的所有控件
        self.bf_sim_widget.set_enabled(False)
        self.hessian_widget.set_enabled(False)
        self.tdv_widget.set_enabled(False)
        self.sparse_widget.set_enabled(False)
        
        # 禁用右侧面板控件
        self.reset_zoom_btn.setEnabled(False)
        self.export_images_btn.setEnabled(False)
        self.global_frame_slider.setEnabled(False)
        self.play_pause_btn.setEnabled(False)  # 禁用视频播放按钮
        
        # 禁用显示区间控制
        self.min_slider.setEnabled(False)
        self.max_slider.setEnabled(False)
        self.reset_range_btn.setEnabled(False)
        
        # 禁用图像交互
        for container in self.image_containers:
            container.set_interaction_enabled(False)
    
    def disable_controls(self):
        """禁用所有控件"""
        # 禁用左侧面板控件
        self.device_combo.setEnabled(False)
        self.load_defaults_btn.setEnabled(False)
        self.load_previous_btn.setEnabled(False)
        self.export_params_btn.setEnabled(False)
        
        # 禁用参数标签页中的所有控件
        self.bf_sim_widget.set_enabled(False)
        self.hessian_widget.set_enabled(False)
        self.tdv_widget.set_enabled(False)
        self.sparse_widget.set_enabled(False)
        
        # 禁用运行按钮，启用停止按钮
        self.run_btn.setEnabled(False)
        self.stop_recon_btn.setEnabled(True)

        # 禁用右侧面板控件
        self.reset_zoom_btn.setEnabled(False)
        self.export_images_btn.setEnabled(False)
        self.global_frame_slider.setEnabled(False)
        self.play_pause_btn.setEnabled(False)  # 禁用视频播放按钮
        
        # 禁用显示区间控制
        self.min_slider.setEnabled(False)
        self.max_slider.setEnabled(False)
        self.reset_range_btn.setEnabled(False)
        
        # 禁用图像交互
        for container in self.image_containers:
            container.set_interaction_enabled(False)

    def enable_controls_(self):
        """启用所有控件"""
        # 启用左侧面板控件
        self.device_combo.setEnabled(True)
        self.load_defaults_btn.setEnabled(True)
        self.load_previous_btn.setEnabled(True)
        self.export_params_btn.setEnabled(True)
        
        # 启用参数标签页中的所有控件
        self.bf_sim_widget.set_enabled(True)
        self.hessian_widget.set_enabled(True)
        self.tdv_widget.set_enabled(True)
        self.sparse_widget.set_enabled(True)
        
        # 启用右侧面板控件（根据条件）
        self.reset_zoom_btn.setEnabled(True)
        self.export_images_btn.setEnabled(bool(self.results))  # 只有有结果时才启用
        self.global_frame_slider.setEnabled(self.global_frame_slider.maximum() > 0)  # 只有有帧时才启用
        self.play_pause_btn.setEnabled(bool(self.results) and self.global_frame_slider.maximum() > 0)  # 与Export results相同逻辑
        
        # 启用显示区间控制（只有有结果时才启用）
        has_results = bool(self.results)
        self.min_slider.setEnabled(has_results)
        self.max_slider.setEnabled(has_results)
        self.reset_range_btn.setEnabled(has_results)
        
        # 启用图像交互
        for container in self.image_containers:
            container.set_interaction_enabled(True)
    
    def enable_controls(self):
        """启用所有控件"""
        # 启用左侧面板控件
        self.device_combo.setEnabled(True)
        self.load_defaults_btn.setEnabled(True)
        self.load_previous_btn.setEnabled(True)
        self.export_params_btn.setEnabled(True)
        
        # 启用参数标签页中的所有控件
        self.bf_sim_widget.set_enabled(True)
        self.hessian_widget.set_enabled(True)
        self.tdv_widget.set_enabled(True)
        self.sparse_widget.set_enabled(True)
        
        # 启用运行按钮，禁用停止按钮
        self.run_btn.setEnabled(True)
        self.stop_recon_btn.setEnabled(False)

        # 启用右侧面板控件（根据条件）
        self.reset_zoom_btn.setEnabled(True)
        self.export_images_btn.setEnabled(bool(self.results))  # 只有有结果时才启用
        self.global_frame_slider.setEnabled(self.global_frame_slider.maximum() > 0)  # 只有有帧时才启用
        self.play_pause_btn.setEnabled(bool(self.results) and self.global_frame_slider.maximum() > 0)  # 与Export results相同逻辑
        
        # 启用显示区间控制（只有有结果时才启用）
        has_results = bool(self.results)
        self.min_slider.setEnabled(has_results)
        self.max_slider.setEnabled(has_results)
        self.reset_range_btn.setEnabled(has_results)
        
        # 启用图像交互
        for container in self.image_containers:
            container.set_interaction_enabled(True)

    def force_cleanup(self):
        """强制清理内存"""
        try:
            # 清理Python内存
            import gc
            gc.collect()
            
            # 清理PyTorch GPU缓存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except:
                pass
            
            # 清理线程引用
            self.reconstruction_thread = None
            
            # 清理结果数据
            self.results = {}
            
            # 重置进度条
            self.progress_bar.setValue(0)
            
        except Exception as e:
            print(f"Force cleanup error: {e}")
    
    def reset_right_panel(self):
        """重置右侧面板（视野和帧数）"""
        # 清空所有图像显示
        for container in self.image_containers:
            container.clear_display()
        
        # 重置所有图像的视野
        self.reset_view_all()
        
        # 停止视频播放
        self.stop_video_playback()
        
        # 重置帧数为第一帧
        self.current_frame = 0
        self.global_frame_slider.setRange(0, 0)
        self.global_frame_slider.setValue(0)
        self.global_frame_label.setText("0/0")
        
        # 重置显示区间滑块
        self.min_slider.setValue(0)
        self.max_slider.setValue(1000)
        
        # 重置全局范围变量
        self.global_min_val = 0
        self.global_max_val = 1
        self.display_min_val = 0
        self.display_max_val = 1
        
        # 禁用右侧相关控件
        self.reset_zoom_btn.setEnabled(False)
        self.export_images_btn.setEnabled(False)
        self.global_frame_slider.setEnabled(False)
        self.play_pause_btn.setEnabled(False)  # 禁用视频播放按钮
        self.min_slider.setEnabled(False)
        self.max_slider.setEnabled(False)
        self.reset_range_btn.setEnabled(False)
    
    def load_default_parameters(self):
        """加载默认参数"""
        # 停止视频播放
        self.stop_video_playback()
        
        self.status_label.setText("Loading default parameters ...")
        QApplication.processEvents()  # 立即更新UI
        
        # 更新所有参数控件的值为默认值
        for widget in [self.bf_sim_widget, self.hessian_widget, self.tdv_widget, self.sparse_widget]:
            for param_widget in widget.parameters.values():
                param_widget.reset_value()
        
        # 重新检测设备并更新下拉框
        available_devices = self.detect_available_devices()
        self.device_combo.clear()
        for device in available_devices:
            self.device_combo.addItem(device)
        
        # 修改：设置默认设备为cuda:0
        # 查找包含"cuda:0"的设备
        for i in range(self.device_combo.count()):
            if 'cuda:0' in self.device_combo.itemText(i).split('(')[0].strip():
                self.device_combo.setCurrentIndex(i)
                break
            elif available_devices:
                self.device_combo.setCurrentText(available_devices[0])
        
        self.device_info_label.setText(f"Detected: {len(available_devices)} device(s)")
        self.status_label.setText("Default parameters loaded")
        # 重置进度条为空的灰色
        self.progress_update_signal.emit(0)
    
    def load_previous_parameters(self):
        """加载之前保存的参数"""
        # 停止视频播放
        self.stop_video_playback()
        
        self.collect_parameters()
        try:
            # 构建默认文件名
            try:
                filedir, filename = os.path.split(self.args.SIM_raw_data_path)
                base_name = filedir +'/'
            except AttributeError:
                # 如果 fname 不存在，使用空字符串作为基础名称
                base_name = ""

            filename, _ = QFileDialog.getOpenFileName(
                self, "Load Parameters", base_name, "JSON Files (*.json);;Text Files (*.txt);;All Files (*)"
            )
            if filename:
                self.status_label.setText(f"Loading parameters from {Path(filename).name} ...")
                QApplication.processEvents()  # 立即更新UI
                
                with open(filename, 'r') as f:
                    params = json.load(f)
                
                # 设置参数值
                self.set_parameters_from_dict(params)
                
                self.status_label.setText(f"Parameters loaded from {Path(filename).name}")
                
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Failed to load parameters: {str(e)}")
            self.status_label.setText(f"Error loading parameters: {str(e)}")
    
    def export_current_parameters(self):
        """导出当前参数"""
        # 停止视频播放
        self.stop_video_playback()
        
        self.collect_parameters()
        try:
            # 收集当前参数
            params = self.collect_all_parameters()            

            # 构建默认文件名
            try:
                filedir, filename = os.path.split(self.args.SIM_raw_data_path)
                fname, ext = os.path.splitext(filename)
                base_name = filedir +'/'+fname + "_Recon_Parameters.json"
            except AttributeError:
                # 如果 fname 不存在，使用空字符串作为基础名称
                base_name = "Demo_Recon_SIM_Recon_Parameters.json"

            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Parameters", base_name, 
                "JSON Files (*.json);;Text Files (*.txt);;All Files (*)"
            )
            if filename:
                self.status_label.setText(f"Exporting parameters to {Path(filename).name} ...")
                QApplication.processEvents()  # 立即更新UI
                
                with open(filename, 'w') as f:
                    json.dump(params, f, indent=4)
                
                self.status_label.setText(f"Parameters exported to {Path(filename).name}")
                
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export parameters: {str(e)}")
            self.status_label.setText(f"Error exporting parameters: {str(e)}")
    
    def save_image_data(self, data, name, save_dir):
        """保存图像数据到文件"""
        try:
            # 确保数据是numpy数组
            data = np.array(data)
            
            # 确保数据类型是float32
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            
            # 保存为TIFF格式
            filename = f"{name}.tif"
            filepath = os.path.join(save_dir, filename)
            tifffile.imwrite(filepath, data, photometric='minisblack')
            
            return filename
        except Exception as e:
            print(f"Error saving {name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def export_images(self):
        """导出图像数据"""
        # 停止视频播放
        self.stop_video_playback()
        
        if not self.results:
            QMessageBox.warning(self, "No Data", "Please run reconstruction first to generate data.")
            return
        
        # 确定最大帧数
        max_frames = 0
        for key, result in self.results.items():
            if len(result.shape) == 3:
                max_frames = max(max_frames, result.shape[0])
        
        # 创建对话框让用户选择要导出的数据
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Results")
        dialog.setMinimumWidth(400)  # 增加宽度以容纳更多控件
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        layout = QVBoxLayout()
        
        # 第一部分：导出完整数据
        full_data_group = QGroupBox("Export Full Reconstruction")
        full_data_layout = QVBoxLayout()

        

        # 创建复选框
        self.export_bf_checkbox = QCheckBox("BF-SIM Reconstruction")
        self.export_bf_hessian_checkbox = QCheckBox("BF+Hessian-SIM Reconstruction")
        self.export_bf_hessian_tdv_checkbox = QCheckBox("BF+Hessian+TDV-SIM Reconstruction")
        self.export_hybrid_checkbox = QCheckBox("Hybrid-SIM Reconstruction")
        
        # 默认只选择Hybrid-SIM
        self.export_hybrid_checkbox.setChecked(True)
        
        full_data_layout.addWidget(self.export_bf_checkbox)
        full_data_layout.addWidget(self.export_bf_hessian_checkbox)
        full_data_layout.addWidget(self.export_bf_hessian_tdv_checkbox)
        full_data_layout.addWidget(self.export_hybrid_checkbox)
        full_data_group.setLayout(full_data_layout)
        layout.addWidget(full_data_group)
        
        # 第二部分：导出特定帧堆栈
        frame_stack_group = QGroupBox("Export Single-Frame Comparison")
        frame_stack_layout = QVBoxLayout()
        
        # 复选框：是否导出帧堆栈
        self.export_frame_stack_checkbox = QCheckBox("Export Comparison")
        self.export_frame_stack_checkbox.setChecked(False)
        
        # 帧选择控件
        frame_selector_widget = QWidget()
        frame_selector_layout = QHBoxLayout()
        frame_selector_layout.setContentsMargins(20, 5, 5, 5)  # 缩进
        
        frame_label = QLabel("Frame index:")
        self.frame_spinbox = QSpinBox()
        
        # 修改：帧编号从1开始显示给用户
        if max_frames > 0:
            # 显示给用户的帧编号从1开始，内部处理时减1
            self.frame_spinbox.setRange(1, max_frames)
            self.frame_spinbox.setValue(1)  # 默认显示第1帧
            frame_range_label = QLabel(f"Range: 1 - {max_frames}")
        else:
            self.frame_spinbox.setRange(1, 1)
            self.frame_spinbox.setValue(1)
            frame_range_label = QLabel("Range: 1 - 1 (2D data only)")
        
        self.frame_spinbox.setEnabled(False)  # 初始不可用
        
        frame_range_label.setStyleSheet("color: #666; font-size: 9pt;")
        
        frame_selector_layout.addWidget(frame_label)
        frame_selector_layout.addWidget(self.frame_spinbox)
        frame_selector_layout.addWidget(frame_range_label)
        frame_selector_layout.addStretch()
        
        frame_selector_widget.setLayout(frame_selector_layout)
        
        # 连接复选框状态改变信号
        self.export_frame_stack_checkbox.toggled.connect(self.frame_spinbox.setEnabled)
        
        frame_stack_layout.addWidget(self.export_frame_stack_checkbox)
        frame_stack_layout.addWidget(frame_selector_widget)
        frame_stack_group.setLayout(frame_stack_layout)
        layout.addWidget(frame_stack_group)
        
        # 添加一些说明
        note_label = QLabel("Note: The comparison will sequentially contain BF, BF+Hessian, BF+Hessian+TDV and Hybrid reconstructions.")
        note_label.setStyleSheet("color: #666; font-size: 9pt; padding: 5px;")
        layout.addWidget(note_label)
        
        # 导出按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() != QDialog.Accepted:
            return
        
        # 获取导出选项
        export_options = {
            'bf': self.export_bf_checkbox.isChecked(),
            'bf_hessian': self.export_bf_hessian_checkbox.isChecked(),
            'bf_hessian_tdv': self.export_bf_hessian_tdv_checkbox.isChecked(),
            'hybrid': self.export_hybrid_checkbox.isChecked(),
            'frame_stack': self.export_frame_stack_checkbox.isChecked(),
            'frame_index': self.frame_spinbox.value() if self.export_frame_stack_checkbox.isChecked() else None
        }
        
        # 检查是否有选中的数据
        has_full_data = any([export_options['bf'], export_options['bf_hessian'], 
                            export_options['bf_hessian_tdv'], export_options['hybrid']])
        
        if not has_full_data and not export_options['frame_stack']:
            QMessageBox.warning(self, "No Selection", "Please select at least one data type to export.")
            return
        
        # 构建默认文件名
        try:
            filedir, filename = os.path.split(self.args.SIM_raw_data_path)
            base_name = filedir +'/'
        except AttributeError:
            base_name = ""
        
        # 选择保存路径
        save_dir = QFileDialog.getExistingDirectory(
            self, "Select Save Directory", base_name, QFileDialog.ShowDirsOnly
        )
        if not save_dir:
            return
        
        # 导出数据
        try:
            self.status_label.setText("Exporting images ...")
            QApplication.processEvents()  # 立即更新UI
            
            success_count = 0
            
            # 计算总文件数（包括可能的堆栈文件）
            total_files = sum([1 for v in [export_options['bf'], export_options['bf_hessian'], 
                                        export_options['bf_hessian_tdv'], export_options['hybrid']] if v])
            if export_options['frame_stack']:
                total_files += 1  # 堆栈文件算作一个文件
            
            # 导出完整3D数据
            if has_full_data:
                # 导出BF-SIM数据
                if export_options['bf'] and 'bf_result' in self.results:
                    data = self.results['bf_result']
                    filename = self.save_image_data(data, self.args.fname+'_BF', save_dir)
                    if filename:
                        success_count += 1
                    self.progress_update_signal.emit((success_count * 100) // total_files)
                    QApplication.processEvents()
                
                # 导出BF+Hessian-SIM数据
                if export_options['bf_hessian'] and 'bf_hessian_result' in self.results:
                    data = self.results['bf_hessian_result']
                    filename = self.save_image_data(data, self.args.fname+'_BF+Hessian', save_dir)
                    if filename:
                        success_count += 1
                    self.progress_update_signal.emit((success_count * 100) // total_files)
                    QApplication.processEvents()
                
                # 导出BF+Hessian+TDV-SIM数据
                if export_options['bf_hessian_tdv'] and 'bf_hessian_tdv_result' in self.results:
                    data = self.results['bf_hessian_tdv_result']
                    filename = self.save_image_data(data, self.args.fname+'_BF+Hessian+TDV', save_dir)
                    if filename:
                        success_count += 1
                    self.progress_update_signal.emit((success_count * 100) // total_files)
                    QApplication.processEvents()
                
                # 导出Hybrid-SIM数据
                if export_options['hybrid'] and 'hybrid_result' in self.results:
                    data = self.results['hybrid_result']
                    filename = self.save_image_data(data, self.args.fname+'_Hybrid', save_dir)
                    if filename:
                        success_count += 1
                    self.progress_update_signal.emit((success_count * 100) // total_files)
                    QApplication.processEvents()
            
            # 导出帧堆栈
            if export_options['frame_stack']:
                try:
                    # 修改：用户看到的帧编号从1开始，内部处理时减1
                    user_frame_idx = export_options['frame_index']  # 用户选择的帧编号（1-based）
                    internal_frame_idx = user_frame_idx - 1  # 转换为0-based索引
                    
                    stack_frames = []
                    method_names = []  # 记录哪些方法被包含
                    
                    # 按照固定顺序提取帧：BF, BF+Hessian, BF+Hessian+TDV, Hybrid
                    if 'bf_result' in self.results:
                        method_names.append("BF")
                        if len(self.results['bf_result'].shape) == 3:
                            if internal_frame_idx < self.results['bf_result'].shape[0]:
                                frame_data = self.results['bf_result'][internal_frame_idx]
                            else:
                                frame_data = self.results['bf_result'][-1]
                        else:
                            frame_data = self.results['bf_result']
                        
                        # 确保数据是2D的
                        if len(frame_data.shape) == 2:
                            stack_frames.append(frame_data)
                        elif len(frame_data.shape) == 3 and frame_data.shape[2] <= 3:
                            # 如果是彩色图像，转换为灰度
                            if frame_data.shape[2] == 3:
                                # RGB转灰度公式: Y = 0.299R + 0.587G + 0.114B
                                gray_frame = np.dot(frame_data[..., :3], [0.299, 0.587, 0.114])
                                stack_frames.append(gray_frame)
                            else:
                                stack_frames.append(frame_data[:, :, 0])
                    
                    if 'bf_hessian_result' in self.results:
                        method_names.append("BF+Hessian")
                        if len(self.results['bf_hessian_result'].shape) == 3:
                            if internal_frame_idx < self.results['bf_hessian_result'].shape[0]:
                                frame_data = self.results['bf_hessian_result'][internal_frame_idx]
                            else:
                                frame_data = self.results['bf_hessian_result'][-1]
                        else:
                            frame_data = self.results['bf_hessian_result']
                        
                        # 确保数据是2D的
                        if len(frame_data.shape) == 2:
                            stack_frames.append(frame_data)
                        elif len(frame_data.shape) == 3 and frame_data.shape[2] <= 3:
                            # 如果是彩色图像，转换为灰度
                            if frame_data.shape[2] == 3:
                                gray_frame = np.dot(frame_data[..., :3], [0.299, 0.587, 0.114])
                                stack_frames.append(gray_frame)
                            else:
                                stack_frames.append(frame_data[:, :, 0])
                    
                    if 'bf_hessian_tdv_result' in self.results:
                        method_names.append("BF+Hessian+TDV")
                        if len(self.results['bf_hessian_tdv_result'].shape) == 3:
                            if internal_frame_idx < self.results['bf_hessian_tdv_result'].shape[0]:
                                frame_data = self.results['bf_hessian_tdv_result'][internal_frame_idx]
                            else:
                                frame_data = self.results['bf_hessian_tdv_result'][-1]
                        else:
                            frame_data = self.results['bf_hessian_tdv_result']
                        
                        # 确保数据是2D的
                        if len(frame_data.shape) == 2:
                            stack_frames.append(frame_data)
                        elif len(frame_data.shape) == 3 and frame_data.shape[2] <= 3:
                            # 如果是彩色图像，转换为灰度
                            if frame_data.shape[2] == 3:
                                gray_frame = np.dot(frame_data[..., :3], [0.299, 0.587, 0.114])
                                stack_frames.append(gray_frame)
                            else:
                                stack_frames.append(frame_data[:, :, 0])
                    
                    if 'hybrid_result' in self.results:
                        method_names.append("Hybrid")
                        if len(self.results['hybrid_result'].shape) == 3:
                            if internal_frame_idx < self.results['hybrid_result'].shape[0]:
                                frame_data = self.results['hybrid_result'][internal_frame_idx]
                            else:
                                frame_data = self.results['hybrid_result'][-1]
                        else:
                            frame_data = self.results['hybrid_result']
                        
                        # 确保数据是2D的
                        if len(frame_data.shape) == 2:
                            stack_frames.append(frame_data)
                        elif len(frame_data.shape) == 3 and frame_data.shape[2] <= 3:
                            # 如果是彩色图像，转换为灰度
                            if frame_data.shape[2] == 3:
                                gray_frame = np.dot(frame_data[..., :3], [0.299, 0.587, 0.114])
                                stack_frames.append(gray_frame)
                            else:
                                stack_frames.append(frame_data[:, :, 0])
                    
                    if stack_frames:
                        # 确保所有帧具有相同的形状
                        first_shape = stack_frames[0].shape
                        processed_frames = []
                        
                        for frame in stack_frames:
                            if frame.shape == first_shape:
                                processed_frames.append(frame)
                            else:
                                # 如果形状不匹配，尝试调整大小（使用最近邻插值）
                                try:
                                    import cv2
                                    resized_frame = cv2.resize(frame, (first_shape[1], first_shape[0]), interpolation=cv2.INTER_NEAREST)
                                    processed_frames.append(resized_frame)
                                except:
                                    # 如果调整大小失败，跳过这一帧
                                    continue
                        
                        if processed_frames:
                            # 堆叠帧 - 确保是float32类型
                            stack_data = np.stack(processed_frames, axis=0).astype(np.float32)
                            
                            # 修改：使用用户看到的帧编号（1-based）来命名文件
                            stack_name = f"{self.args.fname}_Frame{user_frame_idx}_Comparison"
                            
                            filename = self.save_image_data(stack_data, stack_name, save_dir)
                            if filename:
                                success_count += 1
                        
                    self.progress_update_signal.emit((success_count * 100) // total_files)
                    QApplication.processEvents()
                    
                except Exception as e:
                    print(f"Error exporting frame stack: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 重置进度条为空的灰色
            self.progress_update_signal.emit(0)
            
            if success_count > 0:
                if export_options['frame_stack']:
                    self.status_label.setText(f"Exported {success_count} file(s) including comparison to {save_dir}")
                else:
                    self.status_label.setText(f"Exported {success_count} image(s) to {save_dir}")
            else:
                self.status_label.setText("No files were exported")
                
        except Exception as e:
            self.progress_update_signal.emit(0)
            QMessageBox.critical(self, "Export Error", f"Failed to export images: {str(e)}")
            self.status_label.setText(f"Error exporting images: {str(e)}")

    def collect_all_parameters(self):
        """收集所有参数到字典"""
        params = {}
        
        # 设备参数
        device_text = self.device_combo.currentText()
        if '(' in device_text:
            params['device'] = device_text.split('(')[0].strip()
        else:
            params['device'] = device_text
        
        # BF-SIM参数
        params['bf_sim'] = self.bf_sim_widget.get_values()
        
        # Hessian参数
        params['hessian'] = self.hessian_widget.get_values()
        
        # TDV参数
        params['tdv'] = self.tdv_widget.get_values()
        
        # Sparse参数
        params['sparse'] = self.sparse_widget.get_values()
        
        # 时间戳
        params['timestamp'] = QDateTime.currentDateTime().toString()
        
        return params
    
    def set_parameters_from_dict(self, params):
        """从字典设置参数"""
        try:
            # 设置设备
            if 'device' in params:
                device = params['device']
                for i in range(self.device_combo.count()):
                    if device in self.device_combo.itemText(i):
                        self.device_combo.setCurrentIndex(i)
                        break
            
            # 设置BF-SIM参数
            if 'bf_sim' in params:
                for key, value in params['bf_sim'].items():
                    if key in self.bf_sim_widget.parameters:
                        self.bf_sim_widget.parameters[key].set_value(value)
            
            # 设置Hessian参数
            if 'hessian' in params:
                for key, value in params['hessian'].items():
                    if key in self.hessian_widget.parameters:
                        self.hessian_widget.parameters[key].set_value(value)
            
            # 设置TDV参数
            if 'tdv' in params:
                for key, value in params['tdv'].items():
                    if key in self.tdv_widget.parameters:
                        self.tdv_widget.parameters[key].set_value(value)
            
            # 设置Sparse参数
            if 'sparse' in params:
                for key, value in params['sparse'].items():
                    if key in self.sparse_widget.parameters:
                        self.sparse_widget.parameters[key].set_value(value)
                        
        except Exception as e:
            raise ValueError(f"Error setting parameters: {str(e)}")
    
    def collect_parameters(self):
        """收集所有参数"""
        # 提取设备名称（去除括号中的描述信息）
        device_text = self.device_combo.currentText()
        if '(' in device_text:
            self.args.device = device_text.split('(')[0].strip()
        else:
            self.args.device = device_text
        
        # 收集BF-SIM参数
        bf_values = self.bf_sim_widget.get_values()
        for key, value in bf_values.items():
            setattr(self.args, key, value)
        
        # 收集Hessian参数
        hessian_values = self.hessian_widget.get_values()
        for key, value in hessian_values.items():
            setattr(self.args, key, value)
        
        # 收集TDV参数
        tdv_values = self.tdv_widget.get_values()
        for key, value in tdv_values.items():
            setattr(self.args, key, value)
        
        # 收集Sparse参数
        sparse_values = self.sparse_widget.get_values()
        for key, value in sparse_values.items():
            setattr(self.args, key, value)
    
    def run_reconstruction(self, checked=False):
        """执行重建"""
        try:
            # 收集参数
            self.collect_parameters()
            
            # 重置进度条为空的灰色
            self.progress_update_signal.emit(0)
            
            # 禁用所有控件
            self.disable_controls()
            
            # 运行重建按钮保持禁用状态，直到重建完成
            self.run_btn.setEnabled(False)
            self.status_label.setText("Starting reconstruction ...")
            QApplication.processEvents()  # 立即更新UI
            
            # 清除之前的结果
            self.results = {}
            
            # 重置右侧面板，清空所有显示
            self.reset_right_panel()
            
            # 创建并启动重建线程
            self.reconstruction_thread = ReconstructionThread(self.args)
            self.reconstruction_thread.progress_signal.connect(self.update_progress_from_callback)
            self.reconstruction_thread.result_signal.connect(self.handle_reconstruction_result)
            self.reconstruction_thread.error_signal.connect(self.handle_reconstruction_error)
            self.reconstruction_thread.finished_signal.connect(self.handle_reconstruction_finished)
            
            self.reconstruction_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Reconstruction failed: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
            # 发生错误时重新启用控件
            self.enable_controls()
            self.run_btn.setEnabled(True)
    
    def handle_reconstruction_result(self, results):
        """处理重建结果"""
        # 存储结果
        self.results = results
        
        # 更新显示
        max_frames = 0
        
        # 获取所有图像的第一帧并显示（对整个3D体积归一化）
        current_frame = self.current_frame
        
        if 'bf_result' in results:
            self.bf_display.set_image(results['bf_result'], current_frame, normalize=True)
            if len(results['bf_result'].shape) == 3:
                max_frames = max(max_frames, results['bf_result'].shape[0])
        
        if 'bf_hessian_result' in results:
            self.bf_hessian_display.set_image(results['bf_hessian_result'], current_frame, normalize=True)
            if len(results['bf_hessian_result'].shape) == 3:
                max_frames = max(max_frames, results['bf_hessian_result'].shape[0])
        
        if 'bf_hessian_tdv_result' in results:
            self.bf_hessian_tdv_display.set_image(results['bf_hessian_tdv_result'], current_frame, normalize=True)
            if len(results['bf_hessian_tdv_result'].shape) == 3:
                max_frames = max(max_frames, results['bf_hessian_tdv_result'].shape[0])
        
        if 'hybrid_result' in results:
            self.hybrid_display.set_image(results['hybrid_result'], current_frame, normalize=True)
            if len(results['hybrid_result'].shape) == 3:
                max_frames = max(max_frames, results['hybrid_result'].shape[0])
        
        # 更新全局帧控制
        self.update_global_frame_control(max_frames)
        
        # 更新全局范围并启用显示区间控制
        self.update_global_range()
        
        # 启用导出按钮和Reset View按钮
        self.export_images_btn.setEnabled(True)
        self.reset_zoom_btn.setEnabled(True)  # 确保启用Reset View按钮
        
        self.status_label.setText("Reconstruction completed successfully !")
    
    def handle_reconstruction_error(self, error_msg):
        """处理重建错误"""
        QMessageBox.critical(self, "Error", f"Reconstruction failed: {error_msg}")
        self.status_label.setText(f"Error: {error_msg}")
    
    def handle_reconstruction_finished(self):
        """重建完成处理"""
        # 启用所有控件
        self.enable_controls()
        self.run_btn.setEnabled(True)
        
        # 确保进度条完成
        self.progress_update_signal.emit(100)
    
    def update_progress_from_callback(self, value):
        """从回调函数更新进度条"""
        # 使用信号来确保线程安全地更新UI
        self.progress_update_signal.emit(value)
    
    def set_progress(self, value):
        """外部调用的进度条更新接口"""
        # 这个方法可以从外部函数的for循环中调用
        # 注意：如果外部函数在非GUI线程中运行，需要使用信号
        self.progress_update_signal.emit(value)
    
    def toggle_video_playback(self):
        """切换视频播放/暂停状态"""
        if not self.results:
            return
        
        if self.is_video_playing:
            self.stop_video_playback()
        else:
            self.start_video_playback()
    
    def start_video_playback(self):
        """开始视频播放"""
        if not self.results or self.global_frame_slider.maximum() <= 0:
            return
        
        self.is_video_playing = True
        self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.play_pause_btn.setToolTip("Stop Video")
        self.video_timer.start(self.video_speed)
    
    def stop_video_playback(self):
        """停止视频播放"""
        self.is_video_playing = False
        self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_pause_btn.setToolTip("Play Video")
        self.video_timer.stop()
    
    def next_video_frame(self):
        """播放下一帧（循环播放）"""
        if not self.results or not self.is_video_playing:
            self.stop_video_playback()
            return
        
        current_value = self.global_frame_slider.value()
        max_value = self.global_frame_slider.maximum()
        
        # 计算下一帧索引
        next_value = current_value + 1
        
        # 循环播放：如果到达最后一帧，回到第一帧
        if next_value > max_value:
            next_value = 0
        
        # 设置标志，表示这是由视频播放触发的帧变化
        self._is_video_changing_frame = True
        
        # 更新滑块值（这会触发global_frame_changed）
        self.global_frame_slider.setValue(next_value)
    
    def closeEvent_(self, event):
        """关闭窗口事件"""
        # 停止视频播放
        self.stop_video_playback()
        
        # 如果重建线程正在运行，先停止它
        if self.reconstruction_thread and self.reconstruction_thread.isRunning():
            self.reconstruction_thread.stop()
            self.reconstruction_thread.wait()
        
        event.accept()
    
    def closeEvent(self, event):
        """关闭窗口事件"""
        # 停止视频播放
        self.stop_video_playback()
        
        # 如果重建线程正在运行，先停止它
        if self.reconstruction_thread and self.reconstruction_thread.isRunning():
            reply = QMessageBox.question(self, 'Reconstruction in Progress', 
                                        'Reconstruction process is still running. Do you want to stop it and exit?',
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.reconstruction_thread.stop()
                if not self.reconstruction_thread.wait(3000):
                    pass
                
                # 强制清理内存
                self.force_cleanup()
                
                event.accept()
            else:
                event.ignore()
        else:
            # 强制清理内存
            self.force_cleanup()
            event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion风格
    
    # 设置应用程序全局字体为Arial
    font = QFont("Arial", 9)
    app.setFont(font)
    
    # 设置应用程序样式
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    app.setPalette(palette)
    
    # 设置应用程序样式表
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f0f0f0;
        }
        QGroupBox {
            font-family: Arial;
            font-weight: bold;
            border: 2px solid #ccc;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QTabWidget::pane {
            border: 1px solid #ccc;
            background-color: white;
        }
        QTabBar::tab {
            font-family: Arial;
            background-color: #e0e0e0;
            padding: 8px 16px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: white;
            border-bottom: 2px solid #4CAF50;  /* 绿色 */
        }
        QLabel {
            font-family: Arial;
        }
        QPushButton {
            font-family: Arial;
        }
        QLineEdit {
            font-family: Arial;
        }
        QComboBox {
            font-family: Arial;
        }
        QSpinBox {
            font-family: Arial;
        }
        QDoubleSpinBox {
            font-family: Arial;
        }
        QProgressBar {
            font-family: Arial;
        }
        QSlider {
            font-family: Arial;
        }
    """)
    
    window = SIMReconstructionGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()