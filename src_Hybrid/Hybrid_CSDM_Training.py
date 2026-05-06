import sys
import os
import json
from pathlib import Path
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import gc
import torch

# 导入训练相关模块
from src.default_config import default_config_TDV_FM_training
from src.FM_Dataset import FM_Dataset
from src_TDV.TDV_DNN_Train import TDV_FM_DNN_Train

# 定义全局变量
Microscopy_Type = "CSDM"

# 获取软件根目录路径
def get_root_directory():
    """获取当前文件的上级目录"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    return parent_dir

class TrainingVisualizationWidget(QWidget):
    reset_progress_signal = pyqtSignal()
    best_loss_curves_signal = pyqtSignal(list, list, list, list)
    """训练可视化控件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 训练数据存储
        self.training_loss_history = []  # 存储每个batch的损失
        self.valid_loss_history = []     # 存储每个epoch的验证损失
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 0
        
        # 验证图像数据
        self.valid_input_images = None
        self.valid_output_images = None
        self.valid_gt_images = None
        self.valid_display_frame = 0
        self.valid_total_frames = 0

        # 新增：存储最佳损失曲线数据
        self.best_training_loss_history = []  # 最佳验证损失对应的训练损失
        self.best_valid_loss_history = []     # 最佳验证损失对应的验证损失
        self.best_batch_indices = []          # 批次索引
        self.best_epoch_indices = []          # epoch索引
        self.best_val_loss = float('inf')     # 最佳验证损失
        
        # 模型路径
        self.model_path = ""
        
        # 用于同步图像缩放和平移的变量
        self.is_syncing_zoom = False
        self.is_syncing_pan = False

        # 新增：训练状态标志
        self.training_in_progress = False
        self.pending_reset = False
        
        self.init_ui()

    def set_training_state(self, in_progress, pending_reset=False):
        """设置训练状态"""
        self.training_in_progress = in_progress
        self.pending_reset = pending_reset

    def handle_model_reset(self):
        """处理模型重置的情况"""
        # 清空所有历史数据
        self.training_loss_history.clear()
        self.valid_loss_history.clear()
        self.current_epoch = 0
        self.current_batch = 0
        
        # 重置最佳损失曲线
        self.best_training_loss_history.clear()
        self.best_valid_loss_history.clear()
        self.best_batch_indices.clear()
        self.best_epoch_indices.clear()
        self.best_val_loss = float('inf')
        
        # 重新初始化曲线图
        self.initialize_training_loss_canvas()
        self.initialize_valid_loss_canvas()

        # 重置标志：允许后续的损失数据更新
        self.pending_reset = False  # 添加这一行！
        
        # 清空日志（可选）
        # self.clear_log()
        
        # self._log("Model reinitialized, loss curves reset", is_debug=True)

    def handle_training_stopped(self):
        """处理训练停止的情况"""
        # 不清除损失曲线和当前epoch
        # 只重置图像相关的内容
        
        # 保存当前的模型路径状态
        model_path_exists = os.path.exists(self.model_path) if self.model_path else False
        
        # 重置三张图像
        self.valid_input_images = None
        self.valid_output_images = None
        self.valid_gt_images = None
        self.valid_total_frames = 0
        
        self.input_display.clear()
        self.output_display.clear()
        self.gt_display.clear()
        
        # 重置控件状态
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)
        self.frame_label.setText("0/0")
        
        self.min_slider.setValue(0)
        self.max_slider.setValue(1000)
        self.min_slider.setEnabled(False)
        self.max_slider.setEnabled(False)
        self.reset_range_btn.setEnabled(False)
        self.reset_view_btn.setEnabled(False)
        
        # 恢复模型路径按钮状态
        self.model_path_btn.setEnabled(model_path_exists and self.current_epoch >= 1)

    # 新增方法
    def set_best_loss_curves_(self, training_losses, batch_indices, valid_losses, epoch_indices):
        """设置并显示最佳损失曲线"""
        self.best_training_loss_history = training_losses
        self.best_valid_loss_history = valid_losses
        self.best_batch_indices = batch_indices
        self.best_epoch_indices = epoch_indices
        
        # 更新曲线显示
        self.update_best_training_loss_curve()
        self.update_best_valid_loss_curve()
    def set_best_loss_curves(self, training_losses, batch_indices, valid_losses, epoch_indices):
        """设置并显示最佳损失曲线"""
        # 只有在训练进行中或训练正常完成时才更新最佳损失曲线
        if self.training_in_progress or not self.pending_reset:
            self.best_training_loss_history = training_losses.copy() if training_losses else []
            self.best_valid_loss_history = valid_losses.copy() if valid_losses else []
            self.best_batch_indices = batch_indices.copy() if batch_indices else []
            self.best_epoch_indices = epoch_indices.copy() if epoch_indices else []
            
            # 更新曲线显示
            self.update_best_training_loss_curve()
            self.update_best_valid_loss_curve()

    def update_best_training_loss_curve(self):
        """更新最佳训练损失曲线"""
        if not self.best_training_loss_history:
            return
            
        # 清除当前曲线
        self.training_loss_canvas.axes.clear()
        
        # 绘制最佳训练损失曲线（蓝色）
        losses = [loss * 10 for loss in self.best_training_loss_history]
        self.training_loss_canvas.axes.plot(
            self.best_batch_indices, 
            losses, 
            color='#2196F3', 
            linewidth=2
        )
        
        # 设置图表属性（保持与原有相同样式）
        self._setup_training_loss_axes()
        self.training_loss_canvas.draw()
    
    def update_best_valid_loss_curve(self):
        """更新最佳验证损失曲线"""
        if not self.best_valid_loss_history:
            return
            
        # 清除当前曲线
        self.valid_loss_canvas.axes.clear()
        
        # 绘制最佳验证损失曲线（绿色）
        losses = [loss * 10 for loss in self.best_valid_loss_history]
        self.valid_loss_canvas.axes.plot(
            self.best_epoch_indices, 
            losses, 
            color='#4CAF50', 
            linewidth=2
        )
        
        # 设置图表属性（保持与原有相同样式）
        self._setup_valid_loss_axes()
        self.valid_loss_canvas.draw()
    
    def _setup_training_loss_axes(self):
        """设置训练损失坐标轴属性（复用原有样式）"""
        # 这里复制原有initialize_training_loss_canvas的样式设置
        self.training_loss_canvas.fig.patch.set_alpha(0.0)
        self.training_loss_canvas.axes.patch.set_alpha(0.0)
        
        for spine in self.training_loss_canvas.axes.spines.values():
            spine.set_color('#000000')
            spine.set_linewidth(1.0)
        
        self.training_loss_canvas.axes.tick_params(axis='x', colors='#000000', labelsize=8)
        self.training_loss_canvas.axes.tick_params(axis='y', colors='#000000', labelsize=8)
        self.training_loss_canvas.axes.xaxis.label.set_color('#000000')
        self.training_loss_canvas.axes.yaxis.label.set_color('#000000')
        
        self.training_loss_canvas.axes.set_xlabel('Batch Number', fontsize=9, color='#000000')
        self.training_loss_canvas.axes.set_ylabel('Training Loss (×0.1)', fontsize=9, color='#000000')
        self.training_loss_canvas.axes.grid(True, alpha=0.3)
        
        # 设置y轴范围
        if self.best_training_loss_history:
            losses = [loss * 10 for loss in self.best_training_loss_history]
            min_loss = min(losses)
            max_loss = max(losses)
            
            if min_loss == max_loss:
                min_loss = max(0, min_loss - 0.1)
                max_loss = max_loss + 0.1
            
            y_margin = (max_loss - min_loss) * 0.1
            if y_margin == 0:
                y_margin = 0.1
                
            y_min = max(0, min_loss - y_margin)
            y_max = max_loss + y_margin
            
            self.training_loss_canvas.axes.set_ylim(y_min, y_max)
            self.training_loss_canvas.axes.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            
            import matplotlib.ticker as ticker
            locator = ticker.MaxNLocator(nbins=6, prune='both')
            self.training_loss_canvas.axes.yaxis.set_major_locator(locator)
        else:
            self.training_loss_canvas.axes.set_ylim(0, 1.0)
            self.training_loss_canvas.axes.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        
        # 设置x轴范围
        if self.best_batch_indices and len(self.best_batch_indices) > 1:
            self.training_loss_canvas.axes.set_xlim(0, self.best_batch_indices[-1] * 1.1)
        else:
            self.training_loss_canvas.axes.set_xlim(0, 10)
        
        self.training_loss_canvas.fig.subplots_adjust(left=0.15, right=0.95, top=0.93, bottom=0.15)
    
    def _setup_valid_loss_axes(self):
        """设置验证损失坐标轴属性（复用原有样式）"""
        # 这里复制原有initialize_valid_loss_canvas的样式设置
        self.valid_loss_canvas.fig.patch.set_alpha(0.0)
        self.valid_loss_canvas.axes.patch.set_alpha(0.0)
        
        for spine in self.valid_loss_canvas.axes.spines.values():
            spine.set_color('#000000')
            spine.set_linewidth(1.0)
        
        self.valid_loss_canvas.axes.tick_params(axis='x', colors='#000000', labelsize=8)
        self.valid_loss_canvas.axes.tick_params(axis='y', colors='#000000', labelsize=8)
        self.valid_loss_canvas.axes.xaxis.label.set_color('#000000')
        self.valid_loss_canvas.axes.yaxis.label.set_color('#000000')
        
        self.valid_loss_canvas.axes.set_xlabel('Epoch Number', fontsize=9, color='#000000')
        self.valid_loss_canvas.axes.set_ylabel('Validation Loss (×0.1)', fontsize=9, color='#000000')
        self.valid_loss_canvas.axes.grid(True, alpha=0.3)
        
        # 设置y轴范围
        if self.best_valid_loss_history:
            losses = [loss * 10 for loss in self.best_valid_loss_history]
            min_loss = min(losses)
            max_loss = max(losses)
            
            if min_loss == max_loss:
                min_loss = max(0, min_loss - 0.1)
                max_loss = max_loss + 0.1
            
            y_margin = (max_loss - min_loss) * 0.1
            if y_margin == 0:
                y_margin = 0.1
                
            y_min = max(0, min_loss - y_margin)
            y_max = max_loss + y_margin
            
            self.valid_loss_canvas.axes.set_ylim(y_min, y_max)
            self.valid_loss_canvas.axes.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            
            import matplotlib.ticker as ticker
            locator = ticker.MaxNLocator(nbins=6, prune='both')
            self.valid_loss_canvas.axes.yaxis.set_major_locator(locator)
        else:
            self.valid_loss_canvas.axes.set_ylim(0, 1.0)
            self.valid_loss_canvas.axes.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        
        # 设置x轴范围
        if self.best_epoch_indices and len(self.best_epoch_indices) > 1:
            self.valid_loss_canvas.axes.set_xlim(0, self.best_epoch_indices[-1] * 1.1)
        else:
            self.valid_loss_canvas.axes.set_xlim(0, 10)
        
        self.valid_loss_canvas.fig.subplots_adjust(left=0.15, right=0.95, top=0.93, bottom=0.15)

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 10, 5, 5)
        layout.setSpacing(5.5)
        
        # 1. 标题
        title_label = QLabel(f"Hybrid-{Microscopy_Type} Training Visualization")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setStyleSheet("padding: 5px; color: #000000;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        
        # 2. 损失曲线图区域
        curves_group = QGroupBox("Training Curves")
        curves_group.setFont(QFont("Arial", 10, QFont.Bold))
        curves_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #ccc;
                border-radius: 5px;
                margin-top: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                color: #000000;
            }
        """)
        curves_layout = QHBoxLayout()
        curves_layout.setContentsMargins(5, 5, 5, 5)
        curves_layout.setSpacing(10)
        
        # 训练损失曲线图（左）
        self.training_loss_canvas = MPLCanvas(self, width=4, height=2.5, dpi=100)
        self.training_loss_canvas.setMinimumHeight(250)
        self.training_loss_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.initialize_training_loss_canvas()
        curves_layout.addWidget(self.training_loss_canvas)
        
        # 验证损失曲线图（右）
        self.valid_loss_canvas = MPLCanvas(self, width=4, height=2.5, dpi=100)
        self.valid_loss_canvas.setMinimumHeight(250)
        self.valid_loss_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.initialize_valid_loss_canvas()
        curves_layout.addWidget(self.valid_loss_canvas)
        
        curves_group.setLayout(curves_layout)
        layout.addWidget(curves_group)
        
        # 3. 验证结果可视化区域
        images_group = QGroupBox("Training Results")
        images_group.setFont(QFont("Arial", 10, QFont.Bold))
        images_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #ccc;
                border-radius: 5px;
                margin-top: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                color: #000000;
            }
        """)
        images_layout = QVBoxLayout()
        images_layout.setContentsMargins(5, 0, 5, 5)
        images_layout.setSpacing(5)
        
        # 图像显示行
        images_row = QHBoxLayout()
        images_row.setContentsMargins(0, 0, 0, 0)
        images_row.setSpacing(10)
        
        # 输入图像
        self.input_display = ImageDisplayContainer("TDV-DNN Input")
        images_row.addWidget(self.input_display)
        
        # 输出图像
        self.output_display = ImageDisplayContainer("TDV-DNN Output")
        images_row.addWidget(self.output_display)
        
        # 真值图像
        self.gt_display = ImageDisplayContainer("Ground Truth")
        images_row.addWidget(self.gt_display)
        
        images_layout.addLayout(images_row)
        
        # 4. 控制按钮区域 - Display Range行
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)
        controls_layout.setContentsMargins(5, 0, 5, 5)

        # 显示区间标签
        range_label = QLabel("Display Range:")
        range_label.setFixedWidth(85)
        range_label.setFont(QFont("Arial", 9))
        range_label.setStyleSheet("color: #000000;")
        controls_layout.addWidget(range_label)
        
        # 最小值滑块
        self.min_slider = QSlider(Qt.Horizontal)
        self.min_slider.setRange(0, 1000)
        self.min_slider.setValue(0)
        self.min_slider.setEnabled(False)
        self.min_slider.valueChanged.connect(self.display_range_changed)
        self.min_slider.setMinimumWidth(110)
        controls_layout.addWidget(self.min_slider)
        
        # 最大值滑块
        self.max_slider = QSlider(Qt.Horizontal)
        self.max_slider.setRange(0, 1000)
        self.max_slider.setValue(1000)
        self.max_slider.setEnabled(False)
        self.max_slider.valueChanged.connect(self.display_range_changed)
        self.max_slider.setMinimumWidth(110)
        controls_layout.addWidget(self.max_slider)
        
        # 重置显示区间按钮
        self.reset_range_btn = QPushButton("Reset Range")
        self.reset_range_btn.setFont(QFont("Arial", 9))
        self.reset_range_btn.setFixedHeight(30)
        self.reset_range_btn.setFixedWidth(98)
        self.reset_range_btn.setEnabled(False)
        self.reset_range_btn.clicked.connect(self.reset_display_range)
        controls_layout.addWidget(self.reset_range_btn)
        
        # Reset View按钮
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.setFont(QFont("Arial", 9))
        self.reset_view_btn.setFixedHeight(30)
        self.reset_view_btn.setFixedWidth(98)
        self.reset_view_btn.setEnabled(False)
        self.reset_view_btn.clicked.connect(self.reset_view_all)
        controls_layout.addWidget(self.reset_view_btn)
        
        # 添加弹簧，使View Model按钮右对齐
        controls_layout.addStretch()
        
        # 模型路径可视化按钮
        self.model_path_btn = QPushButton("Check Model")
        self.model_path_btn.setFont(QFont("Arial", 9))
        self.model_path_btn.setFixedHeight(30)
        self.model_path_btn.setFixedWidth(98)
        self.model_path_btn.setEnabled(False)
        self.model_path_btn.clicked.connect(self.view_model_path)
        self.model_path_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 4px;
                padding: 6px;
                border: 1px solid #388E3C;
            }
            QPushButton:hover {
                background-color: #45a049;
                border-color: #2E7D32;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                border-color: #999999;
            }
        """)
        controls_layout.addWidget(self.model_path_btn)
        
        images_layout.addLayout(controls_layout)

        # 5. Frame控制行
        frame_control_layout = QHBoxLayout()
        frame_control_layout.setSpacing(8)
        frame_control_layout.setContentsMargins(5, 0, 5, 5)
        
        # Frame标签
        frame_label = QLabel("Frame:")
        frame_label.setFixedWidth(85)
        frame_label.setFont(QFont("Arial", 9))
        frame_label.setStyleSheet("color: #000000;")
        frame_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        frame_control_layout.addWidget(frame_label)
        
        # 滑块
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.update_validation_frame)
        self.frame_slider.setMinimumWidth(250)  # 修改：加长滑块
        frame_control_layout.addWidget(self.frame_slider, 1)  # 添加拉伸因子
        
        # 帧数标签
        self.frame_label = QLabel("0/0")
        self.frame_label.setFixedWidth(50)
        self.frame_label.setFont(QFont("Arial", 9))
        self.frame_label.setStyleSheet("color: #000000;")
        self.frame_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        frame_control_layout.addWidget(self.frame_label)
        
        images_layout.addLayout(frame_control_layout)
        
        images_group.setLayout(images_layout)
        layout.addWidget(images_group)

        # 6. 日志显示区域
        log_group = QGroupBox("Training Log")        
        log_group.setFont(QFont("Arial", 10, QFont.Bold))
        log_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #ccc;
                border-radius: 5px;
                margin-top: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                color: #000000;
            }
        """)

        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(5, 5, 5, 5)
        
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setFont(QFont("Arial", 9))
        self.log_widget.setMaximumHeight(170)
        
        self.log_widget.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ccc;
                background-color: #f8f8f8;
                padding: 5px;
                font-family: Arial;
                color: #000000;
            }
        """)
        log_layout.addWidget(self.log_widget)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)    
        
        self.setLayout(layout)
        
        # 初始化全局范围变量
        self.global_min_val = 0
        self.global_max_val = 1
        self.display_min_val = 0
        self.display_max_val = 1
        
    def initialize_training_loss_canvas(self):
        """初始化训练损失曲线图"""
        self.training_loss_canvas.axes.clear()
        self.training_loss_canvas.fig.patch.set_alpha(0.0)
        self.training_loss_canvas.axes.patch.set_alpha(0.0)

        for spine in self.training_loss_canvas.axes.spines.values():
            spine.set_color('#000000')
            spine.set_linewidth(1.0)

        self.training_loss_canvas.axes.tick_params(axis='x', colors='#000000', labelsize=8)
        self.training_loss_canvas.axes.tick_params(axis='y', colors='#000000', labelsize=8)
        self.training_loss_canvas.axes.xaxis.label.set_color('#000000')
        self.training_loss_canvas.axes.yaxis.label.set_color('#000000')
        
        self.training_loss_canvas.axes.set_xlabel('Batch Number', fontsize=9, color='#000000')
        self.training_loss_canvas.axes.set_ylabel('Training Loss (×0.1)', fontsize=9, color='#000000')
        self.training_loss_canvas.axes.grid(True, alpha=0.3)
        self.training_loss_canvas.axes.set_ylim(0, 1.0)
        self.training_loss_canvas.axes.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        self.training_loss_canvas.axes.set_xlim(0, 10)
        
        self.training_loss_canvas.fig.subplots_adjust(left=0.15, right=0.95, top=0.93, bottom=0.15)
        self.training_loss_canvas.draw()
        
    def initialize_valid_loss_canvas(self):
        """初始化验证损失曲线图"""
        self.valid_loss_canvas.axes.clear()
        self.valid_loss_canvas.fig.patch.set_alpha(0.0)
        self.valid_loss_canvas.axes.patch.set_alpha(0.0)

        for spine in self.valid_loss_canvas.axes.spines.values():
            spine.set_color('#000000')
            spine.set_linewidth(1.0)
        
        self.valid_loss_canvas.axes.tick_params(axis='x', colors='#000000', labelsize=8)
        self.valid_loss_canvas.axes.tick_params(axis='y', colors='#000000', labelsize=8)
        self.valid_loss_canvas.axes.xaxis.label.set_color('#000000')
        self.valid_loss_canvas.axes.yaxis.label.set_color('#000000')
        
        self.valid_loss_canvas.axes.set_xlabel('Epoch Number', fontsize=9, color='#000000')
        self.valid_loss_canvas.axes.set_ylabel('Validation Loss (×0.1)', fontsize=9, color='#000000')
        self.valid_loss_canvas.axes.grid(True, alpha=0.3)
        self.valid_loss_canvas.axes.set_ylim(0, 1.0)
        self.valid_loss_canvas.axes.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        self.valid_loss_canvas.axes.set_xlim(0, 10)
        
        self.valid_loss_canvas.fig.subplots_adjust(left=0.15, right=0.95, top=0.93, bottom=0.15)
        self.valid_loss_canvas.draw()
        
    def add_training_loss(self, batch_idx, loss):
        # 如果正在重置，忽略新的损失数据
        if self.pending_reset:
            return

        """添加训练损失数据点"""
        self.training_loss_history.append((batch_idx, loss))
        self.update_training_loss_curve()
        
    def add_valid_loss(self, epoch, loss):
        # 如果正在重置，忽略新的损失数据
        if self.pending_reset:
            return

        if epoch == -1:  # 与训练类中的-1对应
            self.reset_visualization()
            # self.progress_bar.setValue(0)
            return

        """添加验证损失数据点"""
        self.valid_loss_history.append((epoch, loss))
        self.current_epoch = epoch
        self.update_valid_loss_curve()
    
    def reset_visualization(self):
        """重置可视化内容但保留模型路径"""
        # 保存当前的模型路径状态
        model_path_exists = os.path.exists(self.model_path) if self.model_path else False
        model_path_enabled = self.model_path_btn.isEnabled()
        self.reset_progress_signal.emit()
        # 重置损失曲线
        self.training_loss_history.clear()
        self.valid_loss_history.clear()
        self.current_epoch = 0
        self.current_batch = 0
        self.initialize_training_loss_canvas()
        self.initialize_valid_loss_canvas()
        # 重置最佳损失曲线
        self.best_training_loss_history.clear()
        self.best_valid_loss_history.clear()
        self.best_batch_indices.clear()
        self.best_epoch_indices.clear()
        self.best_val_loss = float('inf')
        # 重置三张图像
        self.valid_input_images = None
        self.valid_output_images = None
        self.valid_gt_images = None
        self.valid_total_frames = 0
        
        self.input_display.clear()
        self.output_display.clear()
        self.gt_display.clear()
        
        # 重置控件状态
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)
        self.frame_label.setText("0/0")
        
        self.min_slider.setValue(0)
        self.max_slider.setValue(1000)
        self.min_slider.setEnabled(False)
        self.max_slider.setEnabled(False)
        self.reset_range_btn.setEnabled(False)
        self.reset_view_btn.setEnabled(False)
        
        # 恢复模型路径按钮状态
        self.model_path_btn.setEnabled(model_path_enabled and model_path_exists)

    def update_training_loss_curve(self):
        """更新训练损失曲线"""
        # 如果正在重置，不更新曲线
        if self.pending_reset:
            return
        self.training_loss_canvas.axes.clear()
        self.training_loss_canvas.fig.patch.set_alpha(0.0)
        self.training_loss_canvas.axes.patch.set_alpha(0.0)

        for spine in self.training_loss_canvas.axes.spines.values():
            spine.set_color('#000000')
            spine.set_linewidth(1.0)
        
        self.training_loss_canvas.axes.tick_params(axis='x', colors='#000000', labelsize=8)
        self.training_loss_canvas.axes.tick_params(axis='y', colors='#000000', labelsize=8)
        self.training_loss_canvas.axes.xaxis.label.set_color('#000000')
        self.training_loss_canvas.axes.yaxis.label.set_color('#000000')
        
        if self.training_loss_history:
            batches = [x[0] for x in self.training_loss_history]
            losses = [x[1] * 10 for x in self.training_loss_history]
            self.training_loss_canvas.axes.plot(batches, losses, color='#2196F3', linewidth=2)
        
        self.training_loss_canvas.axes.set_xlabel('Batch Number', fontsize=9, color='#000000')
        self.training_loss_canvas.axes.set_ylabel('Training Loss (×0.1)', fontsize=9, color='#000000')
        self.training_loss_canvas.axes.grid(True, alpha=0.3)
        
        if self.training_loss_history:
            losses = [x[1] * 10 for x in self.training_loss_history]
            min_loss = min(losses)
            max_loss = max(losses)
            
            if min_loss == max_loss:
                min_loss = max(0, min_loss - 0.1)
                max_loss = max_loss + 0.1
            
            y_margin = (max_loss - min_loss) * 0.1
            if y_margin == 0:
                y_margin = 0.1
                
            y_min = max(0, min_loss - y_margin)
            y_max = max_loss + y_margin
            
            self.training_loss_canvas.axes.set_ylim(y_min, y_max)
            self.training_loss_canvas.axes.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            
            import matplotlib.ticker as ticker
            locator = ticker.MaxNLocator(nbins=6, prune='both')
            self.training_loss_canvas.axes.yaxis.set_major_locator(locator)
        else:
            self.training_loss_canvas.axes.set_ylim(0, 1.0)
            self.training_loss_canvas.axes.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        
        if self.training_loss_history and len(self.training_loss_history) > 1:
            batches = [x[0] for x in self.training_loss_history]
            self.training_loss_canvas.axes.set_xlim(0, batches[-1] * 1.1)
        else:
            self.training_loss_canvas.axes.set_xlim(0, 10)
        
        self.training_loss_canvas.fig.subplots_adjust(left=0.15, right=0.95, top=0.93, bottom=0.15)
        self.training_loss_canvas.draw()
        
    def update_valid_loss_curve(self):
        """更新验证损失曲线"""
        # 如果正在重置，不更新曲线
        if self.pending_reset:
            return

        self.valid_loss_canvas.axes.clear()
        self.valid_loss_canvas.fig.patch.set_alpha(0.0)
        self.valid_loss_canvas.axes.patch.set_alpha(0.0)

        for spine in self.valid_loss_canvas.axes.spines.values():
            spine.set_color('#000000')
            spine.set_linewidth(1.0)
        
        self.valid_loss_canvas.axes.tick_params(axis='x', colors='#000000', labelsize=8)
        self.valid_loss_canvas.axes.tick_params(axis='y', colors='#000000', labelsize=8)
        self.valid_loss_canvas.axes.xaxis.label.set_color('#000000')
        self.valid_loss_canvas.axes.yaxis.label.set_color('#000000')
        
        if self.valid_loss_history:
            epochs = [x[0] for x in self.valid_loss_history]
            losses = [x[1] * 10 for x in self.valid_loss_history]
            self.valid_loss_canvas.axes.plot(epochs, losses, color='#4CAF50', linewidth=2)
        
        self.valid_loss_canvas.axes.set_xlabel('Epoch Number', fontsize=9, color='#000000')
        self.valid_loss_canvas.axes.set_ylabel('Validation Loss (×0.1)', fontsize=9, color='#000000')
        self.valid_loss_canvas.axes.grid(True, alpha=0.3)
        
        if self.valid_loss_history:
            losses = [x[1] * 10 for x in self.valid_loss_history]
            min_loss = min(losses)
            max_loss = max(losses)
            
            if min_loss == max_loss:
                min_loss = max(0, min_loss - 0.1)
                max_loss = max_loss + 0.1
            
            y_margin = (max_loss - min_loss) * 0.1
            if y_margin == 0:
                y_margin = 0.1
                
            y_min = max(0, min_loss - y_margin)
            y_max = max_loss + y_margin
            
            self.valid_loss_canvas.axes.set_ylim(y_min, y_max)
            self.valid_loss_canvas.axes.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            
            import matplotlib.ticker as ticker
            locator = ticker.MaxNLocator(nbins=6, prune='both')
            self.valid_loss_canvas.axes.yaxis.set_major_locator(locator)
        else:
            self.valid_loss_canvas.axes.set_ylim(0, 1.0)
            self.valid_loss_canvas.axes.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        
        if self.valid_loss_history and len(self.valid_loss_history) > 1:
            epochs = [x[0] for x in self.valid_loss_history]
            self.valid_loss_canvas.axes.set_xlim(0, epochs[-1] * 1.1)
        else:
            self.valid_loss_canvas.axes.set_xlim(0, 10)
        
        self.valid_loss_canvas.fig.subplots_adjust(left=0.15, right=0.95, top=0.93, bottom=0.15)
        self.valid_loss_canvas.draw()
        
    def set_validation_images(self, input_images, output_images, gt_images):
        """设置验证图像数据 - 完全刷新所有显示内容"""
        max_frames = 100
        if input_images.shape[0] > max_frames:
            input_images = input_images[:max_frames]
            output_images = output_images[:max_frames]
            gt_images = gt_images[:max_frames]
            
        self.valid_input_images = input_images
        self.valid_output_images = output_images
        self.valid_gt_images = gt_images
        self.valid_total_frames = input_images.shape[0]
        self.valid_display_frame = 0  # 重置为第一帧
        
        # 重置滑块和标签
        self.frame_slider.setRange(0, self.valid_total_frames - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(self.valid_total_frames > 0)
        self.frame_label.setText(f"1/{self.valid_total_frames}")
        
        # 重置显示区间
        self.update_global_range()
        self.reset_display_range()
        
        # 重置所有图像视图
        self.reset_view_all()
        
        # 强制更新到第1帧
        self.update_validation_frame(0)
        
        # 启用相关控件
        self.reset_view_btn.setEnabled(True)
        self.reset_range_btn.setEnabled(True)
        self.min_slider.setEnabled(True)
        self.max_slider.setEnabled(True)
        
    def update_validation_frame_(self, frame_idx):
        """更新验证图像显示"""
        if (self.valid_input_images is None or 
            self.valid_output_images is None or 
            self.valid_gt_images is None):
            return
            
        if frame_idx >= self.valid_total_frames:
            frame_idx = self.valid_total_frames - 1
            
        self.valid_display_frame = frame_idx
        
        # 使用显示区间显示图像
        self.input_display.set_image(
            self.valid_input_images[frame_idx], 
            normalize=True, 
            use_display_range=True
        )
        self.output_display.set_image(
            self.valid_output_images[frame_idx], 
            normalize=True, 
            use_display_range=True
        )
        self.gt_display.set_image(
            self.valid_gt_images[frame_idx], 
            normalize=True, 
            use_display_range=True
        )
        
        # 更新帧标签
        self.frame_label.setText(f"{frame_idx + 1}/{self.valid_total_frames}")

    def update_validation_frame(self, frame_idx):
        """更新验证图像显示 - 保留缩放和平移状态"""
        if (self.valid_input_images is None or 
            self.valid_output_images is None or 
            self.valid_gt_images is None):
            return
            
        if frame_idx >= self.valid_total_frames:
            frame_idx = self.valid_total_frames - 1
            
        self.valid_display_frame = frame_idx
        
        # 获取当前缩放和平移状态（从输入图像控件）
        input_widget = self.input_display.image_widget
        if input_widget.original_qimage is not None:
            # 保存当前缩放因子和平移偏移
            saved_zoom = input_widget.zoom_factor
            saved_pan = QPoint(input_widget.pan_offset)
        else:
            saved_zoom = None
            saved_pan = None
        
        # 使用显示区间显示图像
        self.input_display.set_image(
            self.valid_input_images[frame_idx], 
            normalize=True, 
            use_display_range=True
        )
        self.output_display.set_image(
            self.valid_output_images[frame_idx], 
            normalize=True, 
            use_display_range=True
        )
        self.gt_display.set_image(
            self.valid_gt_images[frame_idx], 
            normalize=True, 
            use_display_range=True
        )
        
        # 恢复缩放和平移状态
        if saved_zoom is not None and saved_pan is not None:
            # 确保缩放因子不小于最小值
            input_widget.zoom_factor = max(saved_zoom, input_widget.min_zoom_factor)
            output_widget = self.output_display.image_widget
            gt_widget = self.gt_display.image_widget
            output_widget.zoom_factor = max(saved_zoom, output_widget.min_zoom_factor)
            gt_widget.zoom_factor = max(saved_zoom, gt_widget.min_zoom_factor)
            
            # 设置平移偏移
            input_widget.pan_offset = QPoint(saved_pan)
            output_widget.pan_offset = QPoint(saved_pan)
            gt_widget.pan_offset = QPoint(saved_pan)
            
            # 更新显示
            input_widget.update_display()
            output_widget.update_display()
            gt_widget.update_display()
            
            # 强制重绘
            input_widget.repaint()
            output_widget.repaint()
            gt_widget.repaint()
        
        # 更新帧标签
        self.frame_label.setText(f"{frame_idx + 1}/{self.valid_total_frames}")

    def update_global_range(self):
        """更新全局范围"""
        if (self.valid_input_images is None or 
            self.valid_output_images is None or 
            self.valid_gt_images is None):
            return
            
        # 获取所有图像的最小值和最大值
        all_min = min([
            self.valid_input_images.min(),
            self.valid_output_images.min(),
            self.valid_gt_images.min()
        ])
        all_max = max([
            self.valid_input_images.max(),
            self.valid_output_images.max(),
            self.valid_gt_images.max()
        ])
        
        self.global_min_val = all_min
        self.global_max_val = all_max
        self.display_min_val = all_min
        self.display_max_val = all_max
        
    def display_range_changed(self):
        """显示区间变化"""
        if self.valid_input_images is None:
            return
            
        min_val = self.min_slider.value() / 1000.0
        max_val = self.max_slider.value() / 1000.0
        
        if min_val >= max_val:
            min_val = max_val - 0.001
            if min_val < 0:
                min_val = 0
            self.min_slider.setValue(int(min_val * 1000))
        
        self.display_min_val = self.global_min_val + (self.global_max_val - self.global_min_val) * min_val
        self.display_max_val = self.global_min_val + (self.global_max_val - self.global_min_val) * max_val
        
        self.input_display.set_display_range(self.display_min_val, self.display_max_val)
        self.output_display.set_display_range(self.display_min_val, self.display_max_val)
        self.gt_display.set_display_range(self.display_min_val, self.display_max_val)
        
    def reset_display_range(self):
        """重置显示区间"""
        self.min_slider.setValue(0)
        self.max_slider.setValue(1000)
        
        if self.valid_input_images is not None:
            self.display_min_val = self.global_min_val
            self.display_max_val = self.global_max_val
            
            self.input_display.set_display_range(self.display_min_val, self.display_max_val)
            self.output_display.set_display_range(self.display_min_val, self.display_max_val)
            self.gt_display.set_display_range(self.display_min_val, self.display_max_val)
        
    def reset_view_all(self):
        """重置所有图像的视图"""
        self.input_display.reset_view()
        self.output_display.reset_view()
        self.gt_display.reset_view()
        
    def set_model_path(self, path):
        """设置模型路径并启用按钮（仅在epoch>=1时调用）"""
        self.model_path = path
        # self.model_path_btn.setEnabled(bool(path))
        self.model_path_btn.setEnabled(os.path.exists(path) and self.current_epoch >= 1)

    def view_model_path(self):
        """查看模型文件路径"""
        if not self.model_path or not os.path.exists(self.model_path):
            QMessageBox.warning(self, "Model Not Found", 
                            "Model file does not exist or path is not set.")
            return
        
        try:
            # 获取模型文件的绝对路径
            model_path = os.path.abspath(self.model_path)
            
            # 根据操作系统打开资源管理器并选中文件
            if sys.platform == "win32":
                # Windows: 使用explorer /select命令选中文件
                import subprocess
                subprocess.Popen(f'explorer /select,"{model_path}"', shell=True)
                
            elif sys.platform == "darwin":
                # macOS: 使用open -R命令在Finder中显示文件
                import subprocess
                subprocess.Popen(["open", "-R", model_path])
                
            else:
                # Linux和其他系统: 打开文件所在目录
                import subprocess
                # 尝试使用xdg-open打开目录
                try:
                    subprocess.Popen(["xdg-open", os.path.dirname(model_path)])
                except:
                    # 如果xdg-open不可用，尝试其他方法
                    try:
                        subprocess.Popen(["nautilus", "--select", model_path])
                    except:
                        try:
                            subprocess.Popen(["dolphin", "--select", model_path])
                        except:
                            # 最后尝试使用文件管理器打开目录
                            subprocess.Popen(["xdg-open", os.path.dirname(model_path)])
            
        except Exception as e:
            QMessageBox.warning(self, "Open Failed", 
                            f"Cannot open file explorer: {str(e)}")

    def view_model_path_(self):
        """查看模型文件路径"""
        if not self.model_path or not os.path.exists(self.model_path):
            QMessageBox.warning(self, "Model Not Found", 
                              "Model file does not exist or path is not set.")
            return
            
        if sys.platform == "win32":
            os.startfile(self.model_path)
        elif sys.platform == "darwin":
            os.system(f'open -R "{self.model_path}"')
        else:
            os.system(f'xdg-open "{os.path.dirname(self.model_path)}"')
            
    def add_log(self, message):
        """添加日志信息"""
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        log_line = f"[{timestamp}] {message}"
        self.log_widget.append(log_line)
        self.log_widget.verticalScrollBar().setValue(self.log_widget.verticalScrollBar().maximum())
        
    def clear_log(self):
        """清空日志"""
        self.log_widget.clear()
        
    def reset_all(self):
        """重置所有可视化内容"""
        self.training_loss_history.clear()
        self.valid_loss_history.clear()
        self.current_epoch = 0
        self.current_batch = 0
        
        self.valid_input_images = None
        self.valid_output_images = None
        self.valid_gt_images = None
        
        self.initialize_training_loss_canvas()
        self.initialize_valid_loss_canvas()
        
        self.input_display.clear()
        self.output_display.clear()
        self.gt_display.clear()
        
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setEnabled(False)
        self.frame_label.setText("0/0")
        
        self.min_slider.setValue(0)
        self.max_slider.setValue(1000)
        self.min_slider.setEnabled(False)
        self.max_slider.setEnabled(False)
        self.reset_range_btn.setEnabled(False)
        self.reset_view_btn.setEnabled(False)
        self.model_path_btn.setEnabled(False)

class MPLCanvas(FigureCanvas):
    """Matplotlib画布控件"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_alpha(0.0)
        self.axes = self.fig.add_subplot(111)
        self.axes.patch.set_alpha(0.0)
        
        super().__init__(self.fig)
        self.setParent(parent)
        self.setStyleSheet("background-color: transparent;")
        
        plt.style.use('seaborn-whitegrid')
        self.axes.tick_params(axis='both', which='major', labelsize=8)

        for spine in self.axes.spines.values():
            spine.set_color('#000000')
            spine.set_linewidth(1.0)
        
        self.axes.tick_params(axis='x', colors='#000000', labelsize=8)
        self.axes.tick_params(axis='y', colors='#000000', labelsize=8)
        self.axes.xaxis.label.set_color('#000000')
        self.axes.yaxis.label.set_color('#000000')
        
        self.axes.grid(True, alpha=0.3)

class ImageDisplayWidget(QLabel):
    """图像显示控件，支持3D数据和鼠标交互，对整个3D体积归一化，使用复制像素插值"""
    # 添加缩放变化信号
    zoom_changed = pyqtSignal(float, QPoint)  # 缩放因子, 平移偏移
    pan_changed = pyqtSignal(QPoint)  # 平移偏移
    
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
        self.setFixedSize(200, 200)  # 固定显示区域大小
        self.setStyleSheet("""
            QLabel {
                border: 1px solid #ccc;
                background-color: #f0f0f0;
                padding: 2px;
            }
        """)
        
        # 启用鼠标跟踪
        self.setMouseTracking(True)
    
    def set_image(self, image_array, normalize=True, use_display_range=False):
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
            frame_data = image_array[0]
        else:  # 2D数据
            self.total_frames = 1
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
        
        # 发射初始缩放变化信号
        self.zoom_changed.emit(self.zoom_factor, self.pan_offset)
    
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
            frame_data = self.original_image[0]
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
    
    def clear(self):
        """清空显示"""
        super().clear()
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
    """图像显示容器，包含标题"""
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 0, 5, 5)
        layout.setSpacing(0)
        
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 10, QFont.Bold))
        title_label.setFixedHeight(20)
        title_label.setStyleSheet("color: #000000;")
        layout.addWidget(title_label)
        
        self.image_widget = ImageDisplayWidget()
        layout.addWidget(self.image_widget)
        
        self.setLayout(layout)
        self.setFixedSize(220, 250)
        
    def set_image(self, image_data, normalize=True, use_display_range=False):
        """设置图像数据"""
        self.image_widget.set_image(image_data, normalize, use_display_range)
        
    def set_display_range(self, display_min, display_max):
        """设置显示区间"""
        self.image_widget.set_display_range(display_min, display_max)
        
    def reset_view(self):
        """重置视图"""
        self.image_widget.reset_view()
        
    def clear(self):
        """清空显示"""
        self.image_widget.clear()

class TrainingThread(QThread):
    """训练线程，避免阻塞GUI"""
    progress_signal = pyqtSignal(int)  # 进度值（0-100）
    log_signal = pyqtSignal(str)  # 日志信息
    finished_signal = pyqtSignal(bool, str)  # 完成状态，消息
    training_loss_signal = pyqtSignal(int, float)  # batch_idx, loss
    valid_loss_signal = pyqtSignal(int, float)  # epoch, loss
    validation_images_signal = pyqtSignal(object, object, object)  # input, output, gt images
    model_path_signal = pyqtSignal(str)  # 模型路径
    memory_cleared_signal = pyqtSignal()  # 内存已清理信号
    best_loss_curves_signal = pyqtSignal(list, list, list, list)
    model_reset_signal = pyqtSignal()  # 模型重置信号
    
    def __init__(self, args, operation_type="both"):
        super().__init__()
        self.args = args
        self.operation_type = operation_type  # "dataset", "train", "both"
        self._is_running = True
        self._stop_requested = False
        self.fm_dataset = None
        self.tdv_dnn = None
        
    def run(self):
        try:
            # 动态导入模块，避免初始化时加载
            # from src_BF.BF_SIM_Dataset import BF_SIM_Dataset
            # from src_TDV.TDV_DNN_Train import TDV_DNN_Train
           

            if self.operation_type in ["dataset", "both"]:
                # 生成数据集
                self.progress_signal.emit(0)
                self.log_signal.emit("Start dataset generation ...")
                
                if self._stop_requested:
                    self.finished_signal.emit(False, "Stop training by the user")
                    return
                
                def update_dataset_progress(current_file, total_files, message=""):
                    """数据集生成进度回调"""
                    if total_files > 0:
                        progress = int((current_file / total_files) * 100)
                        self.progress_signal.emit(progress)
                        if message:
                            self.log_signal.emit(message)
                        QApplication.processEvents()
                
                self.fm_dataset = FM_Dataset(self.args, update_dataset_progress)
                
                if self._stop_requested:
                    self.fm_dataset.stop_generation()
                    self.log_signal.emit("Dataset generation stopped before execution")
                    self.finished_signal.emit(False, "Stop training by the user")
                    return
                    
                Dataset_Clean, Dataset_Noisy = self.fm_dataset.generation()
                
                if Dataset_Clean is None or Dataset_Noisy is None:
                    self.log_signal.emit("No dataset is generation !")  #   在这儿做一下修改
                    self.progress_signal.emit(0)  # 进度条置0
                    # 立即清理内存
                    self.cleanup_memory()
                    # 发送完成信号，但标记为失败
                    self.finished_signal.emit(False, "Dataset generation failed: no dataset generated")
                    return
                
                if self.fm_dataset and hasattr(self.fm_dataset, 'is_stopped') and self.fm_dataset.is_stopped():
                    self.log_signal.emit("Dataset generation stopped !")
                    self.log_signal.emit("Ready !")
                    self.finished_signal.emit(False, "Operation stopped by user")
                    return
                
                if self._stop_requested:
                    self.log_signal.emit("Dataset generation completed but stop requested")
                    self.finished_signal.emit(False, "Operation stopped by user")
                    return
                
                self.progress_signal.emit(100)
                self.log_signal.emit("Finish dataset generation !")
            
            if self.operation_type == "both":
                self.progress_signal.emit(0)
                
            if self.operation_type in ["train", "both"]:
                self.log_signal.emit("Start TDV-DNN training ...")
                
                if self._stop_requested:
                    self.finished_signal.emit(False, "Operation stopped by user")
                    return
                
                def update_training_progress(current_epoch, total_epochs, message=""):
                    """训练进度回调"""
                    if total_epochs > 0:
                        progress = int((current_epoch / total_epochs) * 100)
                        self.progress_signal.emit(progress)
                        if message:
                            self.log_signal.emit(message)
                        QApplication.processEvents()
                
                def log_callback(message):
                    """日志回调函数"""
                    self.log_signal.emit(message)
                    QApplication.processEvents()
                    
                def training_loss_callback(batch_idx, loss):
                    """训练损失回调函数"""
                    self.training_loss_signal.emit(batch_idx, loss)
                    
                def valid_loss_callback(epoch, loss):
                    """验证损失回调函数"""
                    self.valid_loss_signal.emit(epoch, loss)
                    
                def validation_images_callback(input_images, output_images, gt_images):
                    """验证图像回调函数"""
                    self.validation_images_signal.emit(input_images, output_images, gt_images)
                    
                def model_path_callback(path):
                    """模型路径回调函数"""
                    self.model_path_signal.emit(path)
                
                def best_loss_curves_callback(training_losses, batch_indices, valid_losses, epoch_indices):
                    """最佳损失曲线回调函数"""
                    self.best_loss_curves_signal.emit(training_losses, batch_indices, valid_losses, epoch_indices)
                
                def reset_loss_curves_callback():
                    """重置损失曲线回调函数"""
                    self.model_reset_signal.emit()
                    # self.log_signal.emit("Model reinitialized, resetting loss curves")
                
                self.tdv_dnn = TDV_FM_DNN_Train(self.args, update_training_progress, log_callback)
                
                # if hasattr(self.tdv_dnn, 'set_callbacks'):
                #     self.tdv_dnn.set_callbacks(
                #         training_loss_callback=training_loss_callback,
                #         valid_loss_callback=valid_loss_callback,
                #         validation_images_callback=validation_images_callback,
                #         model_path_callback=model_path_callback
                #     )

                if hasattr(self.tdv_dnn, 'set_callbacks'):
                    self.tdv_dnn.set_callbacks(
                        training_loss_callback=training_loss_callback,
                        valid_loss_callback=valid_loss_callback,
                        validation_images_callback=validation_images_callback,
                        model_path_callback=model_path_callback,
                        best_loss_curves_callback=best_loss_curves_callback  # 新增
                    )
                
                # 在初始化 TDV_DNN 时设置回调
                if hasattr(self.tdv_dnn, 'set_reset_callback'):
                    self.tdv_dnn.set_reset_callback(reset_loss_curves_callback)

                if self._stop_requested:
                    self.tdv_dnn.stop_training()
                    self.finished_signal.emit(False, "Operation stopped by user")
                    return
                    
                self.tdv_dnn.train()
                
                if self.tdv_dnn and hasattr(self.tdv_dnn, 'is_stopped') and self.tdv_dnn.is_stopped():
                    self.log_signal.emit("TDV-DNN training stopped !")                    
                    # self.run_btn.setEnabled(True)
                    self.log_signal.emit("Ready !")
                    self.finished_signal.emit(False, "Operation stopped by user")
                    return
                
                if self._stop_requested:
                    self.log_signal.emit("TDV-DNN training completed but stop requested")
                    self.finished_signal.emit(False, "Operation stopped by user")
                    return
                
                self.progress_signal.emit(100)
                self.log_signal.emit("Finish TDV-DNN training !")
            
            if not self._stop_requested:
                # pass
                self.finished_signal.emit(True, "Finish Hybrid-"+Microscopy_Type+" training !")
                
        except Exception as e:
            if not self._stop_requested:
                self.log_signal.emit(f"Error: {str(e)}")
                self.finished_signal.emit(False, f"Operation failed: {str(e)}")
            else:
                self.log_signal.emit(f"Operation stopped by user: {str(e)}")
                self.finished_signal.emit(False, "Operation stopped by user")
                
        finally:
            # 无论成功还是失败，都执行清理
            self.cleanup_memory()
    
    def cleanup_memory(self):
        """清理内存和显存"""
        try:
            # 1. 清理训练对象
            if self.tdv_dnn is not None:
                try:
                    # 强制删除训练对象
                    if hasattr(self.tdv_dnn, 'cleanup'):
                        self.tdv_dnn.cleanup()
                    
                    # 清空模型和优化器
                    if hasattr(self.tdv_dnn, 'model'):
                        del self.tdv_dnn.model
                    if hasattr(self.tdv_dnn, 'optimizer'):
                        del self.tdv_dnn.optimizer
                    if hasattr(self.tdv_dnn, 'criterion'):
                        del self.tdv_dnn.criterion
                        
                    del self.tdv_dnn
                    self.tdv_dnn = None
                except Exception as e:
                    self.log_signal.emit(f"Warning: Failed to clean TDV-DNN: {str(e)}")
            
            # 2. 清理数据集对象
            if self.fm_dataset is not None:
                try:
                    if hasattr(self.fm_dataset, 'cleanup'):
                        self.fm_dataset.cleanup()
                    del self.fm_dataset
                    self.fm_dataset = None
                except Exception as e:
                    self.log_signal.emit(f"Warning: Failed to clean {Microscopy_Type} Dataset: {str(e)}")
            
            # 3. 清理PyTorch GPU缓存
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # self.log_signal.emit("GPU memory cleared successfully")
            except Exception as e:
                self.log_signal.emit(f"Warning: Failed to clear GPU cache: {str(e)}")
            
            # 4. 清理Python内存
            gc.collect()
            
            # 5. 发送内存清理完成信号
            self.memory_cleared_signal.emit()
            
            # self.log_signal.emit("Memory cleanup completed")
            
        except Exception as e:
            self.log_signal.emit(f"Error during memory cleanup: {str(e)}")
    
    def stop(self):
        """停止线程"""
        self._stop_requested = True
        self._is_running = False
        
        if self.tdv_dnn:
            try:
                self.fm_dataset = None
                self.tdv_dnn.stop_training()
                self.log_signal.emit(f"Stop TDV-DNN training ...")
            except:
                pass
                
        if self.fm_dataset:
            try:
                self.fm_dataset.stop_generation()
                self.log_signal.emit("Stop dataset generation ...")
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
        layout.setSpacing(5)
        
        # 修改：根据参数名显示不同的标签文本
        if param_name == 'FM_raw_data_folder_path':
            label_text = "Raw Data Folder Path"
        elif param_name == 'FM_emission_NA':
            label_text = "Emission NA"
        elif param_name == 'FM_emission_wavelength':
            label_text = "Emission Wavelength"
        elif param_name == 'FM_Raw_pixel_size':
            label_text = "Raw Pixel Size"
        else:
            label_text = param_name.replace('_', ' ').title()
            label_text = label_text.replace('Na', 'NA')
            label_text = label_text.replace('Bf', 'BF')
            label_text = label_text.replace('Psf', 'PSF')
            label_text = label_text.replace('Otf', 'OTF')
            label_text = label_text.replace('Tdv', 'TDV')
            label_text = label_text.replace('Sim', Microscopy_Type)
            label_text = label_text.replace('Dnn', 'DNN')
            label_text = label_text.replace('Xy', 'XY')
            label_text = label_text.replace('Train ', 'Training ')
        
        if unit:
            label_text += f" ({unit})"
            
        self.label = QLabel(label_text)
        self.label.setToolTip(tooltip)
        self.label.setFixedWidth(200)
        self.label.setFont(QFont("Arial", 9))
        self.label.setStyleSheet("color: #000000;")
        
        if param_type == "file":
            self.widget = QLineEdit(str(default_value))
            self.widget.setFont(QFont("Arial", 9))
            self.widget.setStyleSheet("color: #000000;")
            browse_btn = QPushButton("Browse")
            browse_btn.setFont(QFont("Arial", 9))
            browse_btn.setFixedWidth(70)
            browse_btn.clicked.connect(self.browse_file)
            layout.addWidget(self.label)
            layout.addWidget(self.widget)
            layout.addWidget(browse_btn)
        elif param_type == "float":
            self.widget = QDoubleSpinBox()
            self.widget.setRange(0.01, 10000)
            
            if param_name in ['SIM_excitation_wavelength', 'SIM_emission_wavelength', 'SIM_Raw_pixel_size', 
                              'FM_emission_wavelength', 'FM_Raw_pixel_size']:
                nm_value = default_value * 1e9
                self.widget.setValue(float(nm_value))
                self.widget.setDecimals(2)
            elif param_name == 'Train_learning_rate':
                display_value = default_value * 10000
                self.widget.setValue(float(display_value))
                self.widget.setDecimals(2)
            else:
                self.widget.setValue(float(default_value))
                self.widget.setDecimals(2)
            
            self.widget.setFont(QFont("Arial", 9))
            self.widget.setStyleSheet("color: #000000;")
            layout.addWidget(self.label)
            layout.addWidget(self.widget)
        elif param_type == "int":
            self.widget = QSpinBox()
            self.widget.setRange(1, 10000)
            self.widget.setValue(int(default_value))
            self.widget.setFont(QFont("Arial", 9))
            self.widget.setStyleSheet("color: #000000;")
            layout.addWidget(self.label)
            layout.addWidget(self.widget)
        elif param_type == "str":
            self.widget = QLineEdit(str(default_value))
            self.widget.setFont(QFont("Arial", 9))
            self.widget.setStyleSheet("color: #000000;")
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
        root_dir = get_root_directory()
        
        if self.param_name == 'SIM_BF_PSF_path':
            default_dir = os.path.join(root_dir, 'src_Hybrid/src_Optics/BF_PSF')
            file_filter = "TIFF Files (*.tif *.tiff);;All Files (*)"
        elif self.param_name == 'SIM_Recon_OTF_path':
            default_dir = os.path.join(root_dir, 'src_Hybrid/src_Optics/SIM_Recon_OTF')
            file_filter = "TIFF Files (*.tif *.tiff);;All Files (*)"
        # elif self.param_name in ['SIM_raw_data_folder_path', 'FM_raw_data_folder_path']:
        #     default_dir = root_dir
        #     file_filter = "All Files (*)"
        else:
            current_path = self.widget.text()
            if current_path and os.path.exists(current_path):
                default_dir = os.path.dirname(current_path)
            else:
                default_dir = root_dir
            file_filter = "All Files (*)"
        
        if not os.path.exists(default_dir):
            default_dir = root_dir
        
        if self.param_name in ['SIM_raw_data_folder_path', 'FM_raw_data_folder_path']:
            folder_path = QFileDialog.getExistingDirectory(
                self, "Select Folder", default_dir
            )
            if folder_path:
                try:
                    folder_path_norm = os.path.normpath(folder_path)
                    root_dir_norm = os.path.normpath(root_dir)
                    
                    folder_path_lower = folder_path_norm.lower()
                    root_dir_lower = root_dir_norm.lower()
                    
                    if folder_path_lower.startswith(root_dir_lower):
                        rel_path = os.path.relpath(folder_path_norm, root_dir_norm)
                        rel_path = rel_path.replace('\\', '/')
                        if not rel_path.startswith('./') and not rel_path.startswith('../'):
                            rel_path = './' + rel_path
                        self.widget.setText(rel_path)
                    else:
                        self.widget.setText(folder_path)
                except Exception as e:
                    self.widget.setText(folder_path)
        else:
            filename, _ = QFileDialog.getOpenFileName(
                self, "Select File", default_dir, file_filter
            )
            
            if filename:
                try:
                    filename_norm = os.path.normpath(filename)
                    root_dir_norm = os.path.normpath(root_dir)
                    
                    filename_lower = filename_norm.lower()
                    root_dir_lower = root_dir_norm.lower()
                    
                    if filename_lower.startswith(root_dir_lower):
                        rel_path = os.path.relpath(filename_norm, root_dir_norm)
                        rel_path = rel_path.replace('\\', '/')
                        if not rel_path.startswith('./') and not rel_path.startswith('../'):
                            rel_path = './' + rel_path
                        self.widget.setText(rel_path)
                    else:
                        self.widget.setText(filename)
                except Exception as e:
                    self.widget.setText(filename)
    
    def reset_value(self):
        if self.param_type == "file" or self.param_type == "str":
            self.widget.setText(str(self.default_value))
        elif self.param_type == "float":
            if self.param_name in ['SIM_excitation_wavelength', 'SIM_emission_wavelength', 
                                   'SIM_Raw_pixel_size', 'FM_emission_wavelength', 'FM_Raw_pixel_size']:
                nm_value = self.default_value * 1e9
                self.widget.setValue(float(nm_value))
            elif self.param_name == 'Train_learning_rate':
                display_value = self.default_value * 10000
                self.widget.setValue(float(display_value))
            # elif self.param_name == 'FM_emission_NA':
            #     display_value = self.default_value * 2  # 将默认值乘以2显示
            #     self.widget.setValue(float(display_value))
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
            if self.param_name in ['SIM_excitation_wavelength', 'SIM_emission_wavelength', 
                                   'SIM_Raw_pixel_size', 'FM_emission_wavelength', 'FM_Raw_pixel_size']:
                return value * 1e-9
            elif self.param_name == 'Train_learning_rate':
                return value / 10000
            # 新增：对 FM_emission_NA 的特殊处理 - 除以2后返回
            # elif self.param_name == 'FM_emission_NA':
            #     return value / 2
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
            if self.param_name in ['SIM_excitation_wavelength', 'SIM_emission_wavelength', 
                                   'SIM_Raw_pixel_size', 'FM_emission_wavelength', 'FM_Raw_pixel_size']:
                if isinstance(value, float) and value < 1e-6:
                    nm_value = value * 1e9
                else:
                    nm_value = float(value)
                self.widget.setValue(float(nm_value))
            elif self.param_name == 'Train_learning_rate':
                display_value = float(value) * 10000
                self.widget.setValue(float(display_value))
            # 新增：对 FM_emission_NA 的特殊处理 - 乘以2后显示
            # elif self.param_name == 'FM_emission_NA':
            #     display_value = float(value) * 2
            #     self.widget.setValue(float(display_value))
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
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i)
            widget = item.widget()
            if widget and isinstance(widget, QPushButton):
                widget.setEnabled(enabled)

class ModuleWidget(QGroupBox):
    """单个模块的控件组"""
    def __init__(self, title, parameters, parent=None):
        # 修改：替换标题中的SIM为MPM
        title = title.replace('Bf', 'BF')
        title = title.replace('Tdv', 'TDV')
        title = title.replace('Dnn', 'DNN')
        title = title.replace('Sim', Microscopy_Type)
        
        super().__init__(title, parent)
        self.parameters = {}
        
        font = QFont("Arial", 10, QFont.Bold)
        self.setFont(font)
        self.setStyleSheet("""
            QGroupBox {
                color: #000000;
            }
            QGroupBox::title {
                color: #000000;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(7)
        layout.setContentsMargins(10, 15, 10, 10)
        
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

class TDVTrainingGUI(QMainWindow):
    """TDV-MPM训练用户界面"""
    def __init__(self):
        super().__init__()
        self.args = default_config_TDV_FM_training()
        self.training_thread = None
        
        self.stop_check_timer = None
        self.stop_timeout_timer = None
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle(f"Hybrid-{Microscopy_Type} Training")
        self.setGeometry(100, 100, 1200, 900)
        self.setMinimumSize(1200, 900)
        self.setMaximumSize(1200, 900)
        
        self.setFont(QFont("Arial", 9))
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 左侧参数面板
        left_panel = QWidget()
        left_panel.setFixedWidth(500)
        left_panel.setFixedHeight(870)
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        title_label = QLabel(f"Hybrid-{Microscopy_Type} Training Parameters")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setStyleSheet("padding: 10px; color: #000000;")
        title_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title_label)
        
        self.tab_widget = QTabWidget()
        self.tab_widget.setFont(QFont("Arial", 9))
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ccc;
                background-color: white;
            }
            QTabBar::tab {
                font-family: Arial;
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
                color: #000000;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #2196F3;
                color: #000000;
            }
        """)
        
        # 修改：移除BF-SIM Parameters模块，直接在Dataset Generation Parameters模块中添加新参数
        
        # 数据集参数模块
        dataset_params = {
            'FM_raw_data_folder_path': {
                'type': 'file',
                'default': f'./Demo_training_{Microscopy_Type}',
                'tooltip': f'Path to {Microscopy_Type} raw data folder'
            },
            'FM_emission_NA': {
                'type': 'float',
                'default': 1.5,
                'tooltip': f'{Microscopy_Type} emission numerical aperture',
                'unit': ''
            },
            'FM_emission_wavelength': {
                'type': 'float',
                'default': 525e-9,
                'tooltip': f'{Microscopy_Type} emission wavelength',
                'unit': 'nm'
            },
            'FM_Raw_pixel_size': {
                'type': 'float',
                'default': 65e-9,
                'tooltip': f'{Microscopy_Type} raw data pixel size',
                'unit': 'nm'
            },
            'Dataset_Z_average_number': {
                'type': 'int',
                'default': 20,
                'tooltip': 'Number of frames to average in Z direction',
                'unit': 'frame'
            },
            'Dataset_Z_skip_number': {
                'type': 'int',
                'default': 16,
                'tooltip': 'Number of frames to skip in Z direction',
                'unit': 'frame'
            },
            'Dataset_XY_block_size': {
                'type': 'int',
                'default': 256,
                'tooltip': 'XY block size for dataset generation',
                'unit': 'pixel'
            },
            'Dataset_XY_block_interval': {
                'type': 'int',
                'default': 30,
                'tooltip': 'XY block interval for dataset generation',
                'unit': 'pixel'
            },
            'Dataset_XY_Poisson_noise_level': {
                'type': 'float',
                'default': 200,
                'tooltip': 'Poisson noise level',
                'unit': ''
            },
            'Dataset_XY_Gaussian_noise_level': {
                'type': 'float',
                'default': 200,
                'tooltip': 'Gaussian_noise_level',
                'unit': ''
            },
            'Dataset_XY_minimal_heterogeneity': {
                'type': 'float',
                'default': 0.01,
                'tooltip': 'Minimum heterogeneity threshold for XY blocks'
            }
        }
        self.dataset_widget = ModuleWidget("Dataset Generation Parameters", dataset_params)
        self.tab_widget.addTab(self.dataset_widget, "Dataset Generation")
        
        # TDV-DNN 训练参数模块
        train_params = {
            'Train_system': {
                'type': 'str',
                'default': 'CSDM',
                'tooltip': 'System name for model identification'
            },            
            'Train_sample': {
                'type': 'str',
                'default': 'ER',
                'tooltip': 'Sample name for model identification'
            },
            'Train_batch_size': {
                'type': 'int',
                'default': 10,
                'tooltip': 'Batch size for training'
            },
            'Train_epoch_number': {
                'type': 'int',
                'default': 100,
                'tooltip': 'Number of training epochs'
            },
            'Train_learning_rate': {
                'type': 'float',
                'default': 5e-4,
                'tooltip': 'Learning rate for training'
            }
        }
        self.train_widget = ModuleWidget("TDV-DNN Training Parameters", train_params)
        self.tab_widget.addTab(self.train_widget, "TDV-DNN Training")
        
        left_layout.addWidget(self.tab_widget)
        
        # 设备选择
        device_layout = QHBoxLayout()
        device_layout.setSpacing(8)
        device_layout.setContentsMargins(0, 0, 0, 0)
        
        device_label = QLabel("Device:")
        device_label.setFont(QFont("Arial", 9))
        device_label.setStyleSheet("color: #000000;")
        device_label.setFixedWidth(80)
        device_layout.addWidget(device_label)
        
        self.device_combo = QComboBox()
        self.device_combo.setFont(QFont("Arial", 9))
        self.device_combo.setFixedWidth(280)
        self.device_combo.setStyleSheet("color: #000000;")
        
        available_devices = self.detect_available_devices()
        for device in available_devices:
            self.device_combo.addItem(device)
        
        if 'cuda:0' in available_devices:
            self.device_combo.setCurrentText('cuda:0')
        elif available_devices:
            self.device_combo.setCurrentText(available_devices[0])
        
        self.device_info_label = QLabel(f"Detected: {len(available_devices)} device(s)")
        self.device_info_label.setFont(QFont("Arial", 8))
        self.device_info_label.setStyleSheet("color: #000000;")
        self.device_info_label.setFixedWidth(110)
        device_layout.addWidget(self.device_combo)
        device_layout.addWidget(self.device_info_label)
        
        left_layout.addLayout(device_layout)
        
        # 操作类型选择
        operation_layout = QHBoxLayout()
        operation_layout.setSpacing(8)
        operation_layout.setContentsMargins(0, 0, 0, 0)
        
        operation_label = QLabel("Operation:")
        operation_label.setFont(QFont("Arial", 9))
        operation_label.setStyleSheet("color: #000000;")
        operation_label.setFixedWidth(80)
        operation_layout.addWidget(operation_label)
        
        self.operation_combo = QComboBox()
        self.operation_combo.setFont(QFont("Arial", 9))
        self.operation_combo.setFixedWidth(280)
        self.operation_combo.setStyleSheet("color: #000000;")
        self.operation_combo.addItem("Dataset Generation")
        self.operation_combo.addItem("TDV-DNN Training")
        self.operation_combo.addItem("Dataset Generation & TDV-DNN Training")
        self.operation_combo.setCurrentText("Dataset Generation & TDV-DNN Training")
        
        self.operation_placeholder_label = QLabel()
        self.operation_placeholder_label.setFixedWidth(110)
        self.operation_placeholder_label.setStyleSheet("background-color: transparent;")
        
        operation_layout.addWidget(self.operation_combo)
        operation_layout.addWidget(self.operation_placeholder_label)
        
        left_layout.addLayout(operation_layout)
        
        # 参数操作按钮
        param_buttons_layout = QHBoxLayout()
        param_buttons_layout.setSpacing(8)
        param_buttons_layout.setContentsMargins(0, 0, 0, 0)
        
        self.load_defaults_btn = QPushButton("Default Parameters")
        self.load_defaults_btn.setFont(QFont("Arial", 9))
        self.load_defaults_btn.setFixedHeight(35)
        self.load_defaults_btn.setFixedWidth(160)
        self.load_defaults_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 6px;
                color: #000000;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.load_defaults_btn.clicked.connect(self.load_default_parameters)
        
        self.load_previous_btn = QPushButton("Load Parameters")
        self.load_previous_btn.setFont(QFont("Arial", 9))
        self.load_previous_btn.setFixedHeight(35)
        self.load_previous_btn.setFixedWidth(160)
        self.load_previous_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 6px;
                color: #000000;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.load_previous_btn.clicked.connect(self.load_previous_parameters)
        
        self.export_params_btn = QPushButton("Export Parameters")
        self.export_params_btn.setFont(QFont("Arial", 9))
        self.export_params_btn.setFixedHeight(35)
        self.export_params_btn.setFixedWidth(160)
        self.export_params_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 6px;
                color: #000000;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.export_params_btn.clicked.connect(self.export_current_parameters)
        
        param_buttons_layout.addWidget(self.load_defaults_btn)
        param_buttons_layout.addWidget(self.load_previous_btn)
        param_buttons_layout.addWidget(self.export_params_btn)
        
        left_layout.addLayout(param_buttons_layout)
        
        # 运行按钮
        self.run_btn = QPushButton("Start Training")
        self.run_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.run_btn.setFixedHeight(45)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 10px;
                border-radius: 6px;
                border: 2px solid #1976D2;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
                border-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                border-color: #999999;
                color: white;
            }
            QPushButton:pressed {
                background-color: #0b7dda;
                padding-top: 11px;
                padding-bottom: 9px;
            }
        """)
        self.run_btn.clicked.connect(self.start_training)
        left_layout.addWidget(self.run_btn)
        
        # 停止按钮
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.stop_btn.setFixedHeight(45)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 10px;
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
                color: white;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_training)
        left_layout.addWidget(self.stop_btn)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFont(QFont("Arial", 9))
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 3px;
                text-align: center;
                background-color: #f0f0f0;
                min-height: 20px;
                color: #000000;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 2px;
                width: 1px;
            }
        """)
        left_layout.addWidget(self.progress_bar)
        
        left_panel.setLayout(left_layout)
        main_layout.addWidget(left_panel)
        
        # 右侧可视化面板
        self.visualization_widget = TrainingVisualizationWidget()
        main_layout.addWidget(self.visualization_widget, 1)
        
        central_widget.setLayout(main_layout)
        
        self.load_default_parameters()
        self.visualization_widget.add_log("Ready !")
        
        # 连接图像同步信号
        self.connect_image_sync_signals()

        self.visualization_widget.reset_progress_signal.connect(
            lambda: self.progress_bar.setValue(0))
    
    def connect_image_sync_signals(self):
        """连接图像同步信号"""
        # 获取三个图像显示控件
        input_display = self.visualization_widget.input_display.image_widget
        output_display = self.visualization_widget.output_display.image_widget
        gt_display = self.visualization_widget.gt_display.image_widget
        
        # 连接缩放信号
        input_display.zoom_changed.connect(lambda z, p: self.sync_images(input_display, z, p))
        output_display.zoom_changed.connect(lambda z, p: self.sync_images(output_display, z, p))
        gt_display.zoom_changed.connect(lambda z, p: self.sync_images(gt_display, z, p))
        
        # 连接平移信号
        input_display.pan_changed.connect(lambda p: self.sync_pan(input_display, p))
        output_display.pan_changed.connect(lambda p: self.sync_pan(output_display, p))
        gt_display.pan_changed.connect(lambda p: self.sync_pan(gt_display, p))
    
    def sync_images(self, source_display, zoom_factor, pan_offset):
        """同步所有图像的缩放和平移"""
        # 防止递归调用
        if self.visualization_widget.is_syncing_zoom:
            return
        
        self.visualization_widget.is_syncing_zoom = True
        
        # 获取所有图像显示控件
        input_display = self.visualization_widget.input_display.image_widget
        output_display = self.visualization_widget.output_display.image_widget
        gt_display = self.visualization_widget.gt_display.image_widget
        
        # 同步所有图像
        displays = [input_display, output_display, gt_display]
        for display in displays:
            if display != source_display and display.original_qimage is not None:
                display.set_zoom_and_pan(zoom_factor, pan_offset)
        
        self.visualization_widget.is_syncing_zoom = False
    
    def sync_pan(self, source_display, pan_offset):
        """同步所有图像的平移"""
        # 防止递归调用
        if self.visualization_widget.is_syncing_pan:
            return
        
        self.visualization_widget.is_syncing_pan = True
        
        # 获取所有图像显示控件
        input_display = self.visualization_widget.input_display.image_widget
        output_display = self.visualization_widget.output_display.image_widget
        gt_display = self.visualization_widget.gt_display.image_widget
        
        # 获取源显示的当前缩放因子
        zoom_factor = source_display.zoom_factor
        
        # 同步所有图像
        displays = [input_display, output_display, gt_display]
        for display in displays:
            if display != source_display and display.original_qimage is not None:
                display.set_zoom_and_pan(zoom_factor, pan_offset)
        
        self.visualization_widget.is_syncing_pan = False
    
    def detect_available_devices(self):
        """检测可用的计算设备"""
        available_devices = ['cpu']
        
        try:
            import torch
            if torch.cuda.is_available():
                cuda_count = torch.cuda.device_count()
                for i in range(cuda_count):
                    device_name = f"cuda:{i}"
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
    
    def load_default_parameters(self):
        """加载默认参数"""
        for widget in [self.dataset_widget, self.train_widget]:
            for param_widget in widget.parameters.values():
                param_widget.reset_value()
        
        available_devices = self.detect_available_devices()
        self.device_combo.clear()
        for device in available_devices:
            self.device_combo.addItem(device)
        
        for i in range(self.device_combo.count()):
            if 'cuda:0' in self.device_combo.itemText(i).split('(')[0].strip():
                self.device_combo.setCurrentIndex(i)
                break
            elif available_devices:
                self.device_combo.setCurrentText(available_devices[0])
        
        self.device_info_label.setText(f"Detected: {len(available_devices)} device(s)")
        self.progress_bar.setValue(0)
    
    def load_previous_parameters_(self):
        """加载之前保存的参数"""
        try:
            try:
                base_name = "TDV_Training_Parameters.json"
            except:
                base_name = ""

            filename, _ = QFileDialog.getOpenFileName(
                self, "Load Parameters", base_name, "JSON Files (*.json);;All Files (*)"
            )
            if filename:
                with open(filename, 'r') as f:
                    params = json.load(f)
                
                self.set_parameters_from_dict(params)
                
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Failed to load parameters: {str(e)}")
    
    def load_previous_parameters(self):
        """加载之前保存的参数"""
        # 停止视频播放
        # self.stop_video_playback()
        
        self.collect_parameters()
        try:
            # 构建默认文件名
            try:
                filedir, filename = os.path.split(self.args.FM_raw_data_folder_path)
                base_name = filedir +'/'
            except AttributeError:
                # 如果 fname 不存在，使用空字符串作为基础名称
                base_name = ""

            filename, _ = QFileDialog.getOpenFileName(
                self, "Load Parameters", base_name, "JSON Files (*.json);;Text Files (*.txt);;All Files (*)"
            )
            if filename:
                # self.status_label.setText(f"Loading parameters from {Path(filename).name}...")
                # QApplication.processEvents()  # 立即更新UI
                # print(filename)
                with open(filename, 'r') as f:
                    params = json.load(f)
                
                # 设置参数值
                self.set_parameters_from_dict(params)
                
                # self.status_label.setText(f"Parameters loaded from {Path(filename).name}")
                
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Failed to load parameters: {str(e)}")
            # self.status_label.setText(f"Error loading parameters: {str(e)}")

    def export_current_parameters_(self):
        """导出当前参数"""
        try:
            params = self.collect_all_parameters()            

            try:
                base_name = f"TDV_Training_Parameters_{QDateTime.currentDateTime().toString('yyyyMMdd_hhmmss')}.json"
            except:
                base_name = "TDV_Training_Parameters.json"

            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Parameters", base_name, 
                "JSON Files (*.json);;Text Files (*.txt);;All Files (*)"
            )
            if filename:
                with open(filename, 'w') as f:
                    json.dump(params, f, indent=4)
                
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export parameters: {str(e)}")
    
    def export_current_parameters(self):
        """导出当前参数"""
        
        self.collect_parameters()
        try:
            # 收集当前参数
            params = self.collect_all_parameters()            

            # 构建默认文件名
            try:
                filedir, filename = os.path.split(self.args.FM_raw_data_folder_path)
                fname, ext = os.path.splitext(filename)
                base_name = filedir +'/'+fname + "_Training_Parameters.json"
            except AttributeError:
                # 如果 fname 不存在，使用空字符串作为基础名称
                base_name = "Demo_training_MPM_Training_Parameters.json"

            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Parameters", base_name, 
                "JSON Files (*.json);;Text Files (*.txt);;All Files (*)"
            )
            if filename:
                # self.status_label.setText(f"Exporting parameters to {Path(filename).name}...")
                # QApplication.processEvents()  # 立即更新UI
                
                with open(filename, 'w') as f:
                    json.dump(params, f, indent=4)
                
                # self.status_label.setText(f"Parameters exported to {Path(filename).name}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export parameters: {str(e)}")
            # self.status_label.setText(f"Error exporting parameters: {str(e)}")

    def collect_all_parameters(self):
        """收集所有参数到字典"""
        params = {}
        
        device_text = self.device_combo.currentText()
        if '(' in device_text:
            params['device'] = device_text.split('(')[0].strip()
        else:
            params['device'] = device_text
        
        operation_text = self.operation_combo.currentText()
        if operation_text == "Dataset Generation":
            params['operation_type'] = "dataset"
        elif operation_text == "TDV-DNN Training":
            params['operation_type'] = "train"
        else:
            params['operation_type'] = "both"
        
        # 修改：只收集dataset和train参数，不再有bf_sim参数
        params['dataset'] = self.dataset_widget.get_values()
        params['train'] = self.train_widget.get_values()
        params['timestamp'] = QDateTime.currentDateTime().toString()
        
        return params
    
    def set_parameters_from_dict(self, params):
        """从字典设置参数"""
        try:
            if 'device' in params:
                device = params['device']
                for i in range(self.device_combo.count()):
                    if device in self.device_combo.itemText(i):
                        self.device_combo.setCurrentIndex(i)
                        break
            
            if 'operation_type' in params:
                operation_type = params['operation_type']
                if operation_type == "dataset":
                    self.operation_combo.setCurrentText("Dataset Generation")
                elif operation_type == "train":
                    self.operation_combo.setCurrentText("TDV-DNN Training")
                else:
                    self.operation_combo.setCurrentText("Dataset Generation & TDV-DNN Training")
            
            # 修改：只设置dataset和train参数
            if 'dataset' in params:
                for key, value in params['dataset'].items():
                    if key in self.dataset_widget.parameters:
                        self.dataset_widget.parameters[key].set_value(value)
            
            if 'train' in params:
                for key, value in params['train'].items():
                    if key in self.train_widget.parameters:
                        self.train_widget.parameters[key].set_value(value)
                        
        except Exception as e:
            raise ValueError(f"Error setting parameters: {str(e)}")
    
    def collect_parameters(self):
        """收集所有参数到args对象"""
        device_text = self.device_combo.currentText()
        if '(' in device_text:
            self.args.device = device_text.split('(')[0].strip()
        else:
            self.args.device = device_text
        
        # 修改：只收集dataset和train参数
        dataset_values = self.dataset_widget.get_values()
        for key, value in dataset_values.items():
            setattr(self.args, key, value)
        
        train_values = self.train_widget.get_values()
        for key, value in train_values.items():
            setattr(self.args, key, value)
    
    def start_training(self):
        """开始训练过程"""
        try:
            # 设置训练状态标志
            self.visualization_widget.training_in_progress = True
            self.visualization_widget.pending_reset = False
            # 设置训练状态
            self.visualization_widget.set_training_state(True, False)

            self.collect_parameters()
            self.visualization_widget.reset_all()
            self.progress_bar.setValue(0)
            self.disable_controls()
            self.stop_btn.setEnabled(True)
            self.run_btn.setText("Training ...")
            self.run_btn.setEnabled(False)
            
            self.visualization_widget.add_log(f"Start Hybrid-{Microscopy_Type} training ...")
            operation_text = self.operation_combo.currentText()
            
            if operation_text == "Dataset Generation":
                operation_type = "dataset"
            elif operation_text == "TDV-DNN Training":
                operation_type = "train"
            else:
                operation_type = "both"
            
            self.training_thread = TrainingThread(self.args, operation_type)
            
            self.training_thread.progress_signal.connect(self.update_progress)
            self.training_thread.log_signal.connect(self.add_log_message)
            self.training_thread.finished_signal.connect(self.handle_training_finished)
            self.training_thread.training_loss_signal.connect(self.update_training_loss)
            self.training_thread.valid_loss_signal.connect(self.update_valid_loss)
            self.training_thread.validation_images_signal.connect(self.update_validation_images)
            self.training_thread.model_path_signal.connect(self.update_model_path)
            self.training_thread.memory_cleared_signal.connect(self.on_memory_cleared)
            # 连接最佳损失曲线信号
            self.training_thread.best_loss_curves_signal.connect(\
                self.visualization_widget.set_best_loss_curves)
            # 连接重置信号
            self.training_thread.model_reset_signal.connect(self.handle_model_reset)
            self.training_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Training process failed to start: {str(e)}")
            self.visualization_widget.add_log(f"Error: {str(e)}")
            self.enable_controls()
            self.run_btn.setEnabled(True)
            self.run_btn.setText("Start Training")
            self.stop_btn.setEnabled(False)
            # 重置训练状态标志
            self.visualization_widget.training_in_progress = False
            self.visualization_widget.set_training_state(False, False)

    def handle_model_reset(self):
        """处理模型重置"""
        self.visualization_widget.set_training_state(True, False)
        self.visualization_widget.handle_model_reset()

    def stop_training(self):
        """停止训练过程"""
        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(self, 'Stop Training', 
                                        'Are you sure you want to stop the training process?',
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.stop_btn.setEnabled(False)
                self.stop_btn.setText("Stopping ...")
                
                # 设置重置标志，防止后续信号处理
                self.visualization_widget.pending_reset = True
                
                # 停止训练线程
                self.training_thread.stop()
                
                # # 立即重置可视化，显示干净的状态
                self.visualization_widget.handle_training_stopped()

                model_path_exists = os.path.exists(self.visualization_widget.model_path) \
                    if self.visualization_widget.model_path else False
                self.visualization_widget.model_path_btn.setEnabled(model_path_exists)
                
                # 添加日志
                # self.visualization_widget.add_log("Training stopped by user")
                
        else:
            self.enable_controls()
            self.run_btn.setEnabled(True)
            self.run_btn.setText("Start Training")
            self.stop_btn.setEnabled(False)
            self.stop_btn.setText("Stop Training")
            self.progress_bar.setValue(0)
    
    def check_stop_progress(self):
        pass
        # """检查停止进度"""
        # if self.training_thread and not self.training_thread.isRunning():
        #     if self.stop_check_timer:
        #         self.stop_check_timer.stop()
        #         self.stop_check_timer = None
            
        #     if self.stop_timeout_timer:
        #         self.stop_timeout_timer.stop()
        #         self.stop_timeout_timer = None
            
        #     self.manual_finish_stop()
    
    def handle_stop_timeout(self):
        pass
        # """处理停止超时"""
        # if self.training_thread and self.training_thread.isRunning():
        #     if self.stop_check_timer:
        #         self.stop_check_timer.stop()
        #         self.stop_check_timer = None
            
        #     if self.stop_timeout_timer:
        #         self.stop_timeout_timer.stop()
        #         self.stop_timeout_timer = None
            
        #     self.manual_finish_stop()
    
    def manual_finish_stop(self):
        pass
        # """手动完成停止过程"""
        # self.enable_controls()
        # self.run_btn.setEnabled(True)
        # self.run_btn.setText("Start Training")
        # self.stop_btn.setEnabled(False)
        # self.stop_btn.setText("Stop Training")
        # self.progress_bar.setValue(0)
    
    def update_progress(self, value):
        """更新进度"""
        self.progress_bar.setValue(value)
        QApplication.processEvents()
    
    def add_log_message(self, message):
        """添加日志消息"""
        self.visualization_widget.add_log(message)
        QApplication.processEvents()
    
    def update_training_loss(self, batch_idx, loss):
        """更新训练损失"""
        self.visualization_widget.add_training_loss(batch_idx, loss)
        
    def update_valid_loss(self, epoch, loss):
        """更新验证损失"""
        self.visualization_widget.add_valid_loss(epoch, loss)
        
    def update_validation_images(self, input_images, output_images, gt_images):
        """更新验证图像"""
        self.visualization_widget.set_validation_images(input_images, output_images, gt_images)
        
    def update_model_path(self, path):
        """更新模型路径"""
        self.visualization_widget.set_model_path(path)
        
    def on_memory_cleared(self):
        """内存清理完成"""
        pass
        # self.visualization_widget.add_log("Memory cleanup completed successfully")
    
    def handle_training_finished_(self, success, message):
        """处理训练完成"""
        if self.stop_check_timer:
            try:
                self.stop_check_timer.stop()
                self.stop_check_timer = None
            except:
                pass
        
        if self.stop_timeout_timer:
            try:
                self.stop_timeout_timer.stop()
                self.stop_timeout_timer = None
            except:
                pass
        
        # 等待一小段时间确保清理完成
        QTimer.singleShot(1000, self.finalize_training_completion)
        
        # if success:
        #     self.visualization_widget.add_log(message)
        # else:
        #     self.visualization_widget.add_log(f"Training failed: {message}")
        #     self.progress_bar.setValue(0)
    
    def handle_training_finished(self, success, message):
        """处理训练完成"""
        # 清理定时器
        if self.stop_check_timer:
            try:
                self.stop_check_timer.stop()
                self.stop_check_timer = None
            except:
                pass
        
        if self.stop_timeout_timer:
            try:
                self.stop_timeout_timer.stop()
                self.stop_timeout_timer = None
            except:
                pass
        
        # 只有训练正常完成时才更新日志
        if success and not self.visualization_widget.pending_reset:
            self.visualization_widget.add_log(message)
        elif not success and not self.visualization_widget.pending_reset:
            # self.visualization_widget.add_log(f"Training failed: {message}")
            self.progress_bar.setValue(0)
        
        # 重置训练状态标志
        self.visualization_widget.training_in_progress = False
        self.visualization_widget.pending_reset = False
        # 重置训练状态标志
        self.visualization_widget.set_training_state(False, False)
        # # 确保模型按钮状态正确
        # if self.visualization_widget.model_path:
        #     self.visualization_widget.model_path_btn.setEnabled(True)
        # 最终完成训练
        QTimer.singleShot(100, self.finalize_training_completion)

    def finalize_training_completion(self):
        """最终完成训练"""
        self.enable_controls()
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Start Training")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("Stop Training")
        model_path_exists = os.path.exists(self.visualization_widget.model_path) \
            if self.visualization_widget.model_path else False
        self.visualization_widget.model_path_btn.setEnabled(model_path_exists)
        # # 确保模型按钮状态正确
        # if hasattr(self.visualization_widget, 'model_path') and self.visualization_widget.model_path:
        #     self.visualization_widget.model_path_btn.setEnabled(True)
        # 清理训练线程引用
        if self.training_thread:
            try:
                self.training_thread.wait(1000)
                self.training_thread = None
            except:
                pass
        
        # 强制进行垃圾回收
        gc.collect()
        
        # 如果是CUDA设备，清理GPU缓存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # self.visualization_widget.add_log("GPU cache cleared")
        except:
            pass
    
    def disable_controls(self):
        """禁用所有控件"""
        self.device_combo.setEnabled(False)
        self.operation_combo.setEnabled(False)
        self.load_defaults_btn.setEnabled(False)
        self.load_previous_btn.setEnabled(False)
        self.export_params_btn.setEnabled(False)
        self.device_info_label.setEnabled(False)
        self.operation_placeholder_label.setEnabled(False)
        
        self.dataset_widget.set_enabled(False)
        self.train_widget.set_enabled(False)
        
        self.visualization_widget.reset_view_btn.setEnabled(False)
        self.visualization_widget.reset_range_btn.setEnabled(False)
        self.visualization_widget.model_path_btn.setEnabled(False)
        self.visualization_widget.min_slider.setEnabled(False)
        self.visualization_widget.max_slider.setEnabled(False)
        self.visualization_widget.frame_slider.setEnabled(False)
    
    def enable_controls(self):
        """启用所有控件"""
        self.device_combo.setEnabled(True)
        self.operation_combo.setEnabled(True)
        self.load_defaults_btn.setEnabled(True)
        self.load_previous_btn.setEnabled(True)
        self.export_params_btn.setEnabled(True)
        self.device_info_label.setEnabled(True)
        self.operation_placeholder_label.setEnabled(True)
        
        self.dataset_widget.set_enabled(True)
        self.train_widget.set_enabled(True)
        
        if hasattr(self.visualization_widget, 'model_path'):
            self.visualization_widget.model_path_btn.setEnabled(
                os.path.exists(self.visualization_widget.model_path) and 
                self.visualization_widget.current_epoch >= 1
            )
    
    def closeEvent(self, event):
        """关闭窗口事件"""
        # 清理定时器
        if self.stop_check_timer:
            try:
                self.stop_check_timer.stop()
            except:
                pass
        
        if self.stop_timeout_timer:
            try:
                self.stop_timeout_timer.stop()
            except:
                pass
        
        # 清理训练线程
        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(self, 'Training in Progress', 
                                        'Training process is still running. Do you want to stop it and exit?',
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.training_thread.stop()
                if not self.training_thread.wait(3000):
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
    
    def force_cleanup(self):
        """强制清理内存"""
        try:
            # 清理Python内存
            gc.collect()
            
            # 清理PyTorch GPU缓存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except:
                pass
            
            # 清理训练线程引用
            self.training_thread = None
            
            # 清理可视化数据
            self.visualization_widget.reset_all()
            
        except Exception as e:
            print(f"Force cleanup error: {e}")

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    font = QFont("Arial", 9)
    app.setFont(font)
    
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    app.setPalette(palette)
    
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
            border-bottom: 2px solid #2196F3;
        }
        QLabel {
            font-family: Arial;
            color: #000000;
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
        QTextEdit {
            font-family: Arial;
        }
        QSlider {
            font-family: Arial;
        }
    """)
    
    window = TDVTrainingGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()