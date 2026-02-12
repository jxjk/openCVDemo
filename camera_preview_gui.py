# -*- coding: utf-8 -*-
"""
增强实时图像预览功能
Enhanced Real-time Image Preview Module
"""

import cv2
import numpy as np
from typing import Optional, Dict, Tuple
from datetime import datetime

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QSpinBox, QDoubleSpinBox, QComboBox,
    QGroupBox, QFormLayout, QMessageBox, QCheckBox,
    QFileDialog, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

from camera_manager import CameraManager, USBCamera
from logger_config import get_logger
from exceptions import CameraException


class ImageQualityIndicator(QWidget):
    """图像质量指示器"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        # 标题
        title = QLabel("图像质量指标")
        title.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(title)
        
        # 指标表单
        form_layout = QFormLayout()
        
        self.brightness_label = QLabel("--")
        form_layout.addRow("亮度:", self.brightness_label)
        
        self.contrast_label = QLabel("--")
        form_layout.addRow("对比度:", self.contrast_label)
        
        self.blur_label = QLabel("--")
        form_layout.addRow("模糊度:", self.blur_label)
        
        self.noise_label = QLabel("--")
        form_layout.addRow("噪声:", self.noise_label)
        
        layout.addLayout(form_layout)
        self.setLayout(layout)
    
    def update_metrics(self, image: np.ndarray):
        """更新图像质量指标"""
        try:
            if image is None:
                return
            
            # 亮度（灰度均值）
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            self.brightness_label.setText(f"{brightness:.1f}")
            
            # 对比度（灰度标准差）
            contrast = np.std(gray)
            self.contrast_label.setText(f"{contrast:.1f}")
            
            # 模糊度（拉普拉斯方差）
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_score = np.var(laplacian)
            self.blur_label.setText(f"{blur_score:.1f}")
            
            # 噪声（高频能量）
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            noise_score = np.mean(np.sqrt(sobelx**2 + sobely**2))
            self.noise_label.setText(f"{noise_score:.1f}")
            
            # 颜色指示
            if brightness < 50:
                self.brightness_label.setStyleSheet("color: #FF9800;")
            elif brightness > 200:
                self.brightness_label.setStyleSheet("color: #F44336;")
            else:
                self.brightness_label.setStyleSheet("color: #4CAF50;")
            
            if blur_score < 100:
                self.blur_label.setStyleSheet("color: #F44336;")
            elif blur_score < 300:
                self.blur_label.setStyleSheet("color: #FF9800;")
            else:
                self.blur_label.setStyleSheet("color: #4CAF50;")
        
        except Exception as e:
            pass


class EnhancedPreviewWidget(QWidget):
    """增强预览窗口"""
    
    image_captured = pyqtSignal(np.ndarray)  # 图像捕获信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_logger(self.__class__.__name__)
        
        self.camera: Optional[USBCamera] = None
        self.camera_type = 'usb'
        self.device_id = '0'
        self.is_previewing = False
        
        # 预览参数
        self.show_crosshair = False
        self.show_grid = False
        self.show_info = True
        self.auto_exposure = True
        
        # 相机参数
        self.exposure_value = 50  # 0-100
        self.gain_value = 50  # 0-100
        self.brightness_value = 50  # 0-100
        
        self.init_ui()
        self.init_timer()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        # 预览标签
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("""
            QLabel {
                border: 2px solid #555;
                background-color: #333;
                color: #fff;
            }
        """)
        self.preview_label.setText("相机预览")
        layout.addWidget(self.preview_label)
        
        # 控制面板
        control_layout = QHBoxLayout()
        
        # 连接/断开按钮
        self.connect_btn = QPushButton("连接相机")
        self.connect_btn.clicked.connect(self.connect_camera)
        control_layout.addWidget(self.connect_btn)
        
        # 预览按钮
        self.preview_btn = QPushButton("开始预览")
        self.preview_btn.clicked.connect(self.toggle_preview)
        self.preview_btn.setEnabled(False)
        control_layout.addWidget(self.preview_btn)
        
        # 采集按钮
        self.capture_btn = QPushButton("采集图像")
        self.capture_btn.clicked.connect(self.capture_image)
        self.capture_btn.setEnabled(False)
        self.capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        control_layout.addWidget(self.capture_btn)
        
        # 保存按钮
        self.save_btn = QPushButton("保存当前帧")
        self.save_btn.clicked.connect(self.save_current_frame)
        self.save_btn.setEnabled(False)
        control_layout.addWidget(self.save_btn)
        
        layout.addLayout(control_layout)
        
        # 状态信息
        self.status_label = QLabel("状态: 未连接")
        self.status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.status_label)
        
        # 帧率信息
        self.fps_label = QLabel("帧率: -- fps")
        layout.addWidget(self.fps_label)
        
        self.setLayout(layout)
    
    def init_timer(self):
        """初始化定时器"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_preview)
        
        # 帧率计算
        self.frame_count = 0
        self.last_fps_update = datetime.now()
    
    def connect_camera(self):
        """连接相机"""
        try:
            if self.camera and self.camera.is_connected:
                # 断开连接
                self.stop_preview()
                self.camera.disconnect()
                self.camera = None
                self.connect_btn.setText("连接相机")
                self.preview_btn.setEnabled(False)
                self.capture_btn.setEnabled(False)
                self.save_btn.setEnabled(False)
                self.status_label.setText("状态: 已断开")
                self.preview_label.setText("相机预览")
                self.fps_label.setText("帧率: -- fps")
                self.logger.info("相机已断开")
                return
            
            # 连接相机
            success = CameraManager.connect(self.camera_type, self.device_id)
            
            if success:
                self.camera = CameraManager.get_camera(self.camera_type, self.device_id)
                if self.camera:
                    self.connect_btn.setText("断开连接")
                    self.preview_btn.setEnabled(True)
                    self.capture_btn.setEnabled(True)
                    self.save_btn.setEnabled(True)
                    
                    resolution = self.camera.get_resolution()
                    self.status_label.setText(f"状态: 已连接 ({resolution[0]}x{resolution[1]})")
                    self.logger.info(f"相机连接成功: {resolution[0]}x{resolution[1]}")
        
        except CameraException as e:
            QMessageBox.critical(self, "连接失败", f"无法连接相机: {str(e)}")
            self.logger.error(f"相机连接失败: {e}")
    
    def toggle_preview(self):
        """切换预览"""
        if self.is_previewing:
            self.stop_preview()
        else:
            self.start_preview()
    
    def start_preview(self):
        """开始预览"""
        try:
            if not self.camera:
                return
            
            success = CameraManager.start_preview(self.camera_type, self.device_id)
            
            if success:
                self.is_previewing = True
                self.preview_btn.setText("停止预览")
                self.timer.start(30)  # 30ms刷新一次
                self.frame_count = 0
                self.last_fps_update = datetime.now()
                self.logger.info("开始预览")
        
        except Exception as e:
            QMessageBox.critical(self, "预览失败", f"无法开始预览: {str(e)}")
            self.logger.error(f"开始预览失败: {e}")
    
    def stop_preview(self):
        """停止预览"""
        self.is_previewing = False
        self.timer.stop()
        CameraManager.stop_preview(self.camera_type, self.device_id)
        self.preview_btn.setText("开始预览")
        self.fps_label.setText("帧率: -- fps")
        self.logger.info("停止预览")
    
    def update_preview(self):
        """更新预览"""
        try:
            frame = CameraManager.get_preview_frame(self.camera_type, self.device_id)
            
            if frame is not None:
                # 计算帧率
                self.frame_count += 1
                now = datetime.now()
                elapsed = (now - self.last_fps_update).total_seconds()
                if elapsed >= 1.0:
                    fps = self.frame_count / elapsed
                    self.fps_label.setText(f"帧率: {fps:.1f} fps")
                    self.frame_count = 0
                    self.last_fps_update = now
                
                # 保存当前帧用于保存功能
                self.current_frame = frame.copy()
                
                # 添加辅助显示
                display_frame = self.add_overlay(frame)
                
                # 转换为QImage
                h, w, ch = display_frame.shape
                bytes_per_line = ch * w
                q_image = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                
                # 缩放以适应预览窗口
                pixmap = QPixmap.fromImage(q_image)
                scaled_pixmap = pixmap.scaled(
                    self.preview_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                
                self.preview_label.setPixmap(scaled_pixmap)
        
        except Exception as e:
            self.logger.error(f"更新预览失败: {e}")
    
    def add_overlay(self, frame: np.ndarray) -> np.ndarray:
        """添加覆盖层"""
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # 十字线
        if self.show_crosshair:
            center_x, center_y = w // 2, h // 2
            cv2.line(display_frame, (center_x, 0), (center_x, h), (0, 255, 255), 1)
            cv2.line(display_frame, (0, center_y), (w, center_y), (0, 255, 255), 1)
            cv2.circle(display_frame, (center_x, center_y), 5, (0, 255, 255), 2)
        
        # 网格
        if self.show_grid:
            grid_spacing = 100
            for x in range(grid_spacing, w, grid_spacing):
                cv2.line(display_frame, (x, 0), (x, h), (100, 100, 100), 1)
            for y in range(grid_spacing, h, grid_spacing):
                cv2.line(display_frame, (0, y), (w, y), (100, 100, 100), 1)
        
        # 信息
        if self.show_info:
            timestamp = datetime.now().strftime('%H:%M:%S')
            info_text = f"Time: {timestamp}"
            cv2.putText(display_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        return display_frame
    
    def capture_image(self):
        """采集图像"""
        try:
            if not self.camera:
                return
            
            frame = self.camera.capture()
            
            if frame is not None:
                self.current_frame = frame.copy()
                self.image_captured.emit(frame)
                self.logger.info("图像已采集")
        
        except CameraException as e:
            QMessageBox.critical(self, "采集失败", f"无法采集图像: {str(e)}")
            self.logger.error(f"采集图像失败: {e}")
    
    def save_current_frame(self):
        """保存当前帧"""
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            QMessageBox.warning(self, "无图像", "没有可保存的图像！")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存图像",
            f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            "JPEG文件 (*.jpg);;PNG文件 (*.png);;BMP文件 (*.bmp)"
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.current_frame)
                self.logger.info(f"图像已保存: {file_path}")
                QMessageBox.information(self, "保存成功", "图像已保存！")
            
            except Exception as e:
                self.logger.error(f"保存图像失败: {e}")
                QMessageBox.critical(self, "保存失败", f"无法保存图像: {str(e)}")
    
    def set_exposure(self, value: float):
        """设置曝光"""
        self.exposure_value = value
        if self.camera:
            self.camera.set_exposure(value * 100)
    
    def set_gain(self, value: float):
        """设置增益"""
        self.gain_value = value
        if self.camera:
            self.camera.set_gain(value)
    
    def set_camera(self, camera_type: str, device_id: str):
        """设置相机"""
        if self.is_previewing:
            self.stop_preview()
        
        if self.camera and self.camera.is_connected:
            CameraManager.disconnect(self.camera_type, self.device_id)
        
        self.camera_type = camera_type
        self.device_id = device_id
        self.camera = None
    
    def closeEvent(self, event):
        """关闭事件"""
        if self.is_previewing:
            self.stop_preview()
        
        if self.camera and self.camera.is_connected:
            CameraManager.disconnect(self.camera_type, self.device_id)
        
        super().closeEvent(event)


class CameraControlPanel(QWidget):
    """相机控制面板"""
    
    exposure_changed = pyqtSignal(float)
    gain_changed = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        # 曝光控制
        exposure_group = QGroupBox("曝光控制")
        exposure_layout = QVBoxLayout()
        
        exposure_layout.addWidget(QLabel("曝光时间"))
        
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(0, 100)
        self.exposure_slider.setValue(50)
        self.exposure_slider.valueChanged.connect(self.on_exposure_changed)
        exposure_layout.addWidget(self.exposure_slider)
        
        self.exposure_value_label = QLabel("50")
        exposure_layout.addWidget(self.exposure_value_label)
        
        self.auto_exposure_check = QCheckBox("自动曝光")
        self.auto_exposure_check.setChecked(True)
        self.auto_exposure_check.toggled.connect(self.on_auto_exposure_toggled)
        exposure_layout.addWidget(self.auto_exposure_check)
        
        exposure_group.setLayout(exposure_layout)
        layout.addWidget(exposure_group)
        
        # 增益控制
        gain_group = QGroupBox("增益控制")
        gain_layout = QVBoxLayout()
        
        gain_layout.addWidget(QLabel("增益"))
        
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(0, 100)
        self.gain_slider.setValue(50)
        self.gain_slider.valueChanged.connect(self.on_gain_changed)
        gain_layout.addWidget(self.gain_slider)
        
        self.gain_value_label = QLabel("50")
        gain_layout.addWidget(self.gain_value_label)
        
        gain_group.setLayout(gain_layout)
        layout.addWidget(gain_group)
        
        # 显示选项
        display_group = QGroupBox("显示选项")
        display_layout = QVBoxLayout()
        
        self.crosshair_check = QCheckBox("显示十字线")
        display_layout.addWidget(self.crosshair_check)
        
        self.grid_check = QCheckBox("显示网格")
        display_layout.addWidget(self.grid_check)
        
        self.info_check = QCheckBox("显示信息")
        self.info_check.setChecked(True)
        display_layout.addWidget(self.info_check)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def on_exposure_changed(self, value: int):
        """曝光值改变"""
        self.exposure_value_label.setText(str(value))
        if not self.auto_exposure_check.isChecked():
            self.exposure_changed.emit(value)
    
    def on_gain_changed(self, value: int):
        """增益值改变"""
        self.gain_value_label.setText(str(value))
        self.gain_changed.emit(value)
    
    def on_auto_exposure_toggled(self, checked: bool):
        """自动曝光切换"""
        self.exposure_slider.setEnabled(not checked)
        if not checked:
            self.exposure_changed.emit(self.exposure_slider.value())


class CameraPreviewGUI(QMainWindow):
    """相机预览GUI主窗口"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(self.__class__.__name__)
        
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("实时图像预览 - 微小零件视觉检测系统")
        self.setMinimumSize(1100, 700)
        
        # 中央窗口
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout()
        
        # 左侧：预览窗口
        left_layout = QVBoxLayout()
        
        preview_group = QGroupBox("实时预览")
        preview_layout = QVBoxLayout()
        
        self.preview_widget = EnhancedPreviewWidget()
        preview_layout.addWidget(self.preview_widget)
        
        preview_group.setLayout(preview_layout)
        left_layout.addWidget(preview_group, 3)
        
        # 图像质量指示器
        quality_group = QGroupBox("图像质量")
        quality_layout = QVBoxLayout()
        
        self.quality_indicator = ImageQualityIndicator()
        quality_layout.addWidget(self.quality_indicator)
        
        quality_group.setLayout(quality_layout)
        left_layout.addWidget(quality_group, 1)
        
        main_layout.addLayout(left_layout, 3)
        
        # 右侧：控制面板
        right_layout = QVBoxLayout()
        
        # 相机控制
        control_group = QGroupBox("相机控制")
        control_layout = QVBoxLayout()
        
        self.control_panel = CameraControlPanel()
        
        # 连接信号
        self.control_panel.exposure_changed.connect(self.preview_widget.set_exposure)
        self.control_panel.gain_changed.connect(self.preview_widget.set_gain)
        self.control_panel.crosshair_check.toggled.connect(
            lambda checked: setattr(self.preview_widget, 'show_crosshair', checked)
        )
        self.control_panel.grid_check.toggled.connect(
            lambda checked: setattr(self.preview_widget, 'show_grid', checked)
        )
        self.control_panel.info_check.toggled.connect(
            lambda checked: setattr(self.preview_widget, 'show_info', checked)
        )
        
        control_layout.addWidget(self.control_panel)
        control_group.setLayout(control_layout)
        right_layout.addWidget(control_group)
        
        # 相机选择
        camera_group = QGroupBox("相机选择")
        camera_layout = QFormLayout()
        
        self.camera_type_combo = QComboBox()
        self.camera_type_combo.addItems(['usb'])
        camera_layout.addRow("相机类型:", self.camera_type_combo)
        
        self.device_id_spin = QSpinBox()
        self.device_id_spin.setRange(0, 9)
        self.device_id_spin.setValue(0)
        camera_layout.addRow("设备ID:", self.device_id_spin)
        
        self.refresh_cameras_btn = QPushButton("刷新相机列表")
        self.refresh_cameras_btn.clicked.connect(self.refresh_cameras)
        camera_layout.addRow(self.refresh_cameras_btn)
        
        camera_group.setLayout(camera_layout)
        right_layout.addWidget(camera_group)
        
        right_layout.addStretch()
        main_layout.addLayout(right_layout, 1)
        
        central_widget.setLayout(main_layout)
        
        # 更新质量指标定时器
        self.quality_timer = QTimer()
        self.quality_timer.timeout.connect(self.update_quality_metrics)
    
    def update_quality_metrics(self):
        """更新图像质量指标"""
        if self.preview_widget.is_previewing and hasattr(self.preview_widget, 'current_frame'):
            self.quality_indicator.update_metrics(self.preview_widget.current_frame)
    
    def refresh_cameras(self):
        """刷新相机列表"""
        cameras = CameraManager.list_cameras()
        
        if cameras:
            message = "找到以下相机:\n\n"
            for cam in cameras:
                message += f"- {cam['name']} ({cam['device_id']}): {cam['resolution']}\n"
            QMessageBox.information(self, "相机列表", message)
        else:
            QMessageBox.warning(self, "无相机", "未找到可用相机！")
    
    def closeEvent(self, event):
        """关闭事件"""
        self.quality_timer.stop()
        CameraManager.disconnect_all()
        super().closeEvent(event)


def main():
    """主函数"""
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = CameraPreviewGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
