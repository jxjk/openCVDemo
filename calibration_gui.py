# -*- coding: utf-8 -*-
"""
相机标定GUI界面
Camera Calibration GUI Module
"""

import cv2
import numpy as np
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from datetime import datetime
import json

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit,
    QFileDialog, QGroupBox, QFormLayout, QMessageBox,
    QProgressBar, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont

from calibration_manager import ChessboardCalibration, CalibrationManager
from camera_manager import CameraManager, USBCamera
from logger_config import get_logger
from exceptions import CameraException, CalibrationException


class CalibrationWorker(QThread):
    """标定工作线程"""
    
    progress_signal = pyqtSignal(int, str)  # 进度, 消息
    finished_signal = pyqtSignal(dict)  # 标定结果
    error_signal = pyqtSignal(str)  # 错误信息
    
    def __init__(self, images: List[np.ndarray], pattern_size: Tuple[int, int], 
                 square_size: float):
        super().__init__()
        self.images = images
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.logger = get_logger(self.__class__.__name__)
    
    def run(self):
        """执行标定"""
        try:
            self.progress_signal.emit(0, "开始标定...")
            
            calibrator = ChessboardCalibration(self.pattern_size, self.square_size)
            
            # 逐步处理图像
            total = len(self.images)
            for i, img in enumerate(self.images):
                self.progress_signal.emit(
                    int((i / total) * 50),
                    f"处理图像 {i+1}/{total}..."
                )
                self.msleep(10)
            
            self.progress_signal.emit(50, "执行标定计算...")
            result = calibrator.calibrate(self.images)
            
            self.progress_signal.emit(100, "标定完成！")
            self.finished_signal.emit(result)
        
        except CalibrationException as e:
            self.error_signal.emit(f"标定失败: {str(e)}")
        except Exception as e:
            self.logger.error(f"标定过程异常: {e}")
            self.error_signal.emit(f"标定过程异常: {str(e)}")


class CameraPreviewWidget(QWidget):
    """相机预览窗口"""
    
    image_captured = pyqtSignal(np.ndarray)  # 图像捕获信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_logger(self.__class__.__name__)
        
        self.camera: Optional[USBCamera] = None
        self.camera_type = 'usb'
        self.device_id = '0'
        self.is_previewing = False
        
        self.init_ui()
        
        # 定时器用于更新预览
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_preview)
    
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
        
        # 控制按钮
        control_layout = QHBoxLayout()
        
        self.connect_btn = QPushButton("连接相机")
        self.connect_btn.clicked.connect(self.connect_camera)
        control_layout.addWidget(self.connect_btn)
        
        self.preview_btn = QPushButton("开始预览")
        self.preview_btn.clicked.connect(self.toggle_preview)
        self.preview_btn.setEnabled(False)
        control_layout.addWidget(self.preview_btn)
        
        self.capture_btn = QPushButton("采集图像")
        self.capture_btn.clicked.connect(self.capture_image)
        self.capture_btn.setEnabled(False)
        control_layout.addWidget(self.capture_btn)
        
        layout.addLayout(control_layout)
        
        # 状态信息
        self.status_label = QLabel("状态: 未连接")
        self.status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
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
                self.status_label.setText("状态: 已断开")
                self.preview_label.setText("相机预览")
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
        self.logger.info("停止预览")
    
    def update_preview(self):
        """更新预览"""
        try:
            frame = CameraManager.get_preview_frame(self.camera_type, self.device_id)
            
            if frame is not None:
                # 转换为QImage
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                
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
    
    def capture_image(self):
        """采集图像"""
        try:
            if not self.camera:
                return
            
            frame = self.camera.capture()
            
            if frame is not None:
                self.image_captured.emit(frame)
                self.logger.info("图像已采集")
        
        except CameraException as e:
            QMessageBox.critical(self, "采集失败", f"无法采集图像: {str(e)}")
            self.logger.error(f"采集图像失败: {e}")
    
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


class CalibrationResultWidget(QWidget):
    """标定结果显示窗口"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.calibration_result = None
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        # 结果表格
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["参数", "值"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.verticalHeader().setVisible(False)
        layout.addWidget(self.result_table)
        
        # 重投影误差显示
        error_group = QGroupBox("重投影误差")
        error_layout = QVBoxLayout()
        
        self.error_label = QLabel("未标定")
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                padding: 20px;
            }
        """)
        error_layout.addWidget(self.error_label)
        
        self.error_status_label = QLabel("误差状态: 未知")
        error_layout.addWidget(self.error_status_label)
        
        error_group.setLayout(error_layout)
        layout.addWidget(error_group)
        
        # 相机矩阵显示
        matrix_group = QGroupBox("相机内参矩阵")
        matrix_layout = QVBoxLayout()
        self.matrix_text = QTextEdit()
        self.matrix_text.setReadOnly(True)
        self.matrix_text.setMaximumHeight(150)
        matrix_layout.addWidget(self.matrix_text)
        matrix_group.setLayout(matrix_layout)
        layout.addWidget(matrix_group)
        
        self.setLayout(layout)
    
    def update_result(self, result: dict):
        """更新标定结果"""
        self.calibration_result = result
        
        # 更新表格
        self.result_table.setRowCount(0)
        
        items = [
            ("标定时间", result.get('timestamp', 'N/A')),
            ("成功图像数", f"{result.get('successful_images', 0)}/{result.get('total_images', 0)}"),
            ("重投影误差", f"{result.get('reprojection_error', 0):.4f} 像素"),
        ]
        
        self.result_table.setRowCount(len(items))
        for i, (key, value) in enumerate(items):
            self.result_table.setItem(i, 0, QTableWidgetItem(key))
            self.result_table.setItem(i, 1, QTableWidgetItem(value))
        
        # 更新重投影误差
        reprojection_error = result.get('reprojection_error', 0)
        self.error_label.setText(f"{reprojection_error:.4f} 像素")
        
        if reprojection_error < 0.5:
            self.error_label.setStyleSheet("""
                QLabel {
                    font-size: 24px;
                    font-weight: bold;
                    padding: 20px;
                    color: #4CAF50;
                }
            """)
            self.error_status_label.setText("误差状态: 优秀 (< 0.5像素)")
        elif reprojection_error < 1.0:
            self.error_label.setStyleSheet("""
                QLabel {
                    font-size: 24px;
                    font-weight: bold;
                    padding: 20px;
                    color: #FF9800;
                }
            """)
            self.error_status_label.setText("误差状态: 良好 (< 1.0像素)")
        else:
            self.error_label.setStyleSheet("""
                QLabel {
                    font-size: 24px;
                    font-weight: bold;
                    padding: 20px;
                    color: #F44336;
                }
            """)
            self.error_status_label.setText("误差状态: 需要重新标定 (≥ 1.0像素)")
        
        # 更新相机矩阵
        camera_matrix = result.get('camera_matrix', [])
        if camera_matrix:
            matrix_text = "相机内参矩阵:\n"
            matrix_array = np.array(camera_matrix)
            for row in matrix_array:
                matrix_text += f"[{ ' '.join([f'{v:8.3f}' for v in row]) }]\n"
            self.matrix_text.setText(matrix_text)
    
    def clear_result(self):
        """清除结果"""
        self.calibration_result = None
        self.result_table.setRowCount(0)
        self.error_label.setText("未标定")
        self.error_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                padding: 20px;
            }
        """)
        self.error_status_label.setText("误差状态: 未知")
        self.matrix_text.clear()


class CalibrationGUI(QMainWindow):
    """相机标定GUI主窗口"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(self.__class__.__name__)
        
        self.captured_images: List[np.ndarray] = []
        self.calibration_worker: Optional[CalibrationWorker] = None
        
        self.init_ui()
        self.init_calibration_manager()
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("相机标定工具 - 微小零件视觉检测系统")
        self.setMinimumSize(1200, 800)
        
        # 中央窗口
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout()
        
        # 左侧：参数设置
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # 中间：预览和采集
        center_panel = self.create_center_panel()
        main_layout.addWidget(center_panel, 2)
        
        # 右侧：结果显示
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)
        
        central_widget.setLayout(main_layout)
    
    def create_left_panel(self) -> QWidget:
        """创建左侧参数面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 标定参数
        param_group = QGroupBox("标定参数")
        param_layout = QFormLayout()
        
        self.pattern_width_spin = QSpinBox()
        self.pattern_width_spin.setRange(4, 20)
        self.pattern_width_spin.setValue(9)
        param_layout.addRow("棋盘格列数:", self.pattern_width_spin)
        
        self.pattern_height_spin = QSpinBox()
        self.pattern_height_spin.setRange(4, 20)
        self.pattern_height_spin.setValue(6)
        param_layout.addRow("棋盘格行数:", self.pattern_height_spin)
        
        self.square_size_spin = QDoubleSpinBox()
        self.square_size_spin.setRange(0.1, 100.0)
        self.square_size_spin.setValue(1.0)
        self.square_size_spin.setSingleStep(0.1)
        self.square_size_spin.setSuffix(" mm")
        param_layout.addRow("方块大小:", self.square_size_spin)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # 图像信息
        image_group = QGroupBox("图像信息")
        image_layout = QVBoxLayout()
        
        self.image_count_label = QLabel("已采集图像: 0")
        image_layout.addWidget(self.image_count_label)
        
        self.min_images_label = QLabel("最少需要: 10张")
        image_layout.addWidget(self.min_images_label)
        
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)
        
        # 操作按钮
        btn_group = QGroupBox("操作")
        btn_layout = QVBoxLayout()
        
        self.add_image_btn = QPushButton("添加图片文件")
        self.add_image_btn.clicked.connect(self.add_image_files)
        btn_layout.addWidget(self.add_image_btn)
        
        self.clear_images_btn = QPushButton("清除所有图像")
        self.clear_images_btn.clicked.connect(self.clear_images)
        btn_layout.addWidget(self.clear_images_btn)
        
        self.calibrate_btn = QPushButton("开始标定")
        self.calibrate_btn.clicked.connect(self.start_calibration)
        self.calibrate_btn.setEnabled(False)
        self.calibrate_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        btn_layout.addWidget(self.calibrate_btn)
        
        btn_group.setLayout(btn_layout)
        layout.addWidget(btn_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 日志
        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        panel.setLayout(layout)
        
        return panel
    
    def create_center_panel(self) -> QWidget:
        """创建中间预览面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 相机预览
        preview_group = QGroupBox("相机预览")
        preview_layout = QVBoxLayout()
        
        self.preview_widget = CameraPreviewWidget()
        self.preview_widget.image_captured.connect(self.on_image_captured)
        preview_layout.addWidget(self.preview_widget)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # 图像列表
        list_group = QGroupBox("已采集图像")
        list_layout = QVBoxLayout()
        
        self.image_list = QTextEdit()
        self.image_list.setReadOnly(True)
        self.image_list.setMaximumHeight(150)
        list_layout.addWidget(self.image_list)
        
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)
        
        panel.setLayout(layout)
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        """创建右侧结果面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 标定结果
        self.result_widget = CalibrationResultWidget()
        layout.addWidget(self.result_widget)
        
        # 保存/加载
        file_group = QGroupBox("标定文件")
        file_layout = QVBoxLayout()
        
        self.save_btn = QPushButton("保存标定结果")
        self.save_btn.clicked.connect(self.save_calibration)
        self.save_btn.setEnabled(False)
        file_layout.addWidget(self.save_btn)
        
        self.load_btn = QPushButton("加载标定结果")
        self.load_btn.clicked.connect(self.load_calibration)
        file_layout.addWidget(self.load_btn)
        
        self.export_btn = QPushButton("导出标定报告")
        self.export_btn.clicked.connect(self.export_report)
        self.export_btn.setEnabled(False)
        file_layout.addWidget(self.export_btn)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        layout.addStretch()
        panel.setLayout(layout)
        
        return panel
    
    def init_calibration_manager(self):
        """初始化标定管理器"""
        self.calibration_manager = CalibrationManager()
        self.calibration_data = None
    
    def add_image_files(self):
        """添加图片文件"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择标定图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        
        for file_path in files:
            try:
                img = cv2.imread(file_path)
                if img is not None:
                    self.captured_images.append(img)
                    self.update_image_list()
                    self.log_message(f"添加图片: {Path(file_path).name}")
            
            except Exception as e:
                self.log_message(f"加载图片失败 {file_path}: {str(e)}", "error")
    
    def on_image_captured(self, image: np.ndarray):
        """图像采集回调"""
        self.captured_images.append(image)
        self.update_image_list()
        self.log_message(f"采集图像 #{len(self.captured_images)}")
    
    def update_image_list(self):
        """更新图像列表显示"""
        count = len(self.captured_images)
        self.image_count_label.setText(f"已采集图像: {count}")
        
        # 更新按钮状态
        self.calibrate_btn.setEnabled(count >= 10)
        
        # 更新图像列表
        if count <= 10:
            image_list_text = "\n".join([f"图像 {i+1}" for i in range(count)])
        else:
            image_list_text = "\n".join([f"图像 {i+1}" for i in range(10)])
            image_list_text += f"\n... 还有 {count - 10} 张图像"
        
        self.image_list.setText(image_list_text)
    
    def clear_images(self):
        """清除所有图像"""
        reply = QMessageBox.question(
            self,
            "确认清除",
            "确定要清除所有已采集的图像吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.captured_images.clear()
            self.update_image_list()
            self.result_widget.clear_result()
            self.save_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            self.log_message("已清除所有图像")
    
    def start_calibration(self):
        """开始标定"""
        if len(self.captured_images) < 10:
            QMessageBox.warning(
                self,
                "图像不足",
                "至少需要10张标定图像才能进行标定！"
            )
            return
        
        pattern_size = (
            self.pattern_width_spin.value(),
            self.pattern_height_spin.value()
        )
        square_size = self.square_size_spin.value()
        
        self.log_message(f"开始标定... (图案大小: {pattern_size}, 方块大小: {square_size}mm)")
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.calibrate_btn.setEnabled(False)
        
        # 启动标定线程
        self.calibration_worker = CalibrationWorker(
            self.captured_images,
            pattern_size,
            square_size
        )
        self.calibration_worker.progress_signal.connect(self.on_calibration_progress)
        self.calibration_worker.finished_signal.connect(self.on_calibration_finished)
        self.calibration_worker.error_signal.connect(self.on_calibration_error)
        self.calibration_worker.start()
    
    def on_calibration_progress(self, progress: int, message: str):
        """标定进度更新"""
        self.progress_bar.setValue(progress)
        self.log_message(message)
    
    def on_calibration_finished(self, result: dict):
        """标定完成"""
        self.progress_bar.setVisible(False)
        self.calibrate_btn.setEnabled(True)
        
        self.calibration_data = result
        self.result_widget.update_result(result)
        
        self.save_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        
        self.log_message("标定完成！")
        self.log_message(f"重投影误差: {result['reprojection_error']:.4f} 像素")
        
        QMessageBox.information(
            self,
            "标定完成",
            f"标定成功完成！\n重投影误差: {result['reprojection_error']:.4f} 像素\n成功图像: {result['successful_images']}/{result['total_images']}"
        )
    
    def on_calibration_error(self, error: str):
        """标定错误"""
        self.progress_bar.setVisible(False)
        self.calibrate_btn.setEnabled(True)
        
        self.log_message(f"标定失败: {error}", "error")
        
        QMessageBox.critical(self, "标定失败", error)
    
    def save_calibration(self):
        """保存标定结果"""
        if not self.calibration_data:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存标定结果",
            f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON文件 (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.calibration_data, f, indent=2)
                
                self.log_message(f"标定结果已保存: {Path(file_path).name}")
                QMessageBox.information(self, "保存成功", "标定结果已保存！")
            
            except Exception as e:
                self.log_message(f"保存失败: {str(e)}", "error")
                QMessageBox.critical(self, "保存失败", f"无法保存标定结果: {str(e)}")
    
    def load_calibration(self):
        """加载标定结果"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "加载标定结果",
            "",
            "JSON文件 (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.calibration_data = json.load(f)
                
                self.result_widget.update_result(self.calibration_data)
                self.save_btn.setEnabled(True)
                self.export_btn.setEnabled(True)
                
                self.log_message(f"标定结果已加载: {Path(file_path).name}")
                QMessageBox.information(self, "加载成功", "标定结果已加载！")
            
            except Exception as e:
                self.log_message(f"加载失败: {str(e)}", "error")
                QMessageBox.critical(self, "加载失败", f"无法加载标定结果: {str(e)}")
    
    def export_report(self):
        """导出标定报告"""
        if not self.calibration_data:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出标定报告",
            f"calibration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "文本文件 (*.txt)"
        )
        
        if file_path:
            try:
                report = self.generate_report()
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                
                self.log_message(f"标定报告已导出: {Path(file_path).name}")
                QMessageBox.information(self, "导出成功", "标定报告已导出！")
            
            except Exception as e:
                self.log_message(f"导出失败: {str(e)}", "error")
                QMessageBox.critical(self, "导出失败", f"无法导出标定报告: {str(e)}")
    
    def generate_report(self) -> str:
        """生成标定报告"""
        report = "=" * 60 + "\n"
        report += "相机标定报告\n"
        report += "=" * 60 + "\n\n"
        
        report += f"标定时间: {self.calibration_data.get('timestamp', 'N/A')}\n\n"
        
        report += "标定参数:\n"
        report += f"  棋盘格列数: {self.pattern_width_spin.value()}\n"
        report += f"  棋盘格行数: {self.pattern_height_spin.value()}\n"
        report += f"  方块大小: {self.square_size_spin.value()} mm\n\n"
        
        report += "标定结果:\n"
        report += f"  成功图像数: {self.calibration_data.get('successful_images', 0)}\n"
        report += f"  总图像数: {self.calibration_data.get('total_images', 0)}\n"
        report += f"  重投影误差: {self.calibration_data.get('reprojection_error', 0):.4f} 像素\n\n"
        
        report += "相机内参矩阵:\n"
        camera_matrix = self.calibration_data.get('camera_matrix', [])
        if camera_matrix:
            matrix_array = np.array(camera_matrix)
            for row in matrix_array:
                report += f"  [{row[0]:8.3f} {row[1]:8.3f} {row[2]:8.3f}]\n"
        
        report += "\n畸变系数:\n"
        dist_coeffs = self.calibration_data.get('dist_coeffs', [])
        if dist_coeffs:
            report += f"  {dist_coeffs}\n"
        
        report += "\n" + "=" * 60 + "\n"
        report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "=" * 60 + "\n"
        
        return report
    
    def log_message(self, message: str, level: str = "info"):
        """记录日志消息"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if level == "error":
            color = "#F44336"
        elif level == "warning":
            color = "#FF9800"
        else:
            color = "#4CAF50"
        
        html = f'<span style="color: {color}">[{timestamp}] {message}</span>'
        self.log_text.append(html)
        
        self.logger.info(message)
    
    def closeEvent(self, event):
        """关闭事件"""
        if self.calibration_worker and self.calibration_worker.isRunning():
            self.calibration_worker.terminate()
            self.calibration_worker.wait()
        
        CameraManager.disconnect_all()
        super().closeEvent(event)


def main():
    """主函数"""
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    window = CalibrationGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()