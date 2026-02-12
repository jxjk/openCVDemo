# -*- coding: utf-8 -*-
"""
像素-毫米标定GUI界面
Pixel-to-Millimeter Calibration GUI Module
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path
from datetime import datetime
import json

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QDoubleSpinBox, QTextEdit,
    QFileDialog, QGroupBox, QFormLayout, QMessageBox,
    QRadioButton, QButtonGroup, QComboBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont

from calibration_manager import PixelToMmCalibration, CalibrationManager
from camera_manager import CameraManager, USBCamera
from logger_config import get_logger
from exceptions import CalibrationException


class ImageLabel(QLabel):
    """可点击的图像标签"""
    
    point_selected = pyqtSignal(int, int)  # 点选中信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.points: List[Tuple[int, int]] = []
        self.max_points = 2
        self.point_radius = 5
        self.show_crosshair = False
        
        self.setMinimumSize(600, 400)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #555;
                background-color: #333;
            }
        """)
    
    def set_image(self, image: np.ndarray):
        """设置图像"""
        self.image = image
        self.points.clear()
        self.update_display()
    
    def update_display(self):
        """更新显示"""
        if self.image is None:
            self.setText("请加载图像")
            return
        
        # 转换为QImage
        h, w = self.image.shape[:2]
        if len(self.image.shape) == 3:
            bytes_per_line = 3 * w
            q_image = QImage(self.image.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        else:
            bytes_per_line = w
            q_image = QImage(self.image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        
        # 绘制选中点
        painter = QPainter(q_image)
        pen = QPen(QColor(0, 255, 0), 3)
        painter.setPen(pen)
        
        for i, point in enumerate(self.points):
            x, y = point
            painter.drawEllipse(QPoint(x, y), self.point_radius, self.point_radius)
            
            # 标注点编号
            font = QFont()
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(x + 10, y - 10, f"P{i+1}")
        
        painter.end()
        
        # 缩放以适应窗口
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.setPixmap(scaled_pixmap)
    
    def mousePressEvent(self, event):
        """鼠标点击事件"""
        if self.image is None or len(self.points) >= self.max_points:
            return
        
        # 计算点击位置在原图上的坐标
        pixmap = self.pixmap()
        if pixmap is None:
            return
        
        label_size = self.size()
        pixmap_size = pixmap.size()
        
        # 计算缩放比例
        scale_x = self.image.shape[1] / pixmap_size.width()
        scale_y = self.image.shape[0] / pixmap_size.height()
        
        # 考虑保持纵横比后的偏移
        offset_x = (label_size.width() - pixmap_size.width()) / 2
        offset_y = (label_size.height() - pixmap_size.height()) / 2
        
        click_x = (event.x() - offset_x) * scale_x
        click_y = (event.y() - offset_y) * scale_y
        
        # 边界检查
        click_x = max(0, min(click_x, self.image.shape[1] - 1))
        click_y = max(0, min(click_y, self.image.shape[0] - 1))
        
        # 添加点
        self.points.append((int(click_x), int(click_y)))
        self.point_selected.emit(int(click_x), int(click_y))
        self.update_display()
    
    def clear_points(self):
        """清除所有点"""
        self.points.clear()
        self.update_display()
    
    def resizeEvent(self, event):
        """调整大小事件"""
        super().resizeEvent(event)
        self.update_display()


class ReferenceCalibrationWidget(QWidget):
    """参考长度标定面板"""
    
    calibration_finished = pyqtSignal(float)  # 标定完成信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_logger(self.__class__.__name__)
        self.calibrator = PixelToMmCalibration()
        self.image = None
        
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        # 参数设置
        param_group = QGroupBox("标定参数")
        param_layout = QFormLayout()
        
        self.reference_length_spin = QDoubleSpinBox()
        self.reference_length_spin.setRange(0.1, 1000.0)
        self.reference_length_spin.setValue(10.0)
        self.reference_length_spin.setSingleStep(0.1)
        self.reference_length_spin.setSuffix(" mm")
        param_layout.addRow("参考长度:", self.reference_length_spin)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # 图像显示
        image_group = QGroupBox("标定图像")
        image_layout = QVBoxLayout()
        
        self.image_label = ImageLabel()
        self.image_label.point_selected.connect(self.on_point_selected)
        image_layout.addWidget(self.image_label)
        
        image_btn_layout = QHBoxLayout()
        self.load_image_btn = QPushButton("加载图像")
        self.load_image_btn.clicked.connect(self.load_image)
        image_btn_layout.addWidget(self.load_image_btn)
        
        self.clear_points_btn = QPushButton("清除选点")
        self.clear_points_btn.clicked.connect(self.clear_points)
        self.clear_points_btn.setEnabled(False)
        image_btn_layout.addWidget(self.clear_points_btn)
        
        image_layout.addLayout(image_btn_layout)
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)
        
        # 点信息
        point_group = QGroupBox("选点信息")
        point_layout = QVBoxLayout()
        
        self.point_info_label = QLabel("请在图像上选择两个参考点")
        point_layout.addWidget(self.point_info_label)
        
        self.pixel_distance_label = QLabel("像素距离: --")
        point_layout.addWidget(self.pixel_distance_label)
        
        point_group.setLayout(point_layout)
        layout.addWidget(point_group)
        
        # 标定按钮
        self.calibrate_btn = QPushButton("开始标定")
        self.calibrate_btn.clicked.connect(self.calibrate)
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
        layout.addWidget(self.calibrate_btn)
        
        # 结果
        result_group = QGroupBox("标定结果")
        result_layout = QFormLayout()
        
        self.pixel_to_mm_label = QLabel("--")
        self.pixel_to_mm_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        result_layout.addRow("转换比例:", self.pixel_to_mm_label)
        
        self.mm_to_pixel_label = QLabel("--")
        result_layout.addRow("逆转换:", self.mm_to_pixel_label)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def load_image(self):
        """加载图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择标定图像",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            try:
                self.image = cv2.imread(file_path)
                if self.image is not None:
                    self.image_label.set_image(self.image)
                    self.clear_points_btn.setEnabled(True)
                    self.logger.info(f"加载图像: {Path(file_path).name}")
                else:
                    QMessageBox.warning(self, "加载失败", "无法加载图像！")
            
            except Exception as e:
                QMessageBox.critical(self, "加载失败", f"加载图像失败: {str(e)}")
                self.logger.error(f"加载图像失败: {e}")
    
    def on_point_selected(self, x: int, y: int):
        """点选中事件"""
        points = self.image_label.points
        
        if len(points) == 1:
            self.point_info_label.setText(f"已选择 1/2 个点: ({x}, {y})")
        elif len(points) == 2:
            self.point_info_label.setText(f"已选择 2/2 个点")
            self.calculate_pixel_distance()
            self.calibrate_btn.setEnabled(True)
    
    def calculate_pixel_distance(self):
        """计算像素距离"""
        if len(self.image_label.points) != 2:
            return
        
        p1, p2 = self.image_label.points
        pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        self.pixel_distance_label.setText(f"像素距离: {pixel_distance:.2f} 像素")
    
    def clear_points(self):
        """清除选点"""
        self.image_label.clear_points()
        self.point_info_label.setText("请在图像上选择两个参考点")
        self.pixel_distance_label.setText("像素距离: --")
        self.calibrate_btn.setEnabled(False)
    
    def calibrate(self):
        """执行标定"""
        if len(self.image_label.points) != 2:
            QMessageBox.warning(self, "选点不足", "请选择两个参考点！")
            return
        
        try:
            reference_length = self.reference_length_spin.value()
            points = self.image_label.points
            
            pixel_to_mm = self.calibrator.calibrate_by_reference(
                self.image,
                reference_length,
                points
            )
            
            # 更新结果显示
            self.pixel_to_mm_label.setText(f"{pixel_to_mm:.6f} mm/像素")
            self.mm_to_pixel_label.setText(f"{1.0/pixel_to_mm:.2f} 像素/mm")
            
            self.calibration_finished.emit(pixel_to_mm)
            
            QMessageBox.information(
                self,
                "标定完成",
                f"标定成功！\n转换比例: {pixel_to_mm:.6f} mm/像素"
            )
            
            self.logger.info(f"标定完成: {pixel_to_mm:.6f} mm/像素")
        
        except CalibrationException as e:
            QMessageBox.critical(self, "标定失败", f"标定失败: {str(e)}")
            self.logger.error(f"标定失败: {e}")


class CircleCalibrationWidget(QWidget):
    """圆形标定面板"""
    
    calibration_finished = pyqtSignal(float)  # 标定完成信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_logger(self.__class__.__name__)
        self.calibrator = PixelToMmCalibration()
        self.image = None
        self.circle_center = None
        self.circle_radius = 0
        
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        # 参数设置
        param_group = QGroupBox("标定参数")
        param_layout = QFormLayout()
        
        self.circle_diameter_spin = QDoubleSpinBox()
        self.circle_diameter_spin.setRange(0.1, 1000.0)
        self.circle_diameter_spin.setValue(10.0)
        self.circle_diameter_spin.setSingleStep(0.1)
        self.circle_diameter_spin.setSuffix(" mm")
        param_layout.addRow("圆形直径:", self.circle_diameter_spin)
        
        self.min_radius_spin = QDoubleSpinBox()
        self.min_radius_spin.setRange(1.0, 500.0)
        self.min_radius_spin.setValue(10.0)
        param_layout.addRow("最小半径:", self.min_radius_spin)
        
        self.max_radius_spin = QDoubleSpinBox()
        self.max_radius_spin.setRange(1.0, 500.0)
        self.max_radius_spin.setValue(200.0)
        param_layout.addRow("最大半径:", self.max_radius_spin)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # 图像显示
        image_group = QGroupBox("标定图像")
        image_layout = QVBoxLayout()
        
        self.image_label = ImageLabel()
        self.image_label.max_points = 1  # 只需要选择一个点（圆心）
        self.image_label.point_selected.connect(self.on_point_selected)
        image_layout.addWidget(self.image_label)
        
        image_btn_layout = QHBoxLayout()
        self.load_image_btn = QPushButton("加载图像")
        self.load_image_btn.clicked.connect(self.load_image)
        image_btn_layout.addWidget(self.load_image_btn)
        
        self.detect_circle_btn = QPushButton("自动检测圆形")
        self.detect_circle_btn.clicked.connect(self.detect_circle)
        image_btn_layout.addWidget(self.detect_circle_btn)
        
        self.clear_points_btn = QPushButton("清除选点")
        self.clear_points_btn.clicked.connect(self.clear_points)
        self.clear_points_btn.setEnabled(False)
        image_btn_layout.addWidget(self.clear_points_btn)
        
        image_layout.addLayout(image_btn_layout)
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)
        
        # 圆形信息
        circle_group = QGroupBox("圆形信息")
        circle_layout = QFormLayout()
        
        self.center_label = QLabel("--")
        circle_layout.addRow("圆心坐标:", self.center_label)
        
        self.radius_label = QLabel("--")
        circle_layout.addRow("检测半径:", self.radius_label)
        
        circle_group.setLayout(circle_layout)
        layout.addWidget(circle_group)
        
        # 标定按钮
        self.calibrate_btn = QPushButton("开始标定")
        self.calibrate_btn.clicked.connect(self.calibrate)
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
        layout.addWidget(self.calibrate_btn)
        
        # 结果
        result_group = QGroupBox("标定结果")
        result_layout = QFormLayout()
        
        self.pixel_to_mm_label = QLabel("--")
        self.pixel_to_mm_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        result_layout.addRow("转换比例:", self.pixel_to_mm_label)
        
        self.mm_to_pixel_label = QLabel("--")
        result_layout.addRow("逆转换:", self.mm_to_pixel_label)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def load_image(self):
        """加载图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择标定图像",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            try:
                self.image = cv2.imread(file_path)
                if self.image is not None:
                    self.image_label.set_image(self.image)
                    self.clear_points_btn.setEnabled(True)
                    self.logger.info(f"加载图像: {Path(file_path).name}")
                else:
                    QMessageBox.warning(self, "加载失败", "无法加载图像！")
            
            except Exception as e:
                QMessageBox.critical(self, "加载失败", f"加载图像失败: {str(e)}")
                self.logger.error(f"加载图像失败: {e}")
    
    def detect_circle(self):
        """自动检测圆形"""
        if self.image is None:
            QMessageBox.warning(self, "未加载图像", "请先加载图像！")
            return
        
        try:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
            # 使用Hough变换检测圆形
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=self.min_radius_spin.value() * 2,
                param1=50,
                param2=30,
                minRadius=int(self.min_radius_spin.value()),
                maxRadius=int(self.max_radius_spin.value())
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                # 取第一个圆
                x, y, r = circles[0]
                self.circle_center = (x, y)
                self.circle_radius = r
                
                # 在图像上显示圆
                display_image = self.image.copy()
                cv2.circle(display_image, (x, y), r, (0, 255, 0), 2)
                cv2.circle(display_image, (x, y), 2, (0, 0, 255), 3)
                
                self.image_label.set_image(display_image)
                
                # 更新信息
                self.center_label.setText(f"({x}, {y})")
                self.radius_label.setText(f"{r:.2f} 像素")
                
                self.calibrate_btn.setEnabled(True)
                
                self.logger.info(f"检测到圆形: 中心({x}, {y}), 半径{r}")
            else:
                QMessageBox.warning(self, "检测失败", "未检测到圆形！")
                self.logger.warning("未检测到圆形")
        
        except Exception as e:
            QMessageBox.critical(self, "检测失败", f"检测圆形失败: {str(e)}")
            self.logger.error(f"检测圆形失败: {e}")
    
    def on_point_selected(self, x: int, y: int):
        """点选中事件"""
        self.circle_center = (x, y)
        self.center_label.setText(f"({x}, {y})")
        
        if self.circle_radius > 0:
            self.calibrate_btn.setEnabled(True)
    
    def clear_points(self):
        """清除选点"""
        self.image_label.clear_points()
        self.circle_center = None
        self.circle_radius = 0
        self.center_label.setText("--")
        self.radius_label.setText("--")
        self.calibrate_btn.setEnabled(False)
        
        # 重新加载原始图像
        if self.image is not None:
            self.image_label.set_image(self.image)
    
    def calibrate(self):
        """执行标定"""
        if self.circle_center is None or self.circle_radius == 0:
            QMessageBox.warning(self, "信息不足", "请先检测圆形！")
            return
        
        try:
            circle_diameter = self.circle_diameter_spin.value()
            
            pixel_to_mm = self.calibrator.calibrate_by_circle(
                self.image,
                self.circle_center,
                self.circle_radius,
                circle_diameter
            )
            
            # 更新结果显示
            self.pixel_to_mm_label.setText(f"{pixel_to_mm:.6f} mm/像素")
            self.mm_to_pixel_label.setText(f"{1.0/pixel_to_mm:.2f} 像素/mm")
            
            self.calibration_finished.emit(pixel_to_mm)
            
            QMessageBox.information(
                self,
                "标定完成",
                f"标定成功！\n转换比例: {pixel_to_mm:.6f} mm/像素"
            )
            
            self.logger.info(f"标定完成: {pixel_to_mm:.6f} mm/像素")
        
        except CalibrationException as e:
            QMessageBox.critical(self, "标定失败", f"标定失败: {str(e)}")
            self.logger.error(f"标定失败: {e}")


class PixelMmCalibrationGUI(QMainWindow):
    """像素-毫米标定GUI主窗口"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(self.__class__.__name__)
        
        self.calibration_manager = CalibrationManager()
        self.current_pixel_to_mm = 0.0
        
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("像素-毫米标定工具 - 微小零件视觉检测系统")
        self.setMinimumSize(1000, 700)
        
        # 中央窗口
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout()
        
        # Tab选项卡
        self.tab_widget = QTabWidget()
        
        # 参考长度标定
        self.reference_widget = ReferenceCalibrationWidget()
        self.reference_widget.calibration_finished.connect(self.on_calibration_finished)
        self.tab_widget.addTab(self.reference_widget, "参考长度标定")
        
        # 圆形标定
        self.circle_widget = CircleCalibrationWidget()
        self.circle_widget.calibration_finished.connect(self.on_calibration_finished)
        self.tab_widget.addTab(self.circle_widget, "圆形标定")
        
        main_layout.addWidget(self.tab_widget)
        
        # 当前标定结果
        current_group = QGroupBox("当前标定结果")
        current_layout = QHBoxLayout()
        
        current_layout.addWidget(QLabel("转换比例:"))
        self.current_pixel_to_mm_label = QLabel("--")
        self.current_pixel_to_mm_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #4CAF50;")
        current_layout.addWidget(self.current_pixel_to_mm_label)
        
        current_layout.addStretch()
        
        self.save_btn = QPushButton("保存标定结果")
        self.save_btn.clicked.connect(self.save_calibration)
        self.save_btn.setEnabled(False)
        current_layout.addWidget(self.save_btn)
        
        self.load_btn = QPushButton("加载标定结果")
        self.load_btn.clicked.connect(self.load_calibration)
        current_layout.addWidget(self.load_btn)
        
        self.apply_btn = QPushButton("应用到系统")
        self.apply_btn.clicked.connect(self.apply_calibration)
        self.apply_btn.setEnabled(False)
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        current_layout.addWidget(self.apply_btn)
        
        current_group.setLayout(current_layout)
        main_layout.addWidget(current_group)
        
        # 标定历史
        history_group = QGroupBox("标定历史")
        history_layout = QVBoxLayout()
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(["时间", "方法", "转换比例", "状态"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.verticalHeader().setVisible(False)
        history_layout.addWidget(self.history_table)
        
        history_group.setLayout(history_layout)
        main_layout.addWidget(history_group)
        
        central_widget.setLayout(main_layout)
    
    def on_calibration_finished(self, pixel_to_mm: float):
        """标定完成回调"""
        self.current_pixel_to_mm = pixel_to_mm
        self.current_pixel_to_mm_label.setText(f"{pixel_to_mm:.6f} mm/像素")
        
        self.save_btn.setEnabled(True)
        self.apply_btn.setEnabled(True)
        
        # 添加到历史记录
        method = "参考长度" if self.tab_widget.currentIndex() == 0 else "圆形"
        self.add_history_record(method, pixel_to_mm)
    
    def add_history_record(self, method: str, pixel_to_mm: float):
        """添加历史记录"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)
        
        self.history_table.setItem(row, 0, QTableWidgetItem(timestamp))
        self.history_table.setItem(row, 1, QTableWidgetItem(method))
        self.history_table.setItem(row, 2, QTableWidgetItem(f"{pixel_to_mm:.6f}"))
        self.history_table.setItem(row, 3, QTableWidgetItem("已完成"))
        
        # 滚动到最新记录
        self.history_table.scrollToBottom()
    
    def save_calibration(self):
        """保存标定结果"""
        if self.current_pixel_to_mm == 0.0:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存标定结果",
            f"pixel_mm_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON文件 (*.json)"
        )
        
        if file_path:
            try:
                calibration_data = {
                    'pixel_to_mm': self.current_pixel_to_mm,
                    'calibrated': True,
                    'timestamp': datetime.now().isoformat()
                }
                
                with open(file_path, 'w') as f:
                    json.dump(calibration_data, f, indent=2)
                
                self.logger.info(f"标定结果已保存: {Path(file_path).name}")
                QMessageBox.information(self, "保存成功", "标定结果已保存！")
            
            except Exception as e:
                self.logger.error(f"保存失败: {e}")
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
                    data = json.load(f)
                
                self.current_pixel_to_mm = data.get('pixel_to_mm', 0.0)
                self.current_pixel_to_mm_label.setText(f"{self.current_pixel_to_mm:.6f} mm/像素")
                
                self.save_btn.setEnabled(True)
                self.apply_btn.setEnabled(True)
                
                # 添加到历史记录
                self.add_history_record("加载", self.current_pixel_to_mm)
                
                self.logger.info(f"标定结果已加载: {Path(file_path).name}")
                QMessageBox.information(self, "加载成功", "标定结果已加载！")
            
            except Exception as e:
                self.logger.error(f"加载失败: {e}")
                QMessageBox.critical(self, "加载失败", f"无法加载标定结果: {str(e)}")
    
    def apply_calibration(self):
        """应用标定结果到系统"""
        if self.current_pixel_to_mm == 0.0:
            return
        
        try:
            # 保存标定结果到默认位置
            default_path = Path("data/pixel_mm_calibration.json")
            default_path.parent.mkdir(exist_ok=True)
            
            calibration_data = {
                'pixel_to_mm': self.current_pixel_to_mm,
                'calibrated': True,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(default_path, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            # 更新配置管理器
            success = self.calibration_manager.pixel_to_mm_calib.load_calibration(str(default_path))
            
            if success:
                QMessageBox.information(
                    self,
                    "应用成功",
                    f"标定结果已应用到系统！\n转换比例: {self.current_pixel_to_mm:.6f} mm/像素"
                )
                self.logger.info(f"标定结果已应用到系统: {self.current_pixel_to_mm:.6f} mm/像素")
            else:
                QMessageBox.warning(self, "应用警告", "标定结果已保存，但未能加载到配置管理器！")
        
        except Exception as e:
            self.logger.error(f"应用失败: {e}")
            QMessageBox.critical(self, "应用失败", f"无法应用标定结果: {str(e)}")


def main():
    """主函数"""
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = PixelMmCalibrationGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()