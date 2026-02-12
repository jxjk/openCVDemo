# -*- coding: utf-8 -*-
"""
配置管理GUI界面
Configuration Management GUI Module
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
    QGroupBox, QFormLayout, QMessageBox, QFileDialog,
    QTabWidget, QTreeWidget, QTreeWidgetItem, QTextEdit,
    QCheckBox, QSplitter, QScrollArea, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor

from config_manager import (
    ConfigManager, InspectionConfig, ToleranceTable,
    get_config, update_config, reload_config, save_default_config
)
from logger_config import get_logger


class ToleranceEditorWidget(QWidget):
    """公差表编辑器"""
    
    tolerance_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_logger(self.__class__.__name__)
        self.tolerance_table = ToleranceTable()
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        # 标准选择
        standard_layout = QHBoxLayout()
        standard_layout.addWidget(QLabel("公差标准:"))
        
        self.standard_combo = QComboBox()
        self.standard_combo.addItems(['IT5', 'IT7', 'IT8', 'IT9', 'IT11'])
        self.standard_combo.currentTextChanged.connect(self.load_tolerance_data)
        standard_layout.addWidget(self.standard_combo)
        
        standard_layout.addStretch()
        layout.addLayout(standard_layout)
        
        # 公差值编辑
        tolerance_group = QGroupBox("公差值 (mm)")
        tolerance_layout = QFormLayout()
        
        self.range_1_3 = QDoubleSpinBox()
        self.range_1_3.setRange(0.001, 1.0)
        self.range_1_3.setSingleStep(0.001)
        self.range_1_3.setDecimals(3)
        tolerance_layout.addRow(">1-3 mm:", self.range_1_3)
        
        self.range_3_6 = QDoubleSpinBox()
        self.range_3_6.setRange(0.001, 1.0)
        self.range_3_6.setSingleStep(0.001)
        self.range_3_6.setDecimals(3)
        tolerance_layout.addRow(">3-6 mm:", self.range_3_6)
        
        self.range_6_10 = QDoubleSpinBox()
        self.range_6_10.setRange(0.001, 1.0)
        self.range_6_10.setSingleStep(0.001)
        self.range_6_10.setDecimals(3)
        tolerance_layout.addRow(">6-10 mm:", self.range_6_10)
        
        self.range_10_18 = QDoubleSpinBox()
        self.range_10_18.setRange(0.001, 1.0)
        self.range_10_18.setSingleStep(0.001)
        self.range_10_18.setDecimals(3)
        tolerance_layout.addRow(">10-18 mm:", self.range_10_18)
        
        self.range_18_30 = QDoubleSpinBox()
        self.range_18_30.setRange(0.001, 1.0)
        self.range_18_30.setSingleStep(0.001)
        self.range_18_30.setDecimals(3)
        tolerance_layout.addRow(">18-30 mm:", self.range_18_30)
        
        self.range_30_40 = QDoubleSpinBox()
        self.range_30_40.setRange(0.001, 1.0)
        self.range_30_40.setSingleStep(0.001)
        self.range_30_40.setDecimals(3)
        tolerance_layout.addRow(">30-40 mm:", self.range_30_40)
        
        tolerance_group.setLayout(tolerance_layout)
        layout.addWidget(tolerance_group)
        
        # 应用按钮
        self.apply_btn = QPushButton("应用更改")
        self.apply_btn.clicked.connect(self.apply_changes)
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        layout.addWidget(self.apply_btn)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # 加载初始数据
        self.load_tolerance_data('IT8')
    
    def load_tolerance_data(self, standard: str):
        """加载公差数据"""
        standard_lower = standard.lower()
        tolerance_dict = getattr(self.tolerance_table, standard_lower)
        
        self.range_1_3.setValue(tolerance_dict.get((1, 3), 0.0))
        self.range_3_6.setValue(tolerance_dict.get((3, 6), 0.0))
        self.range_6_10.setValue(tolerance_dict.get((6, 10), 0.0))
        self.range_10_18.setValue(tolerance_dict.get((10, 18), 0.0))
        self.range_18_30.setValue(tolerance_dict.get((18, 30), 0.0))
        self.range_30_40.setValue(tolerance_dict.get((30, 40), 0.0))
    
    def apply_changes(self):
        """应用更改"""
        standard = self.standard_combo.currentText()
        standard_lower = standard.lower()
        
        # 更新公差表
        tolerance_dict = {
            (1, 3): self.range_1_3.value(),
            (3, 6): self.range_3_6.value(),
            (6, 10): self.range_6_10.value(),
            (10, 18): self.range_10_18.value(),
            (18, 30): self.range_18_30.value(),
            (30, 40): self.range_30_40.value()
        }
        
        setattr(self.tolerance_table, standard_lower, tolerance_dict)
        
        self.tolerance_changed.emit()
        self.logger.info(f"已更新{standard}公差表")
    
    def get_tolerance_table(self) -> ToleranceTable:
        """获取公差表"""
        return self.tolerance_table


class DetectionParamsWidget(QWidget):
    """检测参数编辑器"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_logger(self.__class__.__name__)
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        content = QWidget()
        content_layout = QVBoxLayout()
        
        # 像素转换
        pixel_group = QGroupBox("像素转换")
        pixel_layout = QFormLayout()
        
        self.pixel_to_mm_spin = QDoubleSpinBox()
        self.pixel_to_mm_spin.setRange(0.001, 10.0)
        self.pixel_to_mm_spin.setSingleStep(0.001)
        self.pixel_to_mm_spin.setDecimals(6)
        self.pixel_to_mm_spin.setSuffix(" mm/像素")
        pixel_layout.addRow("像素到毫米:", self.pixel_to_mm_spin)
        
        self.subpixel_precision_spin = QDoubleSpinBox()
        self.subpixel_precision_spin.setRange(0.01, 1.0)
        self.subpixel_precision_spin.setSingleStep(0.01)
        self.subpixel_precision_spin.setDecimals(3)
        self.subpixel_precision_spin.setSuffix(" 像素")
        pixel_layout.addRow("亚像素精度:", self.subpixel_precision_spin)
        
        pixel_group.setLayout(pixel_layout)
        content_layout.addWidget(pixel_group)
        
        # 检测参数
        detection_group = QGroupBox("检测参数")
        detection_layout = QFormLayout()
        
        self.min_circularity_spin = QDoubleSpinBox()
        self.min_circularity_spin.setRange(0.0, 1.0)
        self.min_circularity_spin.setSingleStep(0.1)
        self.min_circularity_spin.setDecimals(2)
        detection_layout.addRow("最小圆度:", self.min_circularity_spin)
        
        self.min_contour_area_spin = QSpinBox()
        self.min_contour_area_spin.setRange(1, 1000000)
        detection_layout.addRow("最小轮廓面积:", self.min_contour_area_spin)
        
        self.max_contour_area_spin = QSpinBox()
        self.max_contour_area_spin.setRange(1, 10000000)
        detection_layout.addRow("最大轮廓面积:", self.max_contour_area_spin)
        
        detection_group.setLayout(detection_layout)
        content_layout.addWidget(detection_group)
        
        # 边缘检测参数
        edge_group = QGroupBox("边缘检测参数")
        edge_layout = QFormLayout()
        
        self.canny_threshold1_spin = QSpinBox()
        self.canny_threshold1_spin.setRange(0, 500)
        edge_layout.addRow("Canny阈值1:", self.canny_threshold1_spin)
        
        self.canny_threshold2_spin = QSpinBox()
        self.canny_threshold2_spin.setRange(0, 500)
        edge_layout.addRow("Canny阈值2:", self.canny_threshold2_spin)
        
        self.aperture_size_spin = QSpinBox()
        self.aperture_size_spin.setRange(1, 7)
        self.aperture_size_spin.setSingleStep(2)
        edge_layout.addRow("孔径大小:", self.aperture_size_spin)
        
        edge_group.setLayout(edge_layout)
        content_layout.addWidget(edge_group)
        
        # 圆检测参数
        circle_group = QGroupBox("圆检测参数")
        circle_layout = QFormLayout()
        
        self.dp_spin = QDoubleSpinBox()
        self.dp_spin.setRange(1.0, 10.0)
        self.dp_spin.setSingleStep(0.1)
        circle_layout.addRow("dp:", self.dp_spin)
        
        self.min_dist_spin = QSpinBox()
        self.min_dist_spin.setRange(1, 1000)
        circle_layout.addRow("最小圆心距:", self.min_dist_spin)
        
        self.param1_spin = QSpinBox()
        self.param1_spin.setRange(1, 500)
        circle_layout.addRow("参数1:", self.param1_spin)
        
        self.param2_spin = QSpinBox()
        self.param2_spin.setRange(1, 500)
        circle_layout.addRow("参数2:", self.param2_spin)
        
        self.min_radius_spin = QSpinBox()
        self.min_radius_spin.setRange(1, 500)
        circle_layout.addRow("最小半径:", self.min_radius_spin)
        
        self.max_radius_spin = QSpinBox()
        self.max_radius_spin.setRange(1, 1000)
        circle_layout.addRow("最大半径:", self.max_radius_spin)
        
        circle_group.setLayout(circle_layout)
        content_layout.addWidget(circle_group)
        
        # RANSAC参数
        ransac_group = QGroupBox("RANSAC参数")
        ransac_layout = QFormLayout()
        
        self.ransac_iterations_spin = QSpinBox()
        self.ransac_iterations_spin.setRange(100, 10000)
        ransac_layout.addRow("迭代次数:", self.ransac_iterations_spin)
        
        self.ransac_threshold_spin = QDoubleSpinBox()
        self.ransac_threshold_spin.setRange(0.001, 1.0)
        self.ransac_threshold_spin.setSingleStep(0.001)
        self.ransac_threshold_spin.setDecimals(3)
        ransac_layout.addRow("阈值:", self.ransac_threshold_spin)
        
        ransac_group.setLayout(ransac_layout)
        content_layout.addWidget(ransac_group)
        
        content_layout.addStretch()
        content.setLayout(content_layout)
        scroll.setWidget(content)
        
        layout.addWidget(scroll)
        self.setLayout(layout)
    
    def load_config(self, config: InspectionConfig):
        """加载配置"""
        self.pixel_to_mm_spin.setValue(config.pixel_to_mm)
        self.subpixel_precision_spin.setValue(config.subpixel_precision)
        self.min_circularity_spin.setValue(config.min_circularity)
        self.min_contour_area_spin.setValue(config.min_contour_area)
        self.max_contour_area_spin.setValue(config.max_contour_area)
        self.canny_threshold1_spin.setValue(config.canny_threshold1)
        self.canny_threshold2_spin.setValue(config.canny_threshold2)
        self.aperture_size_spin.setValue(config.aperture_size)
        self.dp_spin.setValue(config.dp)
        self.min_dist_spin.setValue(config.min_dist)
        self.param1_spin.setValue(config.param1)
        self.param2_spin.setValue(config.param2)
        self.min_radius_spin.setValue(config.min_radius)
        self.max_radius_spin.setValue(config.max_radius)
        self.ransac_iterations_spin.setValue(config.ransac_iterations)
        self.ransac_threshold_spin.setValue(config.ransac_threshold)
    
    def get_config_dict(self) -> Dict[str, Any]:
        """获取配置字典"""
        return {
            'pixel_to_mm': self.pixel_to_mm_spin.value(),
            'subpixel_precision': self.subpixel_precision_spin.value(),
            'min_circularity': self.min_circularity_spin.value(),
            'min_contour_area': self.min_contour_area_spin.value(),
            'max_contour_area': self.max_contour_area_spin.value(),
            'canny_threshold1': self.canny_threshold1_spin.value(),
            'canny_threshold2': self.canny_threshold2_spin.value(),
            'aperture_size': self.aperture_size_spin.value(),
            'dp': self.dp_spin.value(),
            'min_dist': self.min_dist_spin.value(),
            'param1': self.param1_spin.value(),
            'param2': self.param2_spin.value(),
            'min_radius': self.min_radius_spin.value(),
            'max_radius': self.max_radius_spin.value(),
            'ransac_iterations': self.ransac_iterations_spin.value(),
            'ransac_threshold': self.ransac_threshold_spin.value()
        }


class PathConfigWidget(QWidget):
    """路径配置编辑器"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_logger(self.__class__.__name__)
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        form_layout = QFormLayout()
        
        self.data_dir_edit = QLineEdit()
        data_dir_btn = QPushButton("浏览...")
        data_dir_btn.clicked.connect(lambda: self.browse_directory(self.data_dir_edit))
        data_dir_layout = QHBoxLayout()
        data_dir_layout.addWidget(self.data_dir_edit)
        data_dir_layout.addWidget(data_dir_btn)
        form_layout.addRow("数据目录:", data_dir_layout)
        
        self.template_dir_edit = QLineEdit()
        template_dir_btn = QPushButton("浏览...")
        template_dir_btn.clicked.connect(lambda: self.browse_directory(self.template_dir_edit))
        template_dir_layout = QHBoxLayout()
        template_dir_layout.addWidget(self.template_dir_edit)
        template_dir_layout.addWidget(template_dir_btn)
        form_layout.addRow("模板目录:", template_dir_layout)
        
        self.image_dir_edit = QLineEdit()
        image_dir_btn = QPushButton("浏览...")
        image_dir_btn.clicked.connect(lambda: self.browse_directory(self.image_dir_edit))
        image_dir_layout = QHBoxLayout()
        image_dir_layout.addWidget(self.image_dir_edit)
        image_dir_layout.addWidget(image_dir_btn)
        form_layout.addRow("图像目录:", image_dir_layout)
        
        self.calibration_file_edit = QLineEdit()
        calibration_file_btn = QPushButton("浏览...")
        calibration_file_btn.clicked.connect(lambda: self.browse_file(self.calibration_file_edit))
        calibration_file_layout = QHBoxLayout()
        calibration_file_layout.addWidget(self.calibration_file_edit)
        calibration_file_layout.addWidget(calibration_file_btn)
        form_layout.addRow("标定文件:", calibration_file_layout)
        
        layout.addLayout(form_layout)
        layout.addStretch()
        self.setLayout(layout)
    
    def browse_directory(self, line_edit: QLineEdit):
        """浏览目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择目录")
        if directory:
            line_edit.setText(directory)
    
    def browse_file(self, line_edit: QLineEdit):
        """浏览文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件")
        if file_path:
            line_edit.setText(file_path)
    
    def load_config(self, config: InspectionConfig):
        """加载配置"""
        self.data_dir_edit.setText(config.data_dir)
        self.template_dir_edit.setText(config.template_dir)
        self.image_dir_edit.setText(config.image_dir)
        self.calibration_file_edit.setText(config.calibration_file)
    
    def get_config_dict(self) -> Dict[str, str]:
        """获取配置字典"""
        return {
            'data_dir': self.data_dir_edit.text(),
            'template_dir': self.template_dir_edit.text(),
            'image_dir': self.image_dir_edit.text(),
            'calibration_file': self.calibration_file_edit.text()
        }


class ConfigEditorGUI(QMainWindow):
    """配置编辑器GUI主窗口"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(self.__class__.__name__)
        
        self.config_file = "config.json"
        self.current_config = None
        
        self.init_ui()
        self.load_config()
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("配置管理 - 微小零件视觉检测系统")
        self.setMinimumSize(1000, 700)
        
        # 中央窗口
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout()
        
        # 工具栏
        toolbar_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("加载配置")
        self.load_btn.clicked.connect(self.load_config)
        toolbar_layout.addWidget(self.load_btn)
        
        self.save_btn = QPushButton("保存配置")
        self.save_btn.clicked.connect(self.save_config)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        toolbar_layout.addWidget(self.save_btn)
        
        self.save_as_btn = QPushButton("另存为")
        self.save_as_btn.clicked.connect(self.save_config_as)
        toolbar_layout.addWidget(self.save_as_btn)
        
        self.reload_btn = QPushButton("重新加载")
        self.reload_btn.clicked.connect(self.reload_config)
        toolbar_layout.addWidget(self.reload_btn)
        
        self.reset_btn = QPushButton("重置为默认")
        self.reset_btn.clicked.connect(self.reset_to_default)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        toolbar_layout.addWidget(self.reset_btn)
        
        toolbar_layout.addStretch()
        
        self.config_file_label = QLabel("配置文件: config.json")
        self.config_file_label.setStyleSheet("color: #888;")
        toolbar_layout.addWidget(self.config_file_label)
        
        main_layout.addLayout(toolbar_layout)
        
        # Tab选项卡
        self.tab_widget = QTabWidget()
        
        # 检测参数
        self.detection_params = DetectionParamsWidget()
        self.tab_widget.addTab(self.detection_params, "检测参数")
        
        # 公差表
        self.tolerance_editor = ToleranceEditorWidget()
        self.tab_widget.addTab(self.tolerance_editor, "公差表")
        
        # 路径配置
        self.path_config = PathConfigWidget()
        self.tab_widget.addTab(self.path_config, "路径配置")
        
        # 配置预览
        self.preview_widget = QTextEdit()
        self.preview_widget.setReadOnly(True)
        self.preview_widget.setFont(QFont("Courier New", 9))
        self.tab_widget.addTab(self.preview_widget, "配置预览")
        
        main_layout.addWidget(self.tab_widget)
        
        # 状态栏
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #4CAF50;")
        main_layout.addWidget(self.status_label)
        
        central_widget.setLayout(main_layout)
    
    def load_config(self, config_file: Optional[str] = None):
        """加载配置"""
        try:
            if config_file:
                self.config_file = config_file
                self.config_file_label.setText(f"配置文件: {Path(config_file).name}")
            
            self.current_config = ConfigManager.load_config(self.config_file)
            
            # 更新UI
            self.detection_params.load_config(self.current_config)
            self.path_config.load_config(self.current_config)
            
            # 更新预览
            self.update_preview()
            
            self.status_label.setText("配置已加载")
            self.logger.info(f"配置已加载: {self.config_file}")
        
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"无法加载配置: {str(e)}")
            self.logger.error(f"加载配置失败: {e}")
            self.status_label.setText("加载失败")
            self.status_label.setStyleSheet("color: #F44336;")
    
    def save_config(self):
        """保存配置"""
        try:
            if self.current_config is None:
                self.current_config = InspectionConfig()
            
            # 更新配置
            detection_params = self.detection_params.get_config_dict()
            for key, value in detection_params.items():
                setattr(self.current_config, key, value)
            
            # 更新公差表
            self.current_config.tolerance_table = self.tolerance_editor.get_tolerance_table()
            
            # 更新路径配置
            path_params = self.path_config.get_config_dict()
            for key, value in path_params.items():
                setattr(self.current_config, key, value)
            
            # 保存
            success = ConfigManager.save_config(self.current_config, self.config_file)
            
            if success:
                self.status_label.setText("配置已保存")
                self.logger.info(f"配置已保存: {self.config_file}")
                QMessageBox.information(self, "保存成功", "配置已保存！")
            else:
                QMessageBox.warning(self, "保存失败", "配置保存失败！")
        
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"无法保存配置: {str(e)}")
            self.logger.error(f"保存配置失败: {e}")
            self.status_label.setText("保存失败")
            self.status_label.setStyleSheet("color: #F44336;")
    
    def save_config_as(self):
        """另存为"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存配置",
            "",
            "JSON文件 (*.json)"
        )
        
        if file_path:
            try:
                if self.current_config is None:
                    self.current_config = InspectionConfig()
                
                # 更新配置
                detection_params = self.detection_params.get_config_dict()
                for key, value in detection_params.items():
                    setattr(self.current_config, key, value)
                
                # 更新公差表
                self.current_config.tolerance_table = self.tolerance_editor.get_tolerance_table()
                
                # 更新路径配置
                path_params = self.path_config.get_config_dict()
                for key, value in path_params.items():
                    setattr(self.current_config, key, value)
                
                # 保存
                success = ConfigManager.save_config(self.current_config, file_path)
                
                if success:
                    self.config_file = file_path
                    self.config_file_label.setText(f"配置文件: {Path(file_path).name}")
                    self.status_label.setText("配置已保存")
                    self.logger.info(f"配置已保存: {file_path}")
                    QMessageBox.information(self, "保存成功", "配置已保存！")
                else:
                    QMessageBox.warning(self, "保存失败", "配置保存失败！")
            
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"无法保存配置: {str(e)}")
                self.logger.error(f"保存配置失败: {e}")
    
    def reload_config(self):
        """重新加载配置"""
        try:
            self.current_config = ConfigManager.reload_config()
            
            # 更新UI
            self.detection_params.load_config(self.current_config)
            self.path_config.load_config(self.current_config)
            
            # 更新预览
            self.update_preview()
            
            self.status_label.setText("配置已重新加载")
            self.logger.info("配置已重新加载")
            QMessageBox.information(self, "重新加载", "配置已重新加载！")
        
        except Exception as e:
            QMessageBox.critical(self, "重新加载失败", f"无法重新加载配置: {str(e)}")
            self.logger.error(f"重新加载配置失败: {e}")
    
    def reset_to_default(self):
        """重置为默认"""
        reply = QMessageBox.question(
            self,
            "确认重置",
            "确定要将所有配置重置为默认值吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                self.current_config = InspectionConfig()
                
                # 更新UI
                self.detection_params.load_config(self.current_config)
                self.path_config.load_config(self.current_config)
                
                # 更新预览
                self.update_preview()
                
                self.status_label.setText("已重置为默认")
                self.logger.info("配置已重置为默认")
                QMessageBox.information(self, "重置成功", "配置已重置为默认值！")
            
            except Exception as e:
                QMessageBox.critical(self, "重置失败", f"无法重置配置: {str(e)}")
                self.logger.error(f"重置配置失败: {e}")
    
    def update_preview(self):
        """更新配置预览"""
        if self.current_config is None:
            self.preview_widget.setText("无配置数据")
            return
        
        try:
            config_dict = self.current_config.to_dict()
            json_text = json.dumps(config_dict, indent=2, ensure_ascii=False)
            self.preview_widget.setText(json_text)
        except Exception as e:
            self.preview_widget.setText(f"无法显示配置: {str(e)}")


def main():
    """主函数"""
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = ConfigEditorGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()