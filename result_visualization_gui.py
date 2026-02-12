# -*- coding: utf-8 -*-
"""
检测结果可视化增强模块
Result Visualization Enhancement Module
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QFormLayout, QMessageBox, QFileDialog,
    QTabWidget, QComboBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QCheckBox, QSplitter
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

from logger_config import get_logger
from config_manager import get_config, get_tolerance


class DetectionResult:
    """检测结果数据类"""
    
    def __init__(self, feature_name: str, measured_value: float, 
                 nominal_value: float, tolerance: float, 
                 is_pass: bool, unit: str = "mm"):
        self.feature_name = feature_name
        self.measured_value = measured_value
        self.nominal_value = nominal_value
        self.tolerance = tolerance
        self.is_pass = is_pass
        self.unit = unit
        self.deviation = measured_value - nominal_value
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'feature_name': self.feature_name,
            'measured_value': self.measured_value,
            'nominal_value': self.nominal_value,
            'tolerance': self.tolerance,
            'is_pass': self.is_pass,
            'unit': self.unit,
            'deviation': self.deviation
        }


class ResultImageWidget(QWidget):
    """结果图像显示窗口"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_logger(self.__class__.__name__)
        self.original_image = None
        self.results: List[DetectionResult] = []
        
        # 显示选项
        self.show_dimensions = True
        self.show_tolerance = True
        self.show_color_coding = True
        
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        # 图像显示
        self.image_label = QLabel()
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #555;
                background-color: #333;
            }
        """)
        self.image_label.setText("检测结果图像")
        layout.addWidget(self.image_label)
        
        # 工具栏
        toolbar_layout = QHBoxLayout()
        
        self.load_image_btn = QPushButton("加载图像")
        self.load_image_btn.clicked.connect(self.load_image)
        toolbar_layout.addWidget(self.load_image_btn)
        
        self.export_image_btn = QPushButton("导出结果图像")
        self.export_image_btn.clicked.connect(self.export_image)
        self.export_image_btn.setEnabled(False)
        toolbar_layout.addWidget(self.export_image_btn)
        
        self.clear_btn = QPushButton("清除")
        self.clear_btn.clicked.connect(self.clear_all)
        toolbar_layout.addWidget(self.clear_btn)
        
        layout.addLayout(toolbar_layout)
        
        self.setLayout(layout)
    
    def load_image(self):
        """加载图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择检测图像",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            try:
                self.original_image = cv2.imread(file_path)
                if self.original_image is not None:
                    self.update_display()
                    self.export_image_btn.setEnabled(True)
                    self.logger.info(f"加载图像: {Path(file_path).name}")
                else:
                    QMessageBox.warning(self, "加载失败", "无法加载图像！")
            
            except Exception as e:
                QMessageBox.critical(self, "加载失败", f"加载图像失败: {str(e)}")
                self.logger.error(f"加载图像失败: {e}")
    
    def add_result(self, result: DetectionResult):
        """添加检测结果"""
        self.results.append(result)
        self.update_display()
    
    def update_display(self):
        """更新显示"""
        if self.original_image is None:
            return
        
        display_image = self.original_image.copy()
        
        # 绘制检测结果
        for result in self.results:
            self.draw_result(display_image, result)
        
        # 转换为QImage
        h, w, ch = display_image.shape
        bytes_per_line = ch * w
        q_image = QImage(display_image.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # 缩放以适应窗口
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.image_label.setPixmap(scaled_pixmap)
    
    def draw_result(self, image: np.ndarray, result: DetectionResult):
        """绘制单个检测结果"""
        # 这里简化处理，实际应该根据特征类型绘制不同的标注
        # 例如：圆形、矩形、线段等
        
        # 颜色编码
        if self.show_color_coding:
            if result.is_pass:
                color = (0, 255, 0)  # 绿色
            else:
                color = (0, 0, 255)  # 红色
        else:
            color = (255, 255, 0)  # 黄色
        
        # 在图像中心绘制示例标注
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 绘制特征边界框（示例）
        box_size = 100
        cv2.rectangle(image, 
                     (center[0] - box_size, center[1] - box_size),
                     (center[0] + box_size, center[1] + box_size),
                     color, 2)
        
        # 绘制尺寸标注
        if self.show_dimensions:
            text = f"{result.feature_name}: {result.measured_value:.3f}{result.unit}"
            cv2.putText(image, text, (center[0] - box_size, center[1] - box_size - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 绘制公差信息
        if self.show_tolerance:
            tolerance_text = f"公差: ±{result.tolerance:.3f}{result.unit}"
            cv2.putText(image, tolerance_text, (center[0] - box_size, center[1] + box_size + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 绘制偏差
        deviation_text = f"偏差: {result.deviation:+.3f}{result.unit}"
        cv2.putText(image, deviation_text, (center[0] - box_size, center[1] + box_size + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def export_image(self):
        """导出结果图像"""
        if self.original_image is None:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出结果图像",
            f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG文件 (*.png);;JPEG文件 (*.jpg);;BMP文件 (*.bmp)"
        )
        
        if file_path:
            try:
                # 重新生成带标注的图像
                display_image = self.original_image.copy()
                for result in self.results:
                    self.draw_result(display_image, result)
                
                cv2.imwrite(file_path, display_image)
                self.logger.info(f"结果图像已导出: {Path(file_path).name}")
                QMessageBox.information(self, "导出成功", "结果图像已导出！")
            
            except Exception as e:
                self.logger.error(f"导出失败: {e}")
                QMessageBox.critical(self, "导出失败", f"无法导出结果图像: {str(e)}")
    
    def clear_all(self):
        """清除所有内容"""
        self.original_image = None
        self.results.clear()
        self.image_label.setText("检测结果图像")
        self.image_label.clear()
        self.export_image_btn.setEnabled(False)
    
    def set_display_options(self, show_dimensions: bool, show_tolerance: bool, show_color_coding: bool):
        """设置显示选项"""
        self.show_dimensions = show_dimensions
        self.show_tolerance = show_tolerance
        self.show_color_coding = show_color_coding
        self.update_display()


class ResultTableWidget(QWidget):
    """结果表格窗口"""
    
    result_selected = pyqtSignal(int)  # 结果选中信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.results: List[DetectionResult] = []
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()
        
        # 统计信息
        stats_layout = QHBoxLayout()
        
        self.total_label = QLabel("总计: 0")
        stats_layout.addWidget(self.total_label)
        
        self.pass_label = QLabel("合格: 0")
        self.pass_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        stats_layout.addWidget(self.pass_label)
        
        self.fail_label = QLabel("不合格: 0")
        self.fail_label.setStyleSheet("color: #F44336; font-weight: bold;")
        stats_layout.addWidget(self.fail_label)
        
        self.pass_rate_label = QLabel("合格率: 0%")
        stats_layout.addWidget(self.pass_rate_label)
        
        stats_layout.addStretch()
        layout.addLayout(stats_layout)
        
        # 结果表格
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(6)
        self.result_table.setHorizontalHeaderLabels([
            "特征名称", "标称值", "实测值", "公差", "偏差", "状态"
        ])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.result_table.itemSelectionChanged.connect(self.on_selection_changed)
        layout.addWidget(self.result_table)
        
        # 操作按钮
        btn_layout = QHBoxLayout()
        
        self.export_csv_btn = QPushButton("导出CSV")
        self.export_csv_btn.clicked.connect(self.export_csv)
        self.export_csv_btn.setEnabled(False)
        btn_layout.addWidget(self.export_csv_btn)
        
        self.export_excel_btn = QPushButton("导出Excel")
        self.export_excel_btn.clicked.connect(self.export_excel)
        self.export_excel_btn.setEnabled(False)
        btn_layout.addWidget(self.export_excel_btn)
        
        self.clear_table_btn = QPushButton("清除表格")
        self.clear_table_btn.clicked.connect(self.clear_table)
        btn_layout.addWidget(self.clear_table_btn)
        
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
    
    def add_result(self, result: DetectionResult):
        """添加检测结果"""
        self.results.append(result)
        self.update_table()
        self.update_stats()
    
    def update_table(self):
        """更新表格"""
        self.result_table.setRowCount(len(self.results))
        
        for i, result in enumerate(self.results):
            # 特征名称
            self.result_table.setItem(i, 0, QTableWidgetItem(result.feature_name))
            
            # 标称值
            self.result_table.setItem(i, 1, 
                QTableWidgetItem(f"{result.nominal_value:.3f} {result.unit}"))
            
            # 实测值
            measured_item = QTableWidgetItem(f"{result.measured_value:.3f} {result.unit}")
            if not result.is_pass:
                measured_item.setForeground(QColor(244, 67, 54))
            self.result_table.setItem(i, 2, measured_item)
            
            # 公差
            self.result_table.setItem(i, 3, 
                QTableWidgetItem(f"±{result.tolerance:.3f} {result.unit}"))
            
            # 偏差
            deviation_item = QTableWidgetItem(f"{result.deviation:+.3f} {result.unit}")
            if abs(result.deviation) > result.tolerance:
                deviation_item.setForeground(QColor(244, 67, 54))
            self.result_table.setItem(i, 4, deviation_item)
            
            # 状态
            status_item = QTableWidgetItem("合格" if result.is_pass else "不合格")
            if result.is_pass:
                status_item.setForeground(QColor(76, 175, 80))
            else:
                status_item.setForeground(QColor(244, 67, 54))
                status_item.setFont(QFont("Arial", 9, QFont.Bold))
            self.result_table.setItem(i, 5, status_item)
        
        # 更新按钮状态
        has_results = len(self.results) > 0
        self.export_csv_btn.setEnabled(has_results)
        self.export_excel_btn.setEnabled(has_results)
    
    def update_stats(self):
        """更新统计信息"""
        total = len(self.results)
        pass_count = sum(1 for r in self.results if r.is_pass)
        fail_count = total - pass_count
        pass_rate = (pass_count / total * 100) if total > 0 else 0
        
        self.total_label.setText(f"总计: {total}")
        self.pass_label.setText(f"合格: {pass_count}")
        self.fail_label.setText(f"不合格: {fail_count}")
        self.pass_rate_label.setText(f"合格率: {pass_rate:.1f}%")
    
    def on_selection_changed(self):
        """选择改变事件"""
        selected_rows = set()
        for item in self.result_table.selectedItems():
            selected_rows.add(item.row())
        
        if selected_rows:
            self.result_selected.emit(list(selected_rows)[0])
    
    def export_csv(self):
        """导出CSV"""
        if not self.results:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出CSV",
            f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV文件 (*.csv)"
        )
        
        if file_path:
            try:
                import csv
                
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['特征名称', '标称值', '实测值', '公差', '偏差', '状态'])
                    
                    for result in self.results:
                        writer.writerow([
                            result.feature_name,
                            f"{result.nominal_value:.3f}",
                            f"{result.measured_value:.3f}",
                            f"{result.tolerance:.3f}",
                            f"{result.deviation:.3f}",
                            "合格" if result.is_pass else "不合格"
                        ])
                
                QMessageBox.information(self, "导出成功", "结果已导出为CSV！")
            
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"无法导出CSV: {str(e)}")
    
    def export_excel(self):
        """导出Excel"""
        if not self.results:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出Excel",
            f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "Excel文件 (*.xlsx)"
        )
        
        if file_path:
            try:
                import pandas as pd
                
                data = [result.to_dict() for result in self.results]
                df = pd.DataFrame(data)
                
                df.to_excel(file_path, index=False)
                
                QMessageBox.information(self, "导出成功", "结果已导出为Excel！")
            
            except ImportError:
                QMessageBox.warning(self, "缺少依赖", "请安装openpyxl和pandas库！")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"无法导出Excel: {str(e)}")
    
    def clear_table(self):
        """清除表格"""
        self.results.clear()
        self.result_table.setRowCount(0)
        self.update_stats()
        self.export_csv_btn.setEnabled(False)
        self.export_excel_btn.setEnabled(False)


class ResultVisualizationGUI(QMainWindow):
    """结果可视化GUI主窗口"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(self.__class__.__name__)
        
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("检测结果可视化 - 微小零件视觉检测系统")
        self.setMinimumSize(1200, 800)
        
        # 中央窗口
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout()
        
        # Splitter分割窗口
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：图像显示
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        image_group = QGroupBox("结果图像")
        image_layout = QVBoxLayout()
        
        self.image_widget = ResultImageWidget()
        image_layout.addWidget(self.image_widget)
        
        image_group.setLayout(image_layout)
        left_layout.addWidget(image_group)
        
        # 显示选项
        display_group = QGroupBox("显示选项")
        display_layout = QHBoxLayout()
        
        self.show_dimensions_check = QCheckBox("显示尺寸")
        self.show_dimensions_check.setChecked(True)
        self.show_dimensions_check.toggled.connect(self.update_display_options)
        display_layout.addWidget(self.show_dimensions_check)
        
        self.show_tolerance_check = QCheckBox("显示公差")
        self.show_tolerance_check.setChecked(True)
        self.show_tolerance_check.toggled.connect(self.update_display_options)
        display_layout.addWidget(self.show_tolerance_check)
        
        self.show_color_coding_check = QCheckBox("颜色编码")
        self.show_color_coding_check.setChecked(True)
        self.show_color_coding_check.toggled.connect(self.update_display_options)
        display_layout.addWidget(self.show_color_coding_check)
        
        display_layout.addStretch()
        display_group.setLayout(display_layout)
        left_layout.addWidget(display_group)
        
        left_widget.setLayout(left_layout)
        splitter.addWidget(left_widget)
        
        # 右侧：结果表格
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        table_group = QGroupBox("检测结果")
        table_layout = QVBoxLayout()
        
        self.table_widget = ResultTableWidget()
        table_layout.addWidget(self.table_widget)
        
        table_group.setLayout(table_layout)
        right_layout.addWidget(table_group)
        
        # 添加测试结果
        test_group = QGroupBox("添加测试结果")
        test_layout = QFormLayout()
        
        self.feature_name_edit = QComboBox()
        self.feature_name_edit.setEditable(True)
        self.feature_name_edit.addItems(['直径', '长度', '宽度', '高度', '圆度'])
        test_layout.addRow("特征名称:", self.feature_name_edit)
        
        self.nominal_value_spin = QDoubleSpinBox()
        self.nominal_value_spin.setRange(0.0, 1000.0)
        self.nominal_value_spin.setValue(10.0)
        self.nominal_value_spin.setSuffix(" mm")
        test_layout.addRow("标称值:", self.nominal_value_spin)
        
        self.measured_value_spin = QDoubleSpinBox()
        self.measured_value_spin.setRange(0.0, 1000.0)
        self.measured_value_spin.setValue(10.0)
        self.measured_value_spin.setSingleStep(0.001)
        self.measured_value_spin.setSuffix(" mm")
        test_layout.addRow("实测值:", self.measured_value_spin)
        
        tolerance_standard_combo = QComboBox()
        tolerance_standard_combo.addItems(['IT5', 'IT7', 'IT8', 'IT9', 'IT11'])
        tolerance_standard_combo.setCurrentText('IT8')
        test_layout.addRow("公差标准:", tolerance_standard_combo)
        
        add_btn = QPushButton("添加结果")
        add_btn.clicked.connect(lambda: self.add_test_result(tolerance_standard_combo.currentText()))
        add_btn.setStyleSheet("""
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
        test_layout.addRow(add_btn)
        
        test_group.setLayout(test_layout)
        right_layout.addWidget(test_group)
        
        right_widget.setLayout(right_layout)
        splitter.addWidget(right_widget)
        
        # 设置分割比例
        splitter.setSizes([600, 400])
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
    
    def update_display_options(self):
        """更新显示选项"""
        self.image_widget.set_display_options(
            self.show_dimensions_check.isChecked(),
            self.show_tolerance_check.isChecked(),
            self.show_color_coding_check.isChecked()
        )
    
    def add_test_result(self, tolerance_standard: str):
        """添加测试结果"""
        feature_name = self.feature_name_edit.currentText()
        nominal_value = self.nominal_value_spin.value()
        measured_value = self.measured_value_spin.value()
        
        # 获取公差
        tolerance = get_tolerance(nominal_value, tolerance_standard)
        
        # 判断是否合格
        deviation = measured_value - nominal_value
        is_pass = abs(deviation) <= tolerance
        
        # 创建结果
        result = DetectionResult(
            feature_name=feature_name,
            measured_value=measured_value,
            nominal_value=nominal_value,
            tolerance=tolerance,
            is_pass=is_pass,
            unit="mm"
        )
        
        # 添加到表格和图像
        self.table_widget.add_result(result)
        self.image_widget.add_result(result)
        
        self.logger.info(f"添加测试结果: {feature_name}={measured_value:.3f}mm, "
                        f"标称={nominal_value:.3f}mm, {'合格' if is_pass else '不合格'}")


def main():
    """主函数"""
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = ResultVisualizationGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()