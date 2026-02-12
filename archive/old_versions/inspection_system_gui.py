# -*- coding: utf-8 -*-
"""
微小零件中高精度视觉检测系统 - GUI版本
Micro Part High-Precision Visual Inspection System - GUI Version

版本: V1.0
创建日期: 2026-02-10
精度目标: IT8级（亚像素检测精度≥1/20像素）
GUI框架: Tkinter

功能:
- 图形用户界面
- 实时图像预览
- 一键检测
- 数据统计显示
- 历史记录查询
"""

import os
import sys
import time
import csv
import json
import datetime
import threading
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

# 导入核心模块
from inspection_system import (
    InspectionConfig,
    CameraCalibration,
    USBCameraDriver,
    DahengCameraDriver,
    SubpixelDetector,
    GeometryFitter,
    InspectionResult,
    InspectionEngine,
    DataManager
)


# =============================================================================
# GUI主窗口
# =============================================================================

class InspectionSystemGUI:
    """检测系统GUI主窗口"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("微小零件中高精度视觉检测系统 V1.0")
        self.root.geometry("1200x800")
        
        # 初始化核心组件
        self.config = InspectionConfig()
        self.calibration = CameraCalibration(self.config)
        self.camera_driver = None
        self.inspection_engine = InspectionEngine(self.config, self.calibration)
        self.data_manager = DataManager(self.config)
        
        # 加载配置和标定数据
        self._load_config()
        self.calibration.load_calibration()
        
        # 控制变量
        self.is_camera_connected = tk.BooleanVar(value=False)
        self.is_previewing = tk.BooleanVar(value=False)
        self.preview_thread = None
        self.stop_preview = False
        
        # 零件参数
        self.part_id_var = tk.StringVar()
        self.part_type_var = tk.StringVar(value="圆形")
        self.nominal_size_var = tk.StringVar(value="10.0")
        self.pixel_to_mm_var = tk.StringVar(value=f"{self.config.PIXEL_TO_MM:.6f}")
        
        # 检测结果
        self.result_display_var = tk.StringVar(value="等待检测...")
        self.diameter_var = tk.StringVar(value="--")
        self.deviation_var = tk.StringVar(value="--")
        self.tolerance_var = tk.StringVar(value="--")
        self.is_qualified_var = tk.StringVar(value="--")
        self.qualified_count_var = tk.StringVar(value="0")
        self.unqualified_count_var = tk.StringVar(value="0")
        self.qualified_rate_var = tk.StringVar(value="0%")
        
        # 创建界面
        self._create_menu()
        self._create_layout()
        self._update_camera_status()
    
    def _load_config(self):
        """加载配置"""
        if os.path.exists('data/config.json'):
            with open('data/config.json', 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                self.config.PIXEL_TO_MM = config_data.get('pixel_to_mm', 0.098)
                self.pixel_to_mm_var.set(f"{self.config.PIXEL_TO_MM:.6f}")
    
    def _save_config(self):
        """保存配置"""
        config_data = {
            'pixel_to_mm': self.config.PIXEL_TO_MM,
            'calibration_date': datetime.datetime.now().isoformat()
        }
        
        with open('data/config.json', 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
    
    def _create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="导出数据到Excel", command=self._export_data)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self._on_close)
        menubar.add_cascade(label="文件", menu=file_menu)
        
        # 相机菜单
        camera_menu = tk.Menu(menubar, tearoff=0)
        camera_menu.add_command(label="连接USB相机", command=lambda: self._connect_camera("usb"))
        camera_menu.add_command(label="连接大恒相机", command=lambda: self._connect_camera("daheng"))
        camera_menu.add_separator()
        camera_menu.add_command(label="断开相机", command=self._disconnect_camera)
        menubar.add_cascade(label="相机", menu=camera_menu)
        
        # 标定菜单
        calib_menu = tk.Menu(menubar, tearoff=0)
        calib_menu.add_command(label="像素-毫米标定", command=self._show_calibration_dialog)
        calib_menu.add_command(label="相机内参标定", command=self._show_camera_calibration_dialog)
        menubar.add_cascade(label="标定", menu=calib_menu)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="关于", command=self._show_about)
        menubar.add_cascade(label="帮助", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def _create_layout(self):
        """创建界面布局"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧面板 - 图像显示
        left_panel = ttk.LabelFrame(main_frame, text="图像预览", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 图像显示区域
        self.image_label = ttk.Label(left_panel, text="未连接相机", 
                                    font=("Arial", 16), foreground="gray")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # 右侧面板 - 控制和结果
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # 相机控制区
        camera_control_frame = ttk.LabelFrame(right_panel, text="相机控制", padding=10)
        camera_control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self._create_camera_control(camera_control_frame)
        
        # 零件参数区
        part_param_frame = ttk.LabelFrame(right_panel, text="零件参数", padding=10)
        part_param_frame.pack(fill=tk.X, pady=(0, 10))
        
        self._create_part_params(part_param_frame)
        
        # 检测控制区
        inspect_control_frame = ttk.LabelFrame(right_panel, text="检测控制", padding=10)
        inspect_control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self._create_inspect_control(inspect_control_frame)
        
        # 检测结果区
        result_frame = ttk.LabelFrame(right_panel, text="检测结果", padding=10)
        result_frame.pack(fill=tk.X, pady=(0, 10))
        
        self._create_result_display(result_frame)
        
        # 统计信息区
        stats_frame = ttk.LabelFrame(right_panel, text="统计信息", padding=10)
        stats_frame.pack(fill=tk.X)
        
        self._create_stats_display(stats_frame)
    
    def _create_camera_control(self, parent):
        """创建相机控制区"""
        # 连接状态
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(status_frame, text="相机状态:").pack(side=tk.LEFT)
        self.camera_status_label = ttk.Label(status_frame, text="未连接", 
                                            foreground="red", font=("Arial", 10, "bold"))
        self.camera_status_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # 按钮区
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="连接USB相机", 
                  command=lambda: self._connect_camera("usb")).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="连接大恒相机", 
                  command=lambda: self._connect_camera("daheng")).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="断开相机", 
                  command=self._disconnect_camera).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="开始预览", 
                  command=self._start_preview).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="停止预览", 
                  command=self._stop_preview).pack(fill=tk.X, pady=2)
    
    def _create_part_params(self, parent):
        """创建零件参数区"""
        # 零件编号
        ttk.Label(parent, text="零件编号:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.part_id_var, width=20).grid(row=0, column=1, pady=5)
        
        # 零件类型
        ttk.Label(parent, text="零件类型:").grid(row=1, column=0, sticky=tk.W, pady=5)
        part_type_combo = ttk.Combobox(parent, textvariable=self.part_type_var, 
                                       values=["圆形", "矩形"], state="readonly", width=17)
        part_type_combo.grid(row=1, column=1, pady=5)
        
        # 标称尺寸
        ttk.Label(parent, text="标称尺寸 (mm):").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.nominal_size_var, width=20).grid(row=2, column=1, pady=5)
        
        # 像素-毫米转换系数
        ttk.Label(parent, text="像素-毫米系数:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.pixel_to_mm_var, width=20).grid(row=3, column=1, pady=5)
    
    def _create_inspect_control(self, parent):
        """创建检测控制区"""
        ttk.Button(parent, text="单次检测", 
                  command=self._single_inspect, 
                  style="Accent.TButton").pack(fill=tk.X, pady=5)
        
        # 连续检测
        continuous_frame = ttk.Frame(parent)
        continuous_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(continuous_frame, text="间隔(秒):").pack(side=tk.LEFT)
        interval_var = tk.StringVar(value="1.0")
        ttk.Entry(continuous_frame, textvariable=interval_var, width=5).pack(side=tk.LEFT, padx=5)
        
        self.continue_inspect_btn = ttk.Button(continuous_frame, text="开始连续检测")
        self.continue_inspect_btn.pack(side=tk.LEFT, padx=5)
    
    def _create_result_display(self, parent):
        """创建结果显示区"""
        # 测量结果
        ttk.Label(parent, text="测量直径:").grid(row=0, column=0, sticky=tk.W, pady=3)
        ttk.Label(parent, textvariable=self.diameter_var, 
                 font=("Arial", 12, "bold")).grid(row=0, column=1, sticky=tk.W, pady=3)
        
        ttk.Label(parent, text="偏差:").grid(row=1, column=0, sticky=tk.W, pady=3)
        ttk.Label(parent, textvariable=self.deviation_var, 
                 font=("Arial", 12, "bold")).grid(row=1, column=1, sticky=tk.W, pady=3)
        
        ttk.Label(parent, text="公差:").grid(row=2, column=0, sticky=tk.W, pady=3)
        ttk.Label(parent, textvariable=self.tolerance_var, 
                 font=("Arial", 12)).grid(row=2, column=1, sticky=tk.W, pady=3)
        
        # 合格判定
        ttk.Separator(parent, orient='horizontal').grid(row=3, column=0, columnspan=2, 
                                                      sticky="ew", pady=10)
        
        result_label_frame = ttk.Frame(parent)
        result_label_frame.grid(row=4, column=0, columnspan=2, sticky="ew")
        
        ttk.Label(result_label_frame, text="检测结果:").pack(side=tk.LEFT)
        self.is_qualified_label = ttk.Label(result_label_frame, textvariable=self.is_qualified_var, 
                                           font=("Arial", 16, "bold"))
        self.is_qualified_label.pack(side=tk.LEFT, padx=(10, 0))
    
    def _create_stats_display(self, parent):
        """创建统计显示区"""
        ttk.Label(parent, text="合格数:").grid(row=0, column=0, sticky=tk.W, pady=3)
        ttk.Label(parent, textvariable=self.qualified_count_var, 
                 foreground="green", font=("Arial", 12, "bold")).grid(row=0, column=1, sticky=tk.W, pady=3)
        
        ttk.Label(parent, text="不合格数:").grid(row=1, column=0, sticky=tk.W, pady=3)
        ttk.Label(parent, textvariable=self.unqualified_count_var, 
                 foreground="red", font=("Arial", 12, "bold")).grid(row=1, column=1, sticky=tk.W, pady=3)
        
        ttk.Label(parent, text="合格率:").grid(row=2, column=0, sticky=tk.W, pady=3)
        ttk.Label(parent, textvariable=self.qualified_rate_var, 
                 font=("Arial", 12, "bold")).grid(row=2, column=1, sticky=tk.W, pady=3)
        
        ttk.Button(parent, text="刷新统计", command=self._update_stats).grid(row=3, column=0, 
                                                                         columnspan=2, pady=10, sticky="ew")
    
    def _update_camera_status(self):
        """更新相机状态显示"""
        if self.is_camera_connected.get():
            self.camera_status_label.config(text="已连接", foreground="green")
        else:
            self.camera_status_label.config(text="未连接", foreground="red")
    
    def _connect_camera(self, camera_type: str):
        """连接相机"""
        self._disconnect_camera()
        
        try:
            if camera_type.lower() == "usb":
                self.camera_driver = USBCameraDriver()
                device_id = simpledialog.askstring("USB相机", "请输入相机设备ID (默认为0):", 
                                                 initialvalue="0")
            elif camera_type.lower() == "daheng":
                self.camera_driver = DahengCameraDriver()
                device_id = None
            else:
                messagebox.showerror("错误", f"不支持的相机类型: {camera_type}")
                return
            
            if self.camera_driver.connect(device_id if device_id else None):
                self.is_camera_connected.set(True)
                self._update_camera_status()
                messagebox.showinfo("成功", "相机连接成功！")
            else:
                messagebox.showerror("错误", "相机连接失败！")
        except Exception as e:
            messagebox.showerror("错误", f"相机连接异常: {e}")
    
    def _disconnect_camera(self):
        """断开相机"""
        self._stop_preview()
        
        if self.camera_driver:
            self.camera_driver.disconnect()
            self.camera_driver = None
        
        self.is_camera_connected.set(False)
        self._update_camera_status()
        self.image_label.config(text="未连接相机", image="")
    
    def _start_preview(self):
        """开始预览"""
        if not self.is_camera_connected.get():
            messagebox.showwarning("警告", "请先连接相机！")
            return
        
        if self.is_previewing.get():
            messagebox.showwarning("警告", "已经在预览中！")
            return
        
        self.is_previewing.set(True)
        self.stop_preview = False
        
        self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self.preview_thread.start()
    
    def _stop_preview(self):
        """停止预览"""
        self.stop_preview = True
        self.is_previewing.set(False)
    
    def _preview_loop(self):
        """预览循环"""
        while not self.stop_preview and self.is_camera_connected.get():
            image = self.camera_driver.capture_image()
            if image is not None:
                self._display_image(image)
            
            time.sleep(0.03)  # ~30 FPS
    
    def _display_image(self, image: np.ndarray):
        """显示图像"""
        # 调整图像大小以适应显示区域
        h, w = image.shape[:2]
        max_size = 600
        
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        # 转换为RGB格式
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL Image
        pil_image = Image.fromarray(image)
        tk_image = ImageTk.PhotoImage(image=pil_image)
        
        # 更新标签
        self.image_label.config(image=tk_image, text="")
        self.image_label.image = tk_image  # 保持引用
    
    def _single_inspect(self):
        """单次检测"""
        if not self.is_camera_connected.get():
            messagebox.showwarning("警告", "请先连接相机！")
            return
        
        try:
            # 停止预览
            was_previewing = self.is_previewing.get()
            self._stop_preview()
            
            # 采集图像
            image = self.camera_driver.capture_image()
            if image is None:
                messagebox.showerror("错误", "图像采集失败！")
                return
            
            # 获取参数
            part_id = self.part_id_var.get().strip()
            part_type = self.part_type_var.get()
            
            try:
                nominal_size = float(self.nominal_size_var.get())
            except ValueError:
                nominal_size = None
            
            # 检测
            if part_type == "圆形":
                result = self.inspection_engine.detect_circle(
                    image, part_id, part_type, nominal_size
                )
            else:
                result = self.inspection_engine.detect_rectangle(
                    image, part_id, part_type, nominal_size
                )
            
            if result is None:
                messagebox.showerror("错误", "检测失败，未检测到零件！")
                return
            
            # 绘制结果
            result_image = self.inspection_engine.draw_result(image, result)
            
            # 保存图像
            result.image_path = self.data_manager.save_image(result_image, result)
            
            # 保存结果
            self.data_manager.save_result(result)
            
            # 更新显示
            self._update_result_display(result)
            self._update_stats()
            
            # 显示结果图像
            self._display_image(result_image)
            
            # 恢复预览
            if was_previewing:
                self._start_preview()
            
        except Exception as e:
            messagebox.showerror("错误", f"检测过程出错: {e}")
    
    def _update_result_display(self, result: InspectionResult):
        """更新结果显示"""
        if result.diameter_mm is not None:
            self.diameter_var.set(f"{result.diameter_mm:.3f} mm")
            
            if result.nominal_size is not None:
                self.deviation_var.set(f"{result.deviation:+.3f} mm")
                self.tolerance_var.set(f"±{result.tolerance:.3f} mm")
            else:
                self.deviation_var.set("--")
                self.tolerance_var.set("--")
        else:
            self.diameter_var.set("--")
            self.deviation_var.set("--")
            self.tolerance_var.set("--")
        
        # 更新合格判定
        if result.is_qualified:
            self.is_qualified_var.set("合格 ✓")
            self.is_qualified_label.config(foreground="green")
        else:
            self.is_qualified_var.set("不合格 ✗")
            self.is_qualified_label.config(foreground="red")
    
    def _update_stats(self):
        """更新统计信息"""
        stats = self.data_manager.get_statistics()
        
        self.qualified_count_var.set(str(stats.get('qualified', 0)))
        self.unqualified_count_var.set(str(stats.get('unqualified', 0)))
        self.qualified_rate_var.set(f"{stats.get('qualified_rate', 0)*100:.2f}%")
    
    def _show_calibration_dialog(self):
        """显示像素-毫米标定对话框"""
        dialog = CalibrationDialog(self.root, self)
        self.root.wait_window(dialog.dialog)
    
    def _show_camera_calibration_dialog(self):
        """显示相机内参标定对话框"""
        dialog = CameraCalibrationDialog(self.root, self.calibration)
        self.root.wait_window(dialog.dialog)
    
    def _export_data(self):
        """导出数据"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel文件", "*.xlsx"), ("所有文件", "*.*")]
        )
        
        if filename:
            if self.data_manager.export_to_excel(filename):
                messagebox.showinfo("成功", f"数据已导出到: {filename}")
            else:
                messagebox.showerror("错误", "数据导出失败！")
    
    def _show_about(self):
        """显示关于对话框"""
        about_text = """微小零件中高精度视觉检测系统 V1.0

精度目标: IT8级（亚像素检测精度≥1/20像素）
核心技术: OpenCV + 亚像素检测算法

开发者: 基于参考程序和需求规格说明书整合开发
创建日期: 2026-02-10

功能:
- 图像采集（支持USB相机和大恒相机）
- 相机标定（棋盘格标定）
- 亚像素边缘检测
- 圆形零件直径测量
- 矩形零件长宽测量
- IT8级公差判断
- 数据记录和导出
"""
        messagebox.showinfo("关于", about_text)
    
    def _on_close(self):
        """关闭窗口"""
        self._stop_preview()
        self._disconnect_camera()
        self.root.destroy()


# =============================================================================
# 标定对话框
# =============================================================================

class CalibrationDialog:
    """像素-毫米标定对话框"""
    
    def __init__(self, parent, system: InspectionSystemGUI):
        self.system = system
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("像素-毫米标定")
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self._create_layout()
    
    def _create_layout(self):
        """创建布局"""
        frame = ttk.Frame(self.dialog, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 说明
        ttk.Label(frame, text="像素-毫米标定", 
                 font=("Arial", 14, "bold")).pack(pady=(0, 20))
        
        info_text = """请按照以下步骤进行标定：

1. 将已知直径的标准件放入视野
2. 输入标准件的实际直径
3. 点击"开始标定"按钮
4. 系统将自动计算像素-毫米转换系数

注意：标定件应尽量充满视野，
以提高标定精度。"""
        
        ttk.Label(frame, text=info_text, justify=tk.LEFT).pack(pady=(0, 20))
        
        # 输入框
        input_frame = ttk.Frame(frame)
        input_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(input_frame, text="标准件直径 (mm):").pack(side=tk.LEFT)
        self.diameter_var = tk.StringVar(value="10.0")
        ttk.Entry(input_frame, textvariable=self.diameter_var, width=15).pack(side=tk.LEFT, padx=(10, 0))
        
        # 按钮
        ttk.Button(frame, text="开始标定", command=self._start_calibration).pack(fill=tk.X)
        ttk.Button(frame, text="关闭", command=self.dialog.destroy).pack(fill=tk.X, pady=(10, 0))
    
    def _start_calibration(self):
        """开始标定"""
        try:
            diameter = float(self.diameter_var.get())
        except ValueError:
            messagebox.showerror("错误", "请输入有效的直径值！")
            return
        
        if not self.system.is_camera_connected.get():
            messagebox.showwarning("警告", "请先连接相机！")
            return
        
        try:
            # 采集图像
            image = self.system.camera_driver.capture_image()
            if image is None:
                messagebox.showerror("错误", "图像采集失败！")
                return
            
            # 检测圆形
            result = self.system.inspection_engine.detect_circle(
                image, "标定件", "圆形", diameter
            )
            
            if result is None or result.diameter_pixel is None:
                messagebox.showerror("错误", "标定失败，未检测到圆形零件！")
                return
            
            # 计算像素-毫米转换系数
            pixel_to_mm = diameter / result.diameter_pixel
            self.system.config.PIXEL_TO_MM = pixel_to_mm
            self.system.pixel_to_mm_var.set(f"{pixel_to_mm:.6f}")
            
            # 保存配置
            self.system._save_config()
            
            messagebox.showinfo("成功", 
                              f"标定完成！\n\n"
                              f"实际直径: {diameter:.3f} mm\n"
                              f"测量直径: {result.diameter_pixel:.2f} 像素\n"
                              f"像素-毫米转换系数: {pixel_to_mm:.6f} mm/像素")
            
        except Exception as e:
            messagebox.showerror("错误", f"标定过程出错: {e}")


class CameraCalibrationDialog:
    """相机内参标定对话框"""
    
    def __init__(self, parent, calibration: CameraCalibration):
        self.calibration = calibration
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("相机内参标定")
        self.dialog.geometry("500x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self._create_layout()
    
    def _create_layout(self):
        """创建布局"""
        frame = ttk.Frame(self.dialog, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 说明
        ttk.Label(frame, text="相机内参标定", 
                 font=("Arial", 14, "bold")).pack(pady=(0, 20))
        
        info_text = """请按照以下步骤进行相机内参标定：

1. 准备一张棋盘格标定板（推荐6×4内角点）
2. 将标定图像放入 data/calib/ 目录
3. 点击"开始标定"按钮
4. 系统将自动计算相机内参和畸变系数

注意：
- 建议使用10-20张不同角度的标定图像
- 标定板应尽量充满视野
- 避免标定板倾斜过大
"""
        
        ttk.Label(frame, text=info_text, justify=tk.LEFT).pack(pady=(0, 20))
        
        # 按钮
        ttk.Button(frame, text="选择标定图像目录", 
                  command=self._select_calibration_dir).pack(fill=tk.X, pady=(0, 10))
        ttk.Button(frame, text="开始标定", 
                  command=self._start_calibration).pack(fill=tk.X, pady=(0, 10))
        ttk.Button(frame, text="关闭", 
                  command=self.dialog.destroy).pack(fill=tk.X)
    
    def _select_calibration_dir(self):
        """选择标定图像目录"""
        directory = filedialog.askdirectory(title="选择标定图像目录")
        if directory:
            self.calibration_dir = directory
    
    def _start_calibration(self):
        """开始标定"""
        if not hasattr(self, 'calibration_dir'):
            messagebox.showwarning("警告", "请先选择标定图像目录！")
            return
        
        try:
            if self.calibration.calibrate_from_images(self.calibration_dir):
                messagebox.showinfo("成功", 
                                  f"相机标定完成！\n\n"
                                  f"重投影误差: {self.calibration.reprojection_error:.4f} 像素\n"
                                  f"标定数据已保存到 data/calibration_data.json")
            else:
                messagebox.showerror("错误", "相机标定失败！")
        except Exception as e:
            messagebox.showerror("错误", f"标定过程出错: {e}")


# =============================================================================
# 导入对话框
# =============================================================================

from tkinter import simpledialog


# =============================================================================
# 主程序入口
# =============================================================================

def main():
    """主函数"""
    root = tk.Tk()
    
    # 设置主题
    style = ttk.Style()
    style.theme_use('clam')
    
    # 创建主窗口
    app = InspectionSystemGUI(root)
    
    # 关闭事件处理
    root.protocol("WM_DELETE_WINDOW", app._on_close)
    
    # 启动主循环
    root.mainloop()


if __name__ == "__main__":
    main()
