# -*- coding: utf-8 -*-
"""
DWG自动测量GUI
DWG Auto Measurement GUI

功能:
- DWG/DXF文件导入
- 图像配准可视化
- 自动测量和结果显示
- 测量报告生成
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
from datetime import datetime
from typing import Optional

from logger_config import get_logger
from auto_measurement import AutoMeasurementEngine, AutoMeasurementReport
from image_registration import ImageRegistration
from drawing_annotation import InspectionTemplate

logger = get_logger(__name__)


class DWGAutoMeasurementGUI:
    """DWG自动测量GUI"""
    
    def __init__(self, root: tk.Tk):
        """
        初始化GUI
        
        Args:
            root: Tkinter根窗口
        """
        self.root = root
        self.root.title("DWG自动测量系统")
        self.root.geometry("1200x800")
        
        # 初始化引擎
        self.measurement_engine = AutoMeasurementEngine()
        self.registration = ImageRegistration()
        
        # 数据
        self.dwg_file = None
        self.image_file = None
        self.template: Optional[InspectionTemplate] = None
        self.report: Optional[AutoMeasurementReport] = None
        self.template_image = None
        self.inspection_image = None
        
        # 创建界面
        self._create_widgets()
        
        logger.info("DWG自动测量GUI已启动")
    
    def _create_widgets(self):
        """创建界面组件"""
        # 主布局
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # ===== 左侧控制面板 =====
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # 文件选择
        ttk.Label(control_frame, text="DWG/DXF文件:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.dwg_file_entry = ttk.Entry(control_frame, width=40)
        self.dwg_file_entry.grid(row=1, column=0, pady=5)
        ttk.Button(control_frame, text="浏览...", command=self._select_dwg_file).grid(row=1, column=1, padx=5)
        
        ttk.Label(control_frame, text="图像文件:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.image_file_entry = ttk.Entry(control_frame, width=40)
        self.image_file_entry.grid(row=3, column=0, pady=5)
        ttk.Button(control_frame, text="浏览...", command=self._select_image_file).grid(row=3, column=1, padx=5)
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=15)
        
        # 配置选项
        ttk.Label(control_frame, text="配准方法:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.registration_method = ttk.Combobox(control_frame, values=['homography', 'affine', 'similarity'], state='readonly')
        self.registration_method.set('homography')
        self.registration_method.grid(row=6, column=0, pady=5)
        
        ttk.Label(control_frame, text="特征检测:").grid(row=7, column=0, sticky=tk.W, pady=5)
        self.feature_method = ttk.Combobox(control_frame, values=['ORB', 'SIFT', 'AKAZE'], state='readonly')
        self.feature_method.set('ORB')
        self.feature_method.grid(row=8, column=0, pady=5)
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=9, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=15)
        
        # 操作按钮
        ttk.Button(control_frame, text="1. 加载图纸", command=self._load_drawing).grid(row=10, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(control_frame, text="2. 加载图像", command=self._load_image).grid(row=11, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(control_frame, text="3. 图像配准", command=self._register_images).grid(row=12, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(control_frame, text="4. 自动测量", command=self._auto_measure).grid(row=13, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=14, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=15)
        
        ttk.Button(control_frame, text="导出报告", command=self._export_report).grid(row=15, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # 进度条
        ttk.Label(control_frame, text="进度:").grid(row=16, column=0, sticky=tk.W, pady=5)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=17, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # 状态
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(control_frame, textvariable=self.status_var, wraplength=300).grid(row=18, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # ===== 右侧显示区域 =====
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # 图像显示
        self.image_label = ttk.Label(display_frame, text="请加载图纸和图像", background='gray', anchor='center')
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # ===== 底部结果区域 =====
        result_frame = ttk.LabelFrame(main_frame, text="测量结果", padding="10")
        result_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        # 结果表格
        columns = ('特征ID', '特征类型', '测量值', '标称值', '公差', '偏差', '状态')
        self.result_tree = ttk.Treeview(result_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.result_tree.heading(col, text=col)
            self.result_tree.column(col, width=100, anchor='center')
        
        scrollbar = ttk.Scrollbar(result_frame, orient='vertical', command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=scrollbar.set)
        
        self.result_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
    
    def _select_dwg_file(self):
        """选择DWG/DXF文件"""
        file_path = filedialog.askopenfilename(
            title="选择DWG/DXF文件",
            filetypes=[("DWG/DXF文件", "*.dwg *.dxf"), ("所有文件", "*.*")]
        )
        if file_path:
            self.dwg_file_entry.delete(0, tk.END)
            self.dwg_file_entry.insert(0, file_path)
            self.dwg_file = file_path
    
    def _select_image_file(self):
        """选择图像文件"""
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        )
        if file_path:
            self.image_file_entry.delete(0, tk.END)
            self.image_file_entry.insert(0, file_path)
            self.image_file = file_path
    
    def _load_drawing(self):
        """加载图纸"""
        if not self.dwg_file:
            messagebox.showerror("错误", "请先选择DWG/DXF文件")
            return
        
        self._update_status("正在加载图纸...")
        self._update_progress(10)
        
        try:
            # 在新线程中加载
            thread = threading.Thread(target=self._load_drawing_thread)
            thread.daemon = True
            thread.start()
        except Exception as e:
            messagebox.showerror("错误", f"加载图纸失败: {str(e)}")
            self._update_status("就绪")
    
    def _load_drawing_thread(self):
        """加载图纸（线程）"""
        try:
            from dxf_parser import DXFParser
            
            # 如果是DWG文件，先转换
            if self.dwg_file.lower().endswith('.dwg'):
                self._update_status("正在转换DWG为DXF...")
                from dwg_converter import convert_dwg_to_dxf
                temp_dxf = os.path.splitext(self.dwg_file)[0] + '_temp.dxf'
                result = convert_dwg_to_dxf(self.dwg_file, temp_dxf)
                
                if not result.success:
                    self.root.after(0, lambda: messagebox.showerror("错误", f"DWG转换失败: {result.message}"))
                    self._update_status("就绪")
                    return
                
                dxf_file = temp_dxf
            else:
                dxf_file = self.dwg_file
            
            self._update_progress(30)
            
            # 解析DXF
            self._update_status("正在解析DXF...")
            parser = DXFParser()
            self.template = parser.parse_to_template(dxf_file)
            
            if self.template is None:
                self.root.after(0, lambda: messagebox.showerror("错误", "DXF解析失败"))
                self._update_status("就绪")
                return
            
            self._update_progress(60)
            
            # 渲染图纸
            self._update_status("正在渲染图纸...")
            self.template_image = self.measurement_engine._render_dxf_to_image(dxf_file)
            
            if self.template_image is None:
                self.root.after(0, lambda: messagebox.showerror("错误", "图纸渲染失败"))
                self._update_status("就绪")
                return
            
            self._update_progress(80)
            
            # 显示图纸
            self._display_image(self.template_image)
            
            self._update_progress(100)
            self._update_status(f"图纸加载成功，提取到 {len(self.template.annotations)} 个标注")
            
            # 清理临时文件
            try:
                if os.path.exists(temp_dxf):
                    os.remove(temp_dxf)
            except:
                pass
        
        except Exception as e:
            logger.error(f"加载图纸失败: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("错误", f"加载图纸失败: {str(e)}"))
            self._update_status("就绪")
    
    def _load_image(self):
        """加载图像"""
        if not self.image_file:
            messagebox.showerror("错误", "请先选择图像文件")
            return
        
        self._update_status("正在加载图像...")
        
        try:
            self.inspection_image = cv2.imread(self.image_file)
            
            if self.inspection_image is None:
                messagebox.showerror("错误", "无法加载图像")
                self._update_status("就绪")
                return
            
            self._display_image(self.inspection_image)
            self._update_status("图像加载成功")
        
        except Exception as e:
            messagebox.showerror("错误", f"加载图像失败: {str(e)}")
            self._update_status("就绪")
    
    def _register_images(self):
        """图像配准"""
        if self.template_image is None or self.inspection_image is None:
            messagebox.showerror("错误", "请先加载图纸和图像")
            return
        
        self._update_status("正在配准图像...")
        self._update_progress(10)
        
        try:
            # 在新线程中配准
            thread = threading.Thread(target=self._register_images_thread)
            thread.daemon = True
            thread.start()
        except Exception as e:
            messagebox.showerror("错误", f"图像配准失败: {str(e)}")
            self._update_status("就绪")
    
    def _register_images_thread(self):
        """图像配准（线程）"""
        try:
            # 转换为灰度图
            template_gray = cv2.cvtColor(self.template_image, cv2.COLOR_BGR2GRAY)
            image_gray = cv2.cvtColor(self.inspection_image, cv2.COLOR_BGR2GRAY)
            
            self._update_progress(30)
            
            # 设置特征检测方法
            method = self.feature_method.get()
            self.registration = ImageRegistration(method=method)
            
            # 执行配准
            self._update_status(f"正在使用{method}进行配准...")
            result = self.registration.register(image_gray, template_gray, method='homography')
            
            self._update_progress(70)
            
            if not result.success:
                self.root.after(0, lambda: messagebox.showerror("错误", f"配准失败: {result.message}"))
                self._update_status("就绪")
                return
            
            # 绘制匹配结果
            self._update_status("正在生成配准可视化...")
            matches_img = self.registration.draw_matches(self.inspection_image, self.template_image, result)
            
            self._update_progress(90)
            
            # 显示匹配结果
            self._display_image(matches_img)
            
            self._update_progress(100)
            self._update_status(f"配准成功，置信度: {result.transformation.confidence:.2f}")
            
            # 保存变换矩阵供后续使用
            self.transformation = result.transformation
        
        except Exception as e:
            logger.error(f"图像配准失败: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("错误", f"图像配准失败: {str(e)}"))
            self._update_status("就绪")
    
    def _auto_measure(self):
        """自动测量"""
        if self.dwg_file is None or self.image_file is None:
            messagebox.showerror("错误", "请先选择DWG/DXF文件和图像文件")
            return
        
        self._update_status("正在执行自动测量...")
        self._update_progress(10)
        
        try:
            # 在新线程中测量
            thread = threading.Thread(target=self._auto_measure_thread)
            thread.daemon = True
            thread.start()
        except Exception as e:
            messagebox.showerror("错误", f"自动测量失败: {str(e)}")
            self._update_status("就绪")
    
    def _auto_measure_thread(self):
        """自动测量（线程）"""
        try:
            # 执行测量
            self._update_status("正在转换DWG并提取标注...")
            self.report = self.measurement_engine.measure_from_dwg(
                self.dwg_file,
                self.image_file,
                output_dir="./reports"
            )
            
            self._update_progress(80)
            
            if not self.report.registration_success:
                self.root.after(0, lambda: messagebox.showerror("错误", "测量失败: 配准不成功"))
                self._update_status("就绪")
                return
            
            # 显示结果
            self._update_progress(90)
            self._display_results()
            
            self._update_progress(100)
            self._update_status(f"测量完成，合格: {self.report.passed_features}/{self.report.measured_features}")
        
        except Exception as e:
            logger.error(f"自动测量失败: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("错误", f"自动测量失败: {str(e)}"))
            self._update_status("就绪")
    
    def _display_results(self):
        """显示测量结果"""
        # 清空表格
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        
        # 添加结果
        for result in self.report.results:
            status = "合格" if result.is_passed else "不合格"
            status_color = "green" if result.is_passed else "red"
            
            self.result_tree.insert('', 'end', values=(
                result.annotation_id,
                result.feature_type,
                f"{result.measured_value:.3f}",
                f"{result.nominal_value:.3f}" if result.nominal_value else "N/A",
                f"±{result.tolerance:.3f}" if result.tolerance else "N/A",
                f"{result.deviation:+.3f}" if result.deviation else "N/A",
                status
            ), tags=(status,))
        
        # 设置颜色
        self.result_tree.tag_configure("green", foreground="green")
        self.result_tree.tag_configure("red", foreground="red")
    
    def _export_report(self):
        """导出报告"""
        if self.report is None:
            messagebox.showerror("错误", "没有可导出的报告")
            return
        
        try:
            # 选择输出目录
            output_dir = filedialog.askdirectory(title="选择输出目录")
            if not output_dir:
                return
            
            # 保存报告
            self.measurement_engine._save_report(self.report, output_dir)
            
            messagebox.showinfo("成功", f"报告已导出到: {output_dir}")
        
        except Exception as e:
            messagebox.showerror("错误", f"导出报告失败: {str(e)}")
    
    def _display_image(self, image: np.ndarray):
        """显示图像"""
        # 调整图像大小以适应显示区域
        max_width = 800
        max_height = 600
        
        h, w = image.shape[:2]
        
        if w > max_width or h > max_height:
            scale = min(max_width / w, max_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        # 转换为Tkinter图像
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(pil_image)
        
        # 更新显示
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo
    
    def _update_status(self, status: str):
        """更新状态"""
        self.status_var.set(status)
        self.root.update_idletasks()
    
    def _update_progress(self, value: float):
        """更新进度"""
        self.progress_var.set(value)
        self.root.update_idletasks()


def main():
    """主函数"""
    root = tk.Tk()
    app = DWGAutoMeasurementGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()