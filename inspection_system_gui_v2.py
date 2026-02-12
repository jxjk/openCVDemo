# -*- coding: utf-8 -*-
"""
微小零件中高精度视觉检测系统 - GUI V2版本
支持图纸标注和基于标注的检测

版本: V2.0
创建日期: 2026-02-10
新增功能: 图纸标注、基于标注的检测
"""

import os
import sys
import json
import uuid
import math
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from PIL import Image, ImageTk
import numpy as np
import cv2

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

# 导入图纸标注模块
from drawing_annotation import (
    InspectionTemplate,
    AnnotationTool,
    AnnotationBasedInspector,
    Point2D,
    BoundingBox,
    CircleAnnotation,
    LineAnnotation,
    RectangleAnnotation,
    AngleAnnotation,
    FeatureType,
    ToleranceStandard,
    create_default_template,
    load_template,
    save_template
)

# 导入DXF解析模块
from dxf_parser import (
    DXFParser,
    DXFToTemplateConverter,
    parse_dxf_file,
    dxf_to_template,
    extract_dimensions_from_dxf
)


# =============================================================================
# 图纸标注窗口
# =============================================================================

class DrawingAnnotationWindow:
    """图纸标注窗口"""
    
    def __init__(self, parent, template: InspectionTemplate = None):
        self.window = tk.Toplevel(parent)
        self.window.title("图纸标注工具")
        self.window.geometry("1400x900")
        self.window.transient(parent)
        self.window.grab_set()
        
        self.parent = parent
        self.template = template or create_default_template()
        self.annotation_tool = AnnotationTool(self.template)
        
        # 当前工具和状态
        self.current_tool = None
        self.drawing_points = []
        self.image = None
        self.image_scale = 1.0
        
        self._create_layout()
        self._create_menu()
    
    def _create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.window)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="加载图纸图像", command=self._load_image)
        file_menu.add_command(label="导入DXF文件", command=self._import_dxf)
        file_menu.add_command(label="加载模板", command=self._load_template)
        file_menu.add_command(label="保存模板", command=self._save_template)
        file_menu.add_separator()
        file_menu.add_command(label="关闭", command=self.window.destroy)
        menubar.add_cascade(label="文件", menu=file_menu)
        
        # 标注菜单
        anno_menu = tk.Menu(menubar, tearoff=0)
        anno_menu.add_command(label="标注直径", command=lambda: self._set_tool('diameter'))
        anno_menu.add_command(label="标注半径", command=lambda: self._set_tool('radius'))
        anno_menu.add_command(label="标注长度", command=lambda: self._set_tool('length'))
        anno_menu.add_command(label="标注宽度", command=lambda: self._set_tool('width'))
        anno_menu.add_command(label="标注高度", command=lambda: self._set_tool('height'))
        anno_menu.add_command(label="标注角度", command=lambda: self._set_tool('angle'))
        menubar.add_cascade(label="标注", menu=anno_menu)
        
        # 编辑菜单
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="编辑标注参数", command=self._edit_annotation)
        edit_menu.add_command(label="删除标注", command=self._delete_annotation)
        edit_menu.add_command(label="清空所有标注", command=self._clear_all_annotations)
        menubar.add_cascade(label="编辑", menu=edit_menu)
        
        self.window.config(menu=menubar)
    
    def _create_layout(self):
        """创建界面布局"""
        # 主框架
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧：图像显示区域
        left_frame = ttk.LabelFrame(main_frame, text="图纸图像", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas用于显示图像和绘制标注
        self.canvas = tk.Canvas(left_frame, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self._on_mouse_press)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
        
        # 右侧：工具和标注列表
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # 工具栏
        tool_frame = ttk.LabelFrame(right_frame, text="标注工具", padding=10)
        tool_frame.pack(fill=tk.X, pady=(0, 10))
        
        self._create_tool_buttons(tool_frame)
        
        # 标注列表
        list_frame = ttk.LabelFrame(right_frame, text="标注列表", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.annotation_tree = ttk.Treeview(list_frame, columns=('type', 'value'), show='headings', height=20)
        self.annotation_tree.heading('type', text='类型')
        self.annotation_tree.heading('value', text='标称值')
        self.annotation_tree.column('type', width=100)
        self.annotation_tree.column('value', width=100)
        self.annotation_tree.pack(fill=tk.BOTH, expand=True)
        
        # 绑定选择事件
        self.annotation_tree.bind("<<TreeviewSelect>>", self._on_annotation_select)
        
        # 操作按钮
        button_frame = ttk.Frame(list_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="编辑参数", command=self._edit_annotation).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="删除标注", command=self._delete_annotation).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="清空所有", command=self._clear_all_annotations).pack(fill=tk.X, pady=2)
    
    def _create_tool_buttons(self, parent):
        """创建工具按钮"""
        buttons = [
            ("标注直径", "diameter", "#4CAF50"),
            ("标注半径", "radius", "#2196F3"),
            ("标注长度", "length", "#FF9800"),
            ("标注宽度", "width", "#9C27B0"),
            ("标注高度", "height", "#E91E63"),
            ("标注角度", "angle", "#00BCD4")
        ]
        
        for text, tool, color in buttons:
            btn = tk.Button(parent, text=text, bg=color, fg='white',
                          command=lambda t=tool: self._set_tool(t))
            btn.pack(fill=tk.X, pady=2)
        
        # 缩放控制
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, pady=10)
        
        ttk.Label(parent, text="图像缩放:").pack(anchor=tk.W)
        self.scale_var = tk.DoubleVar(value=1.0)
        scale = ttk.Scale(parent, from_=0.1, to=3.0, variable=self.scale_var,
                         command=self._on_scale_change)
        scale.pack(fill=tk.X, pady=5)
        
        ttk.Button(parent, text="适应窗口", command=self._fit_to_window).pack(fill=tk.X, pady=2)
    
    def _load_image(self):
        """加载图纸图像"""
        filepath = filedialog.askopenfilename(
            title="选择图纸图像",
            filetypes=[("图像文件", "*.jpg *.png *.bmp *.tif"), ("所有文件", "*.*")]
        )
        
        if filepath:
            # 验证文件扩展名
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            file_ext = os.path.splitext(filepath)[1].lower()
            
            if file_ext not in valid_extensions:
                messagebox.showerror("错误", f"不支持的文件类型: {file_ext}\n\n请选择图像文件 (*.jpg, *.png, *.bmp, *.tif)")
                return
            
            self.image = cv2.imread(filepath)
            if self.image is not None:
                self._display_image()
                messagebox.showinfo("成功", "图纸图像加载成功！")
            else:
                messagebox.showerror("错误", "无法加载图像文件！")
    
    def _load_template(self):
        """加载模板"""
        filepath = filedialog.askopenfilename(
            title="选择模板文件",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if filepath:
            try:
                self.template = load_template(filepath)
                self.annotation_tool = AnnotationTool(self.template)
                self._update_annotation_list()
                messagebox.showinfo("成功", "模板加载成功！")
            except Exception as e:
                messagebox.showerror("错误", f"加载模板失败: {e}")
    
    def _import_dxf(self):
        """导入DXF文件"""
        filepath = filedialog.askopenfilename(
            title="选择DXF文件",
            filetypes=[("DXF文件", "*.dxf"), ("所有文件", "*.*")]
        )
        
        if filepath:
            try:
                # 询问导入选项
                import_dialog = tk.Toplevel(self.window)
                import_dialog.title("DXF导入选项")
                import_dialog.geometry("400x300")
                import_dialog.transient(self.window)
                import_dialog.grab_set()
                
                ttk.Label(import_dialog, text="导入选项", font=("Arial", 12, "bold")).pack(pady=10)
                
                # 自动提取尺寸标注
                auto_extract_var = tk.BooleanVar(value=True)
                ttk.Checkbutton(import_dialog, text="自动提取尺寸标注", 
                              variable=auto_extract_var).pack(pady=5)
                
                # 自动识别几何特征
                auto_identify_var = tk.BooleanVar(value=True)
                ttk.Checkbutton(import_dialog, text="自动识别几何特征", 
                              variable=auto_identify_var).pack(pady=5)
                
                # 公差标准
                ttk.Label(import_dialog, text="公差标准:").pack(pady=5)
                tolerance_var = tk.StringVar(value="IT8")
                tolerance_combo = ttk.Combobox(import_dialog, textvariable=tolerance_var,
                                              values=["IT5", "IT7", "IT8", "IT9", "IT11"])
                tolerance_combo.pack(pady=5)
                
                result = {'confirmed': False}
                
                def confirm_import():
                    try:
                        # 显示进度
                        progress = ttk.Progressbar(import_dialog, mode='indeterminate')
                        progress.pack(pady=10)
                        progress.start()
                        import_dialog.update()
                        
                        # 转换DXF为模板
                        template = dxf_to_template(
                            filepath,
                            template_name=os.path.basename(filepath),
                            tolerance_standard=tolerance_var.get(),
                            auto_extract_dimensions=auto_extract_var.get(),
                            auto_identify_features=auto_identify_var.get()
                        )
                        
                        progress.stop()
                        
                        if template:
                            self.template = template
                            self.annotation_tool = AnnotationTool(self.template)
                            self._update_annotation_list()
                            
                            # 渲染DXF为图像
                            self._render_dxf_to_image(filepath)
                            
                            result['confirmed'] = True
                            messagebox.showinfo("成功", 
                                             f"DXF文件导入成功！\n\n提取了 {len(template.annotations)} 个标注")
                            import_dialog.destroy()
                        else:
                            messagebox.showerror("错误", "DXF文件导入失败")
                    
                    except Exception as e:
                        progress.stop()
                        messagebox.showerror("错误", f"DXF文件导入失败: {e}")
                
                ttk.Button(import_dialog, text="导入", command=confirm_import).pack(pady=10)
                ttk.Button(import_dialog, text="取消", command=import_dialog.destroy).pack(pady=5)
                
                # 等待对话框关闭
                self.window.wait_window(import_dialog)
                
                return result.get('confirmed', False)
            
            except ImportError:
                messagebox.showerror("错误", 
                    "ezdxf库未安装，无法导入DXF文件。\n\n请运行: pip install ezdxf")
            except Exception as e:
                messagebox.showerror("错误", f"导入DXF文件失败: {e}")
    
    def _save_template(self):
        """保存模板"""
        filepath = filedialog.asksaveasfilename(
            title="保存模板",
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if filepath:
            try:
                save_template(self.template, filepath)
                messagebox.showinfo("成功", "模板保存成功！")
            except Exception as e:
                messagebox.showerror("错误", f"保存模板失败: {e}")
    
    def _set_tool(self, tool):
        """设置当前工具"""
        self.current_tool = tool
        self.drawing_points = []
        print(f"当前工具: {tool}")
    
    def _render_dxf_to_image(self, dxf_filepath):
        """将DXF文件渲染为图像"""
        try:
            import ezdxf
            
            # 读取DXF文件
            doc = ezdxf.readfile(dxf_filepath)
            msp = doc.modelspace()
            
            # 计算边界框
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')
            
            for entity in msp:
                if entity.dxftype() == 'CIRCLE':
                    center = entity.dxf.center
                    radius = entity.dxf.radius
                    min_x = min(min_x, center.x - radius)
                    min_y = min(min_y, center.y - radius)
                    max_x = max(max_x, center.x + radius)
                    max_y = max(max_y, center.y + radius)
                elif entity.dxftype() == 'LINE':
                    start = entity.dxf.start
                    end = entity.dxf.end
                    min_x = min(min_x, start[0], end[0])
                    min_y = min(min_y, start[1], end[1])
                    max_x = max(max_x, start[0], end[0])
                    max_y = max(max_y, start[1], end[1])
                elif entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
                    for point in entity.get_points():
                        min_x = min(min_x, point[0])
                        min_y = min(min_y, point[1])
                        max_x = max(max_x, point[0])
                        max_y = max(max_y, point[1])
            
            # 如果没有找到任何实体，使用默认尺寸
            if min_x == float('inf'):
                min_x, min_y = 0, 0
                max_x, max_y = 1000, 1000
            
            # 添加边距
            margin = 50
            width = int(max_x - min_x + 2 * margin)
            height = int(max_y - min_y + 2 * margin)
            
            # 创建白色背景图像
            self.image = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # 绘制所有实体
            for entity in msp:
                try:
                    if entity.dxftype() == 'CIRCLE':
                        center = entity.dxf.center
                        radius = entity.dxf.radius
                        center_x = int(center.x - min_x + margin)
                        center_y = int(center.y - min_y + margin)
                        radius_int = int(radius)
                        cv2.circle(self.image, (center_x, center_y), radius_int, (0, 0, 0), 2)
                    
                    elif entity.dxftype() == 'LINE':
                        start = entity.dxf.start
                        end = entity.dxf.end
                        start_pt = (int(start[0] - min_x + margin), int(start[1] - min_y + margin))
                        end_pt = (int(end[0] - min_x + margin), int(end[1] - min_y + margin))
                        cv2.line(self.image, start_pt, end_pt, (0, 0, 0), 2)
                    
                    elif entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
                        points = []
                        for point in entity.get_points():
                            pt = (int(point[0] - min_x + margin), int(point[1] - min_y + margin))
                            points.append(pt)
                        if len(points) >= 2:
                            points = np.array(points, dtype=np.int32)
                            cv2.polylines(self.image, [points], True, (0, 0, 0), 2)
                    
                    elif entity.dxftype() == 'ARC':
                        center = entity.dxf.center
                        radius = entity.dxf.radius
                        start_angle = np.radians(entity.dxf.start_angle)
                        end_angle = np.radians(entity.dxf.end_angle)
                        center_x = int(center.x - min_x + margin)
                        center_y = int(center.y - min_y + margin)
                        radius_int = int(radius)
                        cv2.ellipse(self.image, (center_x, center_y), 
                                   (radius_int, radius_int), 0, 
                                   -np.degrees(start_angle), -np.degrees(end_angle), 
                                   (0, 0, 0), 2)
                
                except Exception as e:
                    # 忽略单个实体的绘制错误
                    continue
            
            # 显示渲染的图像
            self._display_image()
            
        except ImportError:
            messagebox.showerror("错误", "ezdxf库未安装，无法渲染DXF文件")
        except Exception as e:
            messagebox.showerror("错误", f"渲染DXF文件失败: {e}")
    
    def _display_image(self):
        """显示图像"""
        if self.image is None:
            return
        
        # 转换为RGB格式
        if len(self.image.shape) == 3:
            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        
        # 应用缩放
        scale = self.scale_var.get()
        if scale != 1.0:
            height, width = image_rgb.shape[:2]
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_rgb = cv2.resize(image_rgb, (new_width, new_height))
        
        # 转换为PIL Image
        pil_image = Image.fromarray(image_rgb)
        self.tk_image = ImageTk.PhotoImage(image=pil_image)
        
        # 在Canvas上显示
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        # 保存当前缩放比例
        self.image_scale = scale
        
        # 重绘所有标注
        self._redraw_annotations()
    
    def _redraw_annotations(self):
        """重绘所有标注"""
        for annotation in self.template.annotations:
            self._draw_annotation(annotation)
    
    def _draw_annotation(self, annotation):
        """绘制单个标注"""
        scale = self.image_scale
        
        if isinstance(annotation, CircleAnnotation):
            center_x = annotation.center.x * scale
            center_y = annotation.center.y * scale
            radius = annotation.radius * scale
            
            color = "green" if annotation.feature_type == FeatureType.DIAMETER else "blue"
            self.canvas.create_oval(
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius,
                outline=color, width=2, tags=f"anno_{annotation.id}"
            )
            self.canvas.create_oval(
                center_x - 3, center_y - 3,
                center_x + 3, center_y + 3,
                fill=color, tags=f"anno_{annotation.id}"
            )
        
        elif isinstance(annotation, LineAnnotation):
            start_x = annotation.start.x * scale
            start_y = annotation.start.y * scale
            end_x = annotation.end.x * scale
            end_y = annotation.end.y * scale
            
            color = "red" if annotation.feature_type == FeatureType.LENGTH else "orange"
            self.canvas.create_line(
                start_x, start_y, end_x, end_y,
                fill=color, width=2, tags=f"anno_{annotation.id}"
            )
            self.canvas.create_oval(
                start_x - 3, start_y - 3,
                start_x + 3, start_y + 3,
                fill=color, tags=f"anno_{annotation.id}"
            )
            self.canvas.create_oval(
                end_x - 3, end_y - 3,
                end_x + 3, end_y + 3,
                fill=color, tags=f"anno_{annotation.id}"
            )
        
        elif isinstance(annotation, RectangleAnnotation):
            x = annotation.bbox.x * scale
            y = annotation.bbox.y * scale
            width = annotation.bbox.width * scale
            height = annotation.bbox.height * scale
            
            color = "orange" if annotation.feature_type == FeatureType.WIDTH else "pink"
            self.canvas.create_rectangle(
                x, y, x + width, y + height,
                outline=color, width=2, tags=f"anno_{annotation.id}"
            )
    
    def _on_mouse_press(self, event):
        """鼠标按下事件"""
        if self.image is None:
            messagebox.showwarning("警告", "请先加载图纸图像！")
            return
        
        # 转换为图像坐标
        x = event.x / self.image_scale
        y = event.y / self.image_scale
        
        self.drawing_points.append(Point2D(x, y))
        
        print(f"添加点: ({x:.2f}, {y:.2f}), 当前点数: {len(self.drawing_points)}")
        
        # 根据工具类型处理
        if self.current_tool in ['diameter', 'radius']:
            # 圆形标注需要2个点：圆心和边缘
            if len(self.drawing_points) == 2:
                self._add_circle_annotation()
        
        elif self.current_tool in ['length', 'width', 'height']:
            # 线段标注需要2个点：起点和终点
            if len(self.drawing_points) == 2:
                self._add_line_annotation()
        
        elif self.current_tool == 'angle':
            # 角度标注需要3个点：顶点、起点、终点
            if len(self.drawing_points) == 3:
                self._add_angle_annotation()
    
    def _on_mouse_drag(self, event):
        """鼠标拖动事件"""
        pass
    
    def _on_mouse_release(self, event):
        """鼠标释放事件"""
        pass
    
    def _add_circle_annotation(self):
        """添加圆形标注"""
        center = self.drawing_points[0]
        edge = self.drawing_points[1]
        
        radius = center.distance_to(edge)
        
        feature_type = FeatureType.DIAMETER if self.current_tool == 'diameter' else FeatureType.RADIUS
        
        annotation = self.annotation_tool.add_circle(center, radius, feature_type)
        
        # 询问标称值
        nominal_value = simpledialog.askfloat("参数设置", 
                                             f"请输入标称{'直径' if self.current_tool == 'diameter' else '半径'} (mm):",
                                             minvalue=0.0)
        
        if nominal_value is not None:
            self.annotation_tool.update_annotation_value(annotation.id, nominal_value)
        
        self._update_annotation_list()
        self._redraw_annotations()
        self.drawing_points = []
    
    def _add_line_annotation(self):
        """添加线段标注"""
        start = self.drawing_points[0]
        end = self.drawing_points[1]
        
        if self.current_tool == 'length':
            feature_type = FeatureType.LENGTH
        elif self.current_tool == 'width':
            feature_type = FeatureType.WIDTH
        else:  # height
            feature_type = FeatureType.HEIGHT
        
        annotation = self.annotation_tool.add_line(start, end, feature_type)
        
        # 询问标称值
        nominal_value = simpledialog.askfloat("参数设置",
                                             f"请输入标称{'长度' if self.current_tool == 'length' else '宽度' if self.current_tool == 'width' else '高度'} (mm):",
                                             minvalue=0.0)
        
        if nominal_value is not None:
            self.annotation_tool.update_annotation_value(annotation.id, nominal_value)
        
        self._update_annotation_list()
        self._redraw_annotations()
        self.drawing_points = []
    
    def _add_angle_annotation(self):
        """添加角度标注"""
        vertex = self.drawing_points[0]
        start_point = self.drawing_points[1]
        end_point = self.drawing_points[2]
        
        annotation = self.annotation_tool.add_angle(vertex, start_point, end_point)
        
        # 询问标称值
        nominal_value = simpledialog.askfloat("参数设置",
                                             "请输入标称角度 (度):",
                                             minvalue=0.0, maxvalue=360.0)
        
        if nominal_value is not None:
            self.annotation_tool.update_annotation_value(annotation.id, nominal_value)
        
        self._update_annotation_list()
        self._redraw_annotations()
        self.drawing_points = []
    
    def _update_annotation_list(self):
        """更新标注列表"""
        # 清空列表
        for item in self.annotation_tree.get_children():
            self.annotation_tree.delete(item)
        
        # 添加标注
        for i, annotation in enumerate(self.template.annotations):
            self.annotation_tree.insert('', 'end', values=(
                annotation.feature_type.value,
                f"{annotation.nominal_value:.3f} mm" if annotation.nominal_value else "--"
            ))
    
    def _on_annotation_select(self, event):
        """标注选择事件"""
        selection = self.annotation_tree.selection()
        if selection:
            index = self.annotation_tree.index(selection[0])
            if index < len(self.template.annotations):
                annotation = self.template.annotations[index]
                print(f"选中标注: {annotation.id}, 类型: {annotation.feature_type.value}")
    
    def _edit_annotation(self):
        """编辑标注参数"""
        selection = self.annotation_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请先选择要编辑的标注！")
            return
        
        index = self.annotation_tree.index(selection[0])
        if index >= len(self.template.annotations):
            return
        
        annotation = self.template.annotations[index]
        
        # 创建编辑对话框
        dialog = tk.Toplevel(self.window)
        dialog.title("编辑标注参数")
        dialog.geometry("400x300")
        
        ttk.Label(dialog, text=f"标注类型: {annotation.feature_type.value}").pack(pady=10)
        
        # 标称值
        ttk.Label(dialog, text="标称值 (mm):").pack()
        nominal_var = tk.DoubleVar(value=annotation.nominal_value or 0.0)
        ttk.Entry(dialog, textvariable=nominal_var).pack(pady=5)
        
        # 公差
        ttk.Label(dialog, text="公差 (mm):").pack()
        tolerance_var = tk.DoubleVar(value=annotation.tolerance or 0.0)
        ttk.Entry(dialog, textvariable=tolerance_var).pack(pady=5)
        
        # 描述
        ttk.Label(dialog, text="描述:").pack()
        desc_var = tk.StringVar(value=annotation.description)
        ttk.Entry(dialog, textvariable=desc_var).pack(pady=5)
        
        def save_changes():
            try:
                self.annotation_tool.update_annotation_value(
                    annotation.id,
                    nominal_var.get(),
                    tolerance_var.get()
                )
                annotation.description = desc_var.get()
                self._update_annotation_list()
                dialog.destroy()
                messagebox.showinfo("成功", "参数已更新！")
            except ValueError:
                messagebox.showerror("错误", "请输入有效的数值！")
        
        ttk.Button(dialog, text="保存", command=save_changes).pack(pady=20)
    
    def _delete_annotation(self):
        """删除标注"""
        selection = self.annotation_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请先选择要删除的标注！")
            return
        
        index = self.annotation_tree.index(selection[0])
        if index < len(self.template.annotations):
            annotation = self.template.annotations[index]
            self.annotation_tool.remove_annotation(annotation.id)
            self._update_annotation_list()
            self._redraw_annotations()
    
    def _clear_all_annotations(self):
        """清空所有标注"""
        if messagebox.askyesno("确认", "确定要清空所有标注吗？"):
            self.template.annotations.clear()
            self._update_annotation_list()
            self._redraw_annotations()
    
    def _on_scale_change(self, value):
        """缩放变化"""
        self._display_image()
    
    def _fit_to_window(self):
        """适应窗口"""
        if self.image is None:
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_height, image_width = self.image.shape[:2]
        
        scale_x = canvas_width / image_width
        scale_y = canvas_height / image_height
        scale = min(scale_x, scale_y) * 0.9
        
        self.scale_var.set(scale)
        self._display_image()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    
    # 创建主窗口（简化版本，只演示标注功能）
    app = tk.Tk()
    app.title("图纸标注工具演示")
    app.geometry("1400x900")
    
    # 创建标注窗口
    annotation_window = DrawingAnnotationWindow(app)
    
    app.mainloop()