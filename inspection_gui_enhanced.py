# -*- coding: utf-8 -*-
"""
增强版GUI - 集成批量检测和缺陷检测
Enhanced GUI with Batch Inspection and Defect Detection

版本: V3.0
创建日期: 2026-02-10
新增功能:
- 批量检测控制面板
- 实时预览和结果显示
- 声光报警功能
- 历史数据查询
- 统计报表
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
from PIL import Image, ImageTk
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
from matplotlib.figure import Figure

# 设置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入核心模块
from inspection_system import (
    InspectionConfig,
    InspectionEngine,
    InspectionResult,
    DataManager
)
from batch_inspection import (
    BatchInspectionEngine,
    BatchInspectionConfig,
    CameraBatchAcquisition,
    InspectionStatus
)
from defect_detection import (
    ComprehensiveDefectDetector,
    DefectDetectionResult,
    DefectType
)
from data_export import DataExporter, StatisticsCalculator
from logger_config import get_logger
from exceptions import CameraException

# 导入GUI组件
from inspection_system_gui_v2 import DrawingAnnotationWindow
from calibration_gui import CalibrationGUI
from pixel_mm_calibration_gui import PixelMmCalibrationGUI
from camera_preview_gui import CameraPreviewGUI


# =============================================================================
# 声光报警控制器
# =============================================================================

class AlarmController:
    """声光报警控制器"""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._alarm_enabled = True
        self._current_alarm = None
        self._alarm_thread = None
        self._stop_alarm = threading.Event()
    
    def enable_alarm(self, enabled: bool):
        """启用/禁用报警"""
        self._alarm_enabled = enabled
        self.logger.info(f"报警已{'启用' if enabled else '禁用'}")
    
    def trigger_alarm(self, alarm_type: str = "NG", duration: float = 2.0):
        """
        触发报警
        
        Args:
            alarm_type: 报警类型 (NG, ERROR, WARNING)
            duration: 持续时间（秒）
        """
        if not self._alarm_enabled:
            return
        
        self._stop_alarm.set()  # 停止当前报警
        self._stop_alarm.clear()
        
        self._current_alarm = alarm_type
        self.logger.warning(f"触发报警: {alarm_type}")
        
        # 在实际系统中，这里会连接硬件报警设备
        # 例如：通过串口控制LED灯和蜂鸣器
        self._play_sound(alarm_type, duration)
    
    def stop_alarm(self):
        """停止报警"""
        self._stop_alarm.set()
        self._current_alarm = None
        self.logger.info("报警已停止")
    
    def _play_sound(self, alarm_type: str, duration: float):
        """播放报警声音"""
        try:
            # 使用winsound播放系统声音（Windows）
            import winsound
            
            def _sound_worker():
                start_time = time.time()
                while time.time() - start_time < duration and not self._stop_alarm.is_set():
                    if alarm_type == "NG":
                        # 连续 beep
                        winsound.Beep(1000, 200)
                    elif alarm_type == "ERROR":
                        # 高频 beep
                        winsound.Beep(2000, 100)
                    elif alarm_type == "WARNING":
                        # 低频 beep
                        winsound.Beep(500, 300)
                    time.sleep(0.3)
            
            self._alarm_thread = threading.Thread(target=_sound_worker, daemon=True)
            self._alarm_thread.start()
        
        except ImportError:
            # 如果winsound不可用，使用print模拟
            self.logger.warning("winsound不可用，报警声音已禁用")
        except Exception as e:
            self.logger.error(f"播放报警声音失败: {e}")


# =============================================================================
# 历史数据查询窗口
# =============================================================================

class HistoryQueryWindow:
    """历史数据查询窗口"""
    
    def __init__(self, parent, data_exporter: DataExporter):
        self.window = tk.Toplevel(parent)
        self.window.title("历史数据查询")
        self.window.geometry("1000x700")
        self.window.transient(parent)
        self.window.grab_set()
        
        self.parent = parent
        self.data_exporter = data_exporter
        self.logger = get_logger(self.__class__.__name__)
        
        self._create_layout()
        self._load_data()
    
    def _create_layout(self):
        """创建界面布局"""
        # 主框架
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 查询条件框架
        query_frame = ttk.LabelFrame(main_frame, text="查询条件", padding=10)
        query_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 第一行：时间范围
        ttk.Label(query_frame, text="起始时间:").grid(row=0, column=0, sticky=tk.W)
        self.start_date_var = tk.StringVar()
        self.start_date_entry = ttk.Entry(query_frame, textvariable=self.start_date_var, width=15)
        self.start_date_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(query_frame, text="结束时间:").grid(row=0, column=2, sticky=tk.W)
        self.end_date_var = tk.StringVar()
        self.end_date_entry = ttk.Entry(query_frame, textvariable=self.end_date_var, width=15)
        self.end_date_entry.grid(row=0, column=3, padx=5)
        
        # 第二行：零件类型和检测结果
        ttk.Label(query_frame, text="零件类型:").grid(row=1, column=0, sticky=tk.W)
        self.part_type_var = tk.StringVar()
        self.part_type_combo = ttk.Combobox(query_frame, textvariable=self.part_type_var, width=15)
        self.part_type_combo['values'] = ('全部', '圆形', '矩形')
        self.part_type_combo.current(0)
        self.part_type_combo.grid(row=1, column=1, padx=5)
        
        ttk.Label(query_frame, text="检测结果:").grid(row=1, column=2, sticky=tk.W)
        self.result_var = tk.StringVar()
        self.result_combo = ttk.Combobox(query_frame, textvariable=self.result_var, width=15)
        self.result_combo['values'] = ('全部', '合格', '不合格')
        self.result_combo.current(0)
        self.result_combo.grid(row=1, column=3, padx=5)
        
        # 查询按钮
        button_frame = ttk.Frame(query_frame)
        button_frame.grid(row=2, column=0, columnspan=4, pady=10)
        
        ttk.Button(button_frame, text="查询", command=self._query_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="导出", command=self._export_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清空", command=self._clear_query).pack(side=tk.LEFT, padx=5)
        
        # 结果显示框架
        result_frame = ttk.LabelFrame(main_frame, text="查询结果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview显示数据
        columns = ('timestamp', 'part_id', 'part_type', 'measured_value', 'nominal_value', 'is_passed')
        self.tree = ttk.Treeview(result_frame, columns=columns, show='headings', height=20)
        
        self.tree.heading('timestamp', text='检测时间')
        self.tree.heading('part_id', text='零件编号')
        self.tree.heading('part_type', text='零件类型')
        self.tree.heading('measured_value', text='实测值')
        self.tree.heading('nominal_value', text='标称值')
        self.tree.heading('is_passed', text='结果')
        
        self.tree.column('timestamp', width=150)
        self.tree.column('part_id', width=120)
        self.tree.column('part_type', width=80)
        self.tree.column('measured_value', width=100)
        self.tree.column('nominal_value', width=100)
        self.tree.column('is_passed', width=60)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 统计信息
        self.stats_label = ttk.Label(main_frame, text="")
        self.stats_label.pack(fill=tk.X, pady=10)
    
    def _load_data(self):
        """加载历史数据"""
        try:
            # 读取检测结果文件
            result_file = os.path.join("data", "inspection_results.csv")
            if os.path.exists(result_file):
                self.df = pd.read_csv(result_file)
                self.logger.info(f"已加载 {len(self.df)} 条历史记录")
            else:
                self.df = pd.DataFrame()
                self.logger.warning("历史数据文件不存在")
        
        except Exception as e:
            self.logger.error(f"加载历史数据失败: {e}")
            self.df = pd.DataFrame()
    
    def _query_data(self):
        """执行查询"""
        if self.df.empty:
            messagebox.showinfo("提示", "没有历史数据")
            return
        
        # 复制数据
        filtered_df = self.df.copy()
        
        # 时间范围过滤
        start_date = self.start_date_var.get()
        end_date = self.end_date_var.get()
        
        if start_date:
            filtered_df = filtered_df[filtered_df['timestamp'] >= start_date]
        if end_date:
            filtered_df = filtered_df[filtered_df['timestamp'] <= end_date]
        
        # 零件类型过滤
        part_type = self.part_type_var.get()
        if part_type != '全部':
            filtered_df = filtered_df[filtered_df['part_type'] == part_type]
        
        # 检测结果过滤
        result = self.result_var.get()
        if result == '合格':
            filtered_df = filtered_df[filtered_df['is_passed'] == True]
        elif result == '不合格':
            filtered_df = filtered_df[filtered_df['is_passed'] == False]
        
        # 清空Treeview
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # 填充数据
        for _, row in filtered_df.iterrows():
            values = (
                row.get('timestamp', ''),
                row.get('part_id', ''),
                row.get('part_type', ''),
                f"{row.get('measured_value', 0):.3f}",
                f"{row.get('nominal_value', 0):.3f}",
                '合格' if row.get('is_passed', False) else '不合格'
            )
            self.tree.insert('', tk.END, values=values)
        
        # 更新统计信息
        total = len(filtered_df)
        passed = len(filtered_df[filtered_df['is_passed'] == True])
        failed = total - passed
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        self.stats_label.config(
            text=f"总计: {total} 条 | 合格: {passed} 条 | 不合格: {failed} 条 | 合格率: {pass_rate:.2f}%"
        )
    
    def _export_data(self):
        """导出查询结果"""
        try:
            # 收集当前显示的数据
            data = []
            for item in self.tree.get_children():
                values = self.tree.item(item)['values']
                data.append({
                    'timestamp': values[0],
                    'part_id': values[1],
                    'part_type': values[2],
                    'measured_value': values[3],
                    'nominal_value': values[4],
                    'is_passed': values[5]
                })
            
            if not data:
                messagebox.showinfo("提示", "没有数据可导出")
                return
            
            # 选择文件路径
            filename = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel文件", "*.xlsx"), ("CSV文件", "*.csv")]
            )
            
            if filename:
                df_export = pd.DataFrame(data)
                if filename.endswith('.xlsx'):
                    df_export.to_excel(filename, index=False)
                else:
                    df_export.to_csv(filename, index=False, encoding='utf-8-sig')
                
                messagebox.showinfo("成功", f"数据已导出到: {filename}")
        
        except Exception as e:
            self.logger.error(f"导出数据失败: {e}")
            messagebox.showerror("错误", f"导出失败: {e}")
    
    def _clear_query(self):
        """清空查询条件"""
        self.start_date_var.set("")
        self.end_date_var.set("")
        self.part_type_combo.current(0)
        self.result_combo.current(0)
        self._query_data()


# =============================================================================
# 统计报表窗口
# ============================================================================

class StatisticsReportWindow:
    """统计报表窗口"""
    
    def __init__(self, parent, data_exporter: DataExporter):
        self.window = tk.Toplevel(parent)
        self.window.title("统计报表")
        self.window.geometry("1200x800")
        self.window.transient(parent)
        self.window.grab_set()
        
        self.parent = parent
        self.data_exporter = data_exporter
        self.logger = get_logger(self.__class__.__name__)
        
        self._create_layout()
        self._generate_report()
    
    def _create_layout(self):
        """创建界面布局"""
        # 主框架
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 工具栏
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(toolbar, text="刷新报表", command=self._generate_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="导出报表", command=self._export_report).pack(side=tk.LEFT, padx=5)
        
        # 统计信息框架
        stats_frame = ttk.LabelFrame(main_frame, text="统计摘要", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=6, width=100)
        self.stats_text.pack(fill=tk.X)
        
        # 图表框架
        chart_frame = ttk.LabelFrame(main_frame, text="图表", padding=10)
        chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建matplotlib图形
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _generate_report(self):
        """生成统计报表"""
        try:
            # 读取数据
            result_file = os.path.join("data", "inspection_results.csv")
            if not os.path.exists(result_file):
                messagebox.showinfo("提示", "没有数据可分析")
                return
            
            df = pd.read_csv(result_file)
            
            if df.empty:
                messagebox.showinfo("提示", "数据为空")
                return
            
            # 计算统计数据
            stats = self.data_exporter.calculate_statistics(df.to_dict('records'))
            
            # 显示统计摘要
            summary = stats.get('summary', {})
            
            if 'error' in summary:
                stats_text = f"统计计算失败: {summary['error']}"
            else:
                stats_text = f"""
检测总数: {summary.get('总检测数', 0)}
合格数: {summary.get('合格数', 0)}
不合格数: {summary.get('不合格数', 0)}
合格率: {summary.get('合格率', 0.0):.2f}%
导出时间: {summary.get('导出时间', 'N/A')}
"""
            
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, stats_text)
            
            # 绘制图表（仅当有数据时）
            if not df.empty and 'error' not in summary:
                self._draw_charts(df)
        
        except Exception as e:
            self.logger.error(f"生成报表失败: {e}")
            messagebox.showerror("错误", f"生成报表失败: {e}")
    
    def _draw_charts(self, df: pd.DataFrame):
        """绘制图表"""
        self.figure.clear()
        
        # 创建子图
        ax1 = self.figure.add_subplot(221)
        ax2 = self.figure.add_subplot(222)
        ax3 = self.figure.add_subplot(223)
        ax4 = self.figure.add_subplot(224)
        
        # 图1: 合格率饼图
        passed = len(df[df['is_passed'] == True])
        failed = len(df[df['is_passed'] == False])
        ax1.pie([passed, failed], labels=['合格', '不合格'], autopct='%1.1f%%', 
                colors=['#4CAF50', '#F44336'])
        ax1.set_title('合格率')
        
        # 图2: 零件类型分布
        if 'part_type' in df.columns:
            part_types = df['part_type'].value_counts()
            ax2.bar(part_types.index, part_types.values, color='#2196F3')
            ax2.set_title('零件类型分布')
            ax2.set_ylabel('数量')
        
        # 图3: 检测结果时间趋势
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
            df_hourly = df.groupby(df['datetime'].dt.hour).size()
            ax3.plot(df_hourly.index, df_hourly.values, marker='o', color='#FF9800')
            ax3.set_title('检测数量时间趋势')
            ax3.set_xlabel('小时')
            ax3.set_ylabel('数量')
        
        # 图4: 测量值分布
        if 'measured_value' in df.columns:
            ax4.hist(df['measured_value'], bins=20, color='#9C27B0', alpha=0.7)
            ax4.set_title('测量值分布')
            ax4.set_xlabel('测量值')
            ax4.set_ylabel('频次')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def _export_report(self):
        """导出报表"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel文件", "*.xlsx")]
            )
            
            if filename:
                result_file = os.path.join("data", "inspection_results.csv")
                df = pd.read_csv(result_file)
                
                # 导出统计报告
                self.data_exporter.export_statistics(df.to_dict('records'), filename)
                
                messagebox.showinfo("成功", f"报表已导出到: {filename}")
        
        except Exception as e:
            self.logger.error(f"导出报表失败: {e}")
            messagebox.showerror("错误", f"导出失败: {e}")


# =============================================================================
# 批量检测控制面板
# ============================================================================

class BatchInspectionPanel:
    """批量检测控制面板"""
    
    def __init__(self, parent, batch_engine: BatchInspectionEngine,
                 alarm_controller: AlarmController):
        self.parent = parent
        self.batch_engine = batch_engine
        self.alarm = alarm_controller
        self.logger = get_logger(self.__class__.__name__)
        
        self.frame = ttk.LabelFrame(parent, text="批量检测控制", padding=10)
        
        self._create_layout()
        self._setup_callbacks()
    
    def _create_layout(self):
        """创建界面布局"""
        # 控制按钮
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.start_btn = ttk.Button(button_frame, text="启动", command=self._start_batch)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.pause_btn = ttk.Button(button_frame, text="暂停", command=self._pause_batch, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="停止", command=self._stop_batch, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # 状态显示
        status_frame = ttk.Frame(self.frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(status_frame, text="状态:").pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="空闲")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                     foreground="blue", font=('Arial', 10, 'bold'))
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # 统计信息
        stats_frame = ttk.Frame(self.frame)
        stats_frame.pack(fill=tk.X, pady=5)
        
        self.stats_var = tk.StringVar(value="已检测: 0 | 合格: 0 | 不合格: 0 | 速度: 0件/分")
        ttk.Label(stats_frame, textvariable=self.stats_var).pack(anchor=tk.W)
        
        # 进度条
        self.progress = ttk.Progressbar(self.frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
    
    def _setup_callbacks(self):
        """设置回调函数"""
        self.batch_engine.set_progress_callback(self._on_progress)
        self.batch_engine.set_result_callback(self._on_result)
    
    def _start_batch(self):
        """启动批量检测"""
        if self.batch_engine.start():
            self.start_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_var.set("运行中")
            self.status_label.config(foreground="green")
            self.progress.start()
            self.logger.info("批量检测已启动")
    
    def _pause_batch(self):
        """暂停批量检测"""
        if self.batch_engine.pause():
            self.pause_btn.config(text="恢复")
            self.status_var.set("已暂停")
            self.status_label.config(foreground="orange")
            self.progress.stop()
            self.logger.info("批量检测已暂停")
        elif self.batch_engine.resume():
            self.pause_btn.config(text="暂停")
            self.status_var.set("运行中")
            self.status_label.config(foreground="green")
            self.progress.start()
            self.logger.info("批量检测已恢复")
    
    def _stop_batch(self):
        """停止批量检测"""
        self.batch_engine.stop()
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="暂停")
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("已停止")
        self.status_label.config(foreground="red")
        self.progress.stop()
        self.logger.info("批量检测已停止")
    
    def _on_progress(self, stats: dict):
        """进度回调"""
        stats_text = (f"已检测: {stats['completed_tasks']} | "
                     f"合格: {stats['passed_tasks']} | "
                     f"不合格: {stats['failed_tasks']} | "
                     f"速度: {stats['current_speed']:.1f}件/分")
        self.stats_var.set(stats_text)
    
    def _on_result(self, result):
        """结果回调"""
        # 如果不合格，触发报警
        if not result.is_passed and not result.error:
            self.alarm.trigger_alarm("NG")
    
    def pack(self, **kwargs):
        """打包"""
        self.frame.pack(**kwargs)


# =============================================================================
# 完整增强版GUI
# ============================================================================

class InspectionSystemEnhancedGUI:
    """
    增强版检测系统GUI
    
    整合所有功能：
    - 图像采集和预览
    - 单次检测
    - 批量检测
    - 缺陷检测
    - 声光报警
    - 历史查询
    - 统计报表
    - 图纸标注
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("微小零件中高精度视觉检测系统 V3.0")
        self.root.geometry("1400x900")
        
        # 初始化日志
        self.logger = get_logger(self.__class__.__name__)
        
        # 初始化配置
        self.config = InspectionConfig()
        
        # 初始化核心引擎
        self.inspection_engine = InspectionEngine(self.config)
        self.defect_detector = ComprehensiveDefectDetector()
        
        # 初始化批量检测引擎
        batch_config = BatchInspectionConfig(max_workers=4, target_speed=60)
        self.batch_engine = BatchInspectionEngine(self.inspection_engine, batch_config)
        
        # 初始化数据导出器
        self.data_exporter = DataExporter()
        
        # 初始化报警控制器
        self.alarm = AlarmController()
        
        # 初始化相机
        self.camera = None
        self.camera_type = tk.StringVar(value="USB")
        
        # 图像显示
        self.current_image = None
        self.display_scale = 1.0
        
        # 创建界面
        self._create_menu()
        self._create_layout()
        
        # 设置样式
        self._setup_style()
        
        self.logger.info("增强版GUI初始化完成")
    
    def _setup_style(self):
        """设置样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 配置颜色
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0')
        style.configure('TLabelframe', background='#f0f0f0')
        style.configure('TLabelframe.Label', background='#f0f0f0')
    
    def _create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开图像", command=self._open_image)
        file_menu.add_separator()
        file_menu.add_command(label="历史数据查询", command=self._show_history_query)
        file_menu.add_command(label="统计报表", command=self._show_statistics_report)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menubar.add_cascade(label="文件", menu=file_menu)
        
        # 相机菜单
        camera_menu = tk.Menu(menubar, tearoff=0)
        camera_menu.add_radiobutton(label="USB相机", variable=self.camera_type, 
                                    value="USB", command=self._switch_camera)
        camera_menu.add_radiobutton(label="大恒相机", variable=self.camera_type, 
                                    value="Daheng", command=self._switch_camera)
        camera_menu.add_separator()
        camera_menu.add_command(label="连接相机", command=self._connect_camera)
        camera_menu.add_command(label="断开相机", command=self._disconnect_camera)
        camera_menu.add_command(label="实时预览", command=self._show_camera_preview)
        menubar.add_cascade(label="相机", menu=camera_menu)
        
        # 检测菜单
        detect_menu = tk.Menu(menubar, tearoff=0)
        detect_menu.add_command(label="单次检测", command=self._single_inspection)
        detect_menu.add_command(label="缺陷检测", command=self._defect_inspection)
        menubar.add_cascade(label="检测", menu=detect_menu)
        
        # 工具菜单
        tool_menu = tk.Menu(menubar, tearoff=0)
        tool_menu.add_command(label="图纸标注", command=self._show_annotation_window)
        tool_menu.add_command(label="相机内参标定", command=self._show_calibration_window)
        tool_menu.add_command(label="像素-毫米标定", command=self._show_pixel_mm_calibration_window)
        tool_menu.add_separator()
        tool_menu.add_checkbutton(label="启用报警", command=lambda: self.alarm.enable_alarm(not self.alarm._alarm_enabled))
        menubar.add_cascade(label="工具", menu=tool_menu)
        
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
        
        # 左侧：图像显示区域
        left_frame = ttk.LabelFrame(main_frame, text="图像显示", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas
        self.canvas = tk.Canvas(left_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 缩放控制
        scale_frame = ttk.Frame(left_frame)
        scale_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(scale_frame, text="缩放:").pack(side=tk.LEFT)
        self.scale_var = tk.DoubleVar(value=1.0)
        ttk.Scale(scale_frame, from_=0.1, to=2.0, variable=self.scale_var,
                 command=self._on_scale_change).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(scale_frame, text="适应", command=self._fit_image).pack(side=tk.LEFT)
        
        # 相机控制
        camera_frame = ttk.Frame(left_frame)
        camera_frame.pack(fill=tk.X, pady=5)
        
        self.connect_btn = ttk.Button(camera_frame, text="连接相机", command=self._connect_camera)
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        
        self.capture_btn = ttk.Button(camera_frame, text="采集图像", command=self._capture_image, state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        # 右侧：控制面板
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # 批量检测面板
        self.batch_panel = BatchInspectionPanel(right_frame, self.batch_engine, self.alarm)
        self.batch_panel.pack(fill=tk.X, pady=(0, 10))
        
        # 检测结果面板
        result_frame = ttk.LabelFrame(right_frame, text="检测结果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_text = tk.Text(result_frame, height=15, width=40)
        self.result_text.pack(fill=tk.BOTH, expand=True)
    
    # =========================================================================
    # 相机操作
    # =========================================================================
    
    def _connect_camera(self):
        """连接相机"""
        try:
            if self.camera_type.get() == "USB":
                from inspection_system import USBCameraDriver
                self.camera = USBCameraDriver()
            else:
                from inspection_system import DahengCameraDriver
                self.camera = DahengCameraDriver()
            
            if self.camera.connect():
                self.connect_btn.config(text="断开相机", command=self._disconnect_camera)
                self.capture_btn.config(state=tk.NORMAL)
                messagebox.showinfo("成功", "相机连接成功")
            else:
                messagebox.showerror("错误", "相机连接失败")
        
        except Exception as e:
            self.logger.error(f"连接相机失败: {e}")
            messagebox.showerror("错误", f"连接相机失败: {e}")
    
    def _disconnect_camera(self):
        """断开相机"""
        if self.camera:
            self.camera.disconnect()
            self.camera = None
            self.connect_btn.config(text="连接相机", command=self._connect_camera)
            self.capture_btn.config(state=tk.DISABLED)
            messagebox.showinfo("提示", "相机已断开")
    
    def _switch_camera(self):
        """切换相机类型"""
        if self.camera:
            self._disconnect_camera()
    
    def _capture_image(self):
        """采集图像"""
        if not self.camera:
            return
        
        image = self.camera.capture_image()
        if image is not None:
            self.current_image = image
            self._display_image(image)
    
    # =========================================================================
    # 图像显示
    # =========================================================================
    
    def _open_image(self):
        """打开图像文件"""
        filename = filedialog.askopenfilename(
            filetypes=[("图像文件", "*.jpg *.png *.bmp")]
        )
        
        if filename:
            image = cv2.imread(filename)
            if image is not None:
                self.current_image = image
                self._display_image(image)
    
    def _display_image(self, image: np.ndarray):
        """显示图像"""
        if image is None:
            return
        
        # 转换为RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # 转换为PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # 缩放
        scale = self.scale_var.get()
        if scale != 1.0:
            new_size = (int(image_rgb.shape[1] * scale), int(image_rgb.shape[0] * scale))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        # 转换为Tkinter Image
        self.tk_image = ImageTk.PhotoImage(pil_image)
        
        # 显示
        self.canvas.delete("all")
        self.canvas.config(width=pil_image.width, height=pil_image.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
    
    def _on_scale_change(self, value):
        """缩放变化回调"""
        if self.current_image is not None:
            self._display_image(self.current_image)
    
    def _fit_image(self):
        """适应窗口"""
        if self.current_image is None:
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        image_height, image_width = self.current_image.shape[:2]
        
        scale_x = canvas_width / image_width
        scale_y = canvas_height / image_height
        
        scale = min(scale_x, scale_y, 1.0)
        self.scale_var.set(scale)
        self._display_image(self.current_image)
    
    # =========================================================================
    # 检测操作
    # =========================================================================
    
    def _single_inspection(self):
        """单次检测"""
        if self.current_image is None:
            messagebox.showwarning("警告", "请先加载图像")
            return
        
        try:
            # 执行检测
            result = self.inspection_engine.detect_circle(
                self.current_image,
                part_id=f"MANUAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                part_type="圆形",
                nominal_size=5.0
            )
            
            # 显示结果
            self._display_result(result)
            
            # 在图像上绘制结果
            if result:
                result_image = self.inspection_engine.draw_result(self.current_image, result)
                self._display_image(result_image)
            
            # 保存结果
            self.data_exporter.export_results([result.to_dict()])
        
        except Exception as e:
            self.logger.error(f"检测失败: {e}")
            messagebox.showerror("错误", f"检测失败: {e}")
    
    def _defect_inspection(self):
        """缺陷检测"""
        if self.current_image is None:
            messagebox.showwarning("警告", "请先加载图像")
            return
        
        try:
            # 执行缺陷检测
            result = self.defect_detector.detect_all(self.current_image)
            
            # 显示结果
            result_text = f"""
缺陷检测结果:
- 是否有缺陷: {'是' if result.has_defect else '否'}
- 缺陷数量: {len(result.defects)}
- 质量评分: {result.quality_score:.2f}

缺陷详情:
"""
            for i, defect in enumerate(result.defects):
                result_text += f"{i+1}. {defect.defect_type.value} - 位置: {defect.location}, 严重程度: {defect.severity:.2f}\n"
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result_text)
            
            # 在图像上绘制缺陷
            result_image = self.defect_detector.draw_defects(self.current_image, result)
            self._display_image(result_image)
            
            # 如果有缺陷，触发报警
            if result.has_defect:
                self.alarm.trigger_alarm("NG")
        
        except Exception as e:
            self.logger.error(f"缺陷检测失败: {e}")
            messagebox.showerror("错误", f"缺陷检测失败: {e}")
    
    def _display_result(self, result):
        """显示检测结果"""
        if result is None:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "检测失败，未找到特征")
            return
        
        result_text = f"""
检测结果:
- 零件编号: {result.part_id}
- 零件类型: {result.part_type}
- 检测时间: {result.timestamp}

测量结果:
- 直径: {result.diameter_mm:.3f} mm
- 标称值: {result.nominal_size:.3f} mm
- 公差: {result.tolerance:.3f} mm
- 偏差: {result.deviation:.3f} mm

检测结果: {'合格' if result.is_qualified else '不合格'}
"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result_text)
    
    # =========================================================================
    # 工具窗口
    # =========================================================================
    
    def _show_history_query(self):
        """显示历史查询窗口"""
        HistoryQueryWindow(self.root, self.data_exporter)
    
    def _show_statistics_report(self):
        """显示统计报表窗口"""
        StatisticsReportWindow(self.root, self.data_exporter)
    
    def _show_annotation_window(self):
        """显示图纸标注窗口"""
        DrawingAnnotationWindow(self.root)
    
    def _show_calibration_window(self):
        """显示相机标定窗口"""
        try:
            # 在新线程中运行PyQt5窗口
            import threading
            import sys
            
            def run_calibration():
                from PyQt5.QtWidgets import QApplication
                app = QApplication.instance()
                if app is None:
                    app = QApplication(sys.argv)
                
                calibration_window = CalibrationGUI()
                calibration_window.show()
                app.exec_()
            
            # 创建并启动线程
            thread = threading.Thread(target=run_calibration, daemon=True)
            thread.start()
            
        except ImportError:
            messagebox.showerror("错误", "PyQt5未安装，无法打开相机标定窗口\n\n请运行: pip install PyQt5")
        except Exception as e:
            self.logger.error(f"打开相机标定窗口失败: {e}")
            messagebox.showerror("错误", f"打开相机标定窗口失败: {e}")
    
    def _show_camera_preview(self):
        """显示实时预览窗口"""
        try:
            # 在新线程中运行PyQt5窗口
            import threading
            import sys
            
            def run_preview():
                from PyQt5.QtWidgets import QApplication
                app = QApplication.instance()
                if app is None:
                    app = QApplication(sys.argv)
                
                preview_window = CameraPreviewGUI()
                preview_window.show()
                app.exec_()
            
            # 创建并启动线程
            thread = threading.Thread(target=run_preview, daemon=True)
            thread.start()
            
        except ImportError:
            messagebox.showerror("错误", "PyQt5未安装，无法打开实时预览窗口\n\n请运行: pip install PyQt5")
        except Exception as e:
            self.logger.error(f"打开实时预览窗口失败: {e}")
            messagebox.showerror("错误", f"打开实时预览窗口失败: {e}")
    
    def _show_pixel_mm_calibration_window(self):
        """显示像素-毫米标定窗口"""
        try:
            # 在新线程中运行PyQt5窗口
            import threading
            import sys
            
            def run_calibration():
                from PyQt5.QtWidgets import QApplication
                app = QApplication.instance()
                if app is None:
                    app = QApplication(sys.argv)
                
                calibration_window = PixelMmCalibrationGUI()
                calibration_window.show()
                app.exec_()
            
            # 创建并启动线程
            thread = threading.Thread(target=run_calibration, daemon=True)
            thread.start()
            
        except ImportError:
            messagebox.showerror("错误", "PyQt5未安装，无法打开像素-毫米标定窗口\n\n请运行: pip install PyQt5")
        except Exception as e:
            self.logger.error(f"打开像素-毫米标定窗口失败: {e}")
            messagebox.showerror("错误", f"打开像素-毫米标定窗口失败: {e}")
    
    def _show_about(self):
        """显示关于信息"""
        about_text = """
微小零件中高精度视觉检测系统 V3.0

功能:
- IT8级精度检测（亚像素检测）
- 批量自动检测（≥60件/分钟）
- 缺陷检测（表面、边缘、毛刺）
- 声光报警
- 历史数据查询
- 统计报表
- 图纸标注

技术栈:
- Python 3.8+
- OpenCV 4.5+
- PyQt5 / Tkinter
- NumPy, Pandas

作者: 项目团队
日期: 2026-02-10
"""
        messagebox.showinfo("关于", about_text)


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主函数"""
    root = tk.Tk()
    app = InspectionSystemEnhancedGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
