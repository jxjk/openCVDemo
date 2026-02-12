# -*- coding: utf-8 -*-
"""
模拟GUI DXF导入测试
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drawing_annotation import (
    InspectionTemplate,
    AnnotationTool,
    create_default_template
)

from dxf_parser import dxf_to_template


def test_gui_dxf_import():
    """测试GUI DXF导入功能"""
    
    print("=" * 60)
    print("模拟GUI DXF导入测试")
    print("=" * 60)
    
    # 创建根窗口
    root = tk.Tk()
    root.title("DXF导入测试")
    root.geometry("600x400")
    
    # 创建主框架
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # 测试按钮
    test_button = ttk.Button(
        main_frame,
        text="测试DXF文件导入",
        command=lambda: test_dxf_import(root)
    )
    test_button.pack(pady=20)
    
    # 结果显示
    result_text = tk.Text(main_frame, height=20, width=70)
    result_text.pack(fill=tk.BOTH, expand=True)
    
    # 添加滚动条
    scrollbar = ttk.Scrollbar(result_text)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    result_text.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=result_text.yview)
    
    print("\n启动GUI测试窗口...")
    print("请点击'测试DXF文件导入'按钮\n")
    
    # 运行GUI
    root.mainloop()


def test_dxf_import(parent):
    """测试DXF导入功能"""
    
    def log(message):
        result_text.insert(tk.END, message + "\n")
        result_text.see(tk.END)
        parent.update()
    
    result_text = parent.winfo_children()[0].winfo_children()[1]  # 获取Text控件
    
    log("=" * 60)
    log("开始测试DXF导入")
    log("=" * 60)
    
    # 步骤1: 选择DXF文件
    log("\n[步骤1] 选择DXF文件...")
    
    # 直接使用test.dxf文件路径
    dxf_file = r"D:\Users\00596\Desktop\微小零件低精度视觉检测\参考程序\7.chiCunJianCe.demo\test.dxf"
    
    if not os.path.exists(dxf_file):
        log(f"✗ 文件不存在: {dxf_file}")
        messagebox.showerror("错误", "DXF文件不存在")
        return
    
    log(f"✓ 文件存在: {dxf_file}")
    log(f"  文件大小: {os.path.getsize(dxf_file)} 字节")
    
    # 步骤2: 显示导入选项对话框
    log("\n[步骤2] 显示导入选项...")
    
    import_dialog = tk.Toplevel(parent)
    import_dialog.title("DXF导入选项")
    import_dialog.geometry("400x300")
    import_dialog.transient(parent)
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
    
    result = {'confirmed': False, 'template': None, 'error': None}
    
    def confirm_import():
        try:
            log("\n[步骤3] 开始转换DXF...")
            log(f"  模板名称: {os.path.basename(dxf_file)}")
            log(f"  公差标准: {tolerance_var.get()}")
            log(f"  自动提取尺寸标注: {auto_extract_var.get()}")
            log(f"  自动识别几何特征: {auto_identify_var.get()}")
            
            # 显示进度
            progress = ttk.Progressbar(import_dialog, mode='indeterminate')
            progress.pack(pady=10)
            progress.start()
            import_dialog.update()
            
            # 转换DXF为模板
            template = dxf_to_template(
                dxf_file,
                template_name=os.path.basename(dxf_file),
                tolerance_standard=tolerance_var.get(),
                auto_extract_dimensions=auto_extract_var.get(),
                auto_identify_features=auto_identify_var.get()
            )
            
            progress.stop()
            
            if template:
                log("\n[步骤4] 模板转换成功！")
                log(f"  模板名称: {template.name}")
                log(f"  标注数量: {len(template.annotations)}")
                
                # 显示标注详情
                log("\n标注详情:")
                for i, anno in enumerate(template.annotations, 1):
                    value_str = f"{anno.nominal_value:.3f}" if anno.nominal_value else "未设置"
                    log(f"  {i}. {anno.feature_type.value}: {value_str} mm")
                
                result['template'] = template
                result['confirmed'] = True
                
                messagebox.showinfo("成功", 
                                 f"DXF文件导入成功！\n\n提取了 {len(template.annotations)} 个标注")
                import_dialog.destroy()
            else:
                log("\n✗ 模板转换失败（返回None）")
                result['error'] = "模板转换返回None"
                messagebox.showerror("错误", "DXF文件导入失败")
        
        except ImportError as e:
            progress.stop()
            error_msg = f"ezdxf库未安装: {e}"
            log(f"\n✗ {error_msg}")
            result['error'] = error_msg
            messagebox.showerror("错误", 
                "ezdxf库未安装，无法导入DXF文件。\n\n请运行: pip install ezdxf")
        
        except Exception as e:
            progress.stop()
            error_msg = f"DXF文件导入失败: {e}"
            log(f"\n✗ {error_msg}")
            import traceback
            log("\n详细错误信息:")
            log(traceback.format_exc())
            result['error'] = error_msg
            messagebox.showerror("错误", error_msg)
    
    ttk.Button(import_dialog, text="导入", command=confirm_import).pack(pady=10)
    ttk.Button(import_dialog, text="取消", command=import_dialog.destroy).pack(pady=5)
    
    # 等待对话框关闭
    parent.wait_window(import_dialog)
    
    # 显示结果
    if result['confirmed']:
        log("\n" + "=" * 60)
        log("测试成功！")
        log("=" * 60)
    else:
        log("\n" + "=" * 60)
        log("测试失败")
        if result['error']:
            log(f"错误: {result['error']}")
        log("=" * 60)


if __name__ == "__main__":
    test_gui_dxf_import()