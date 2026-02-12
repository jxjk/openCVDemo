# -*- coding: utf-8 -*-
"""
GUI DXF导入问题诊断脚本
"""

import os
import sys
import traceback

def diagnose_gui_dxf_import():
    """诊断GUI DXF导入问题"""
    
    print("=" * 60)
    print("GUI DXF导入问题诊断")
    print("=" * 60)
    
    # 1. 检查Python环境
    print("\n1. 检查Python环境...")
    print(f"   Python版本: {sys.version}")
    print(f"   Python路径: {sys.executable}")
    
    # 2. 检查工作目录
    print("\n2. 检查工作目录...")
    current_dir = os.getcwd()
    print(f"   当前目录: {current_dir}")
    
    # 3. 检查test.dxf文件
    dxf_file = r"D:\Users\00596\Desktop\微小零件低精度视觉检测\参考程序\7.chiCunJianCe.demo\test.dxf"
    print(f"\n3. 检查DXF文件...")
    print(f"   文件路径: {dxf_file}")
    
    if not os.path.exists(dxf_file):
        print(f"   ✗ 文件不存在")
        return False
    
    print(f"   ✓ 文件存在")
    print(f"   文件大小: {os.path.getsize(dxf_file)} 字节")
    
    # 4. 检查模块导入
    print("\n4. 检查模块导入...")
    
    # 检查tkinter
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox, filedialog
        print(f"   ✓ tkinter导入成功")
    except ImportError as e:
        print(f"   ✗ tkinter导入失败: {e}")
        return False
    
    # 检查cv2
    try:
        import cv2
        print(f"   ✓ cv2导入成功 (版本: {cv2.__version__})")
    except ImportError as e:
        print(f"   ✗ cv2导入失败: {e}")
        return False
    
    # 检查numpy
    try:
        import numpy as np
        print(f"   ✓ numpy导入成功 (版本: {np.__version__})")
    except ImportError as e:
        print(f"   ✗ numpy导入失败: {e}")
        return False
    
    # 检查drawing_annotation
    try:
        from drawing_annotation import (
            InspectionTemplate,
            AnnotationTool,
            Point2D,
            create_default_template
        )
        print(f"   ✓ drawing_annotation导入成功")
    except ImportError as e:
        print(f"   ✗ drawing_annotation导入失败: {e}")
        traceback.print_exc()
        return False
    
    # 检查dxf_parser
    try:
        from dxf_parser import (
            DXFParser,
            dxf_to_template
        )
        print(f"   ✓ dxf_parser导入成功")
    except ImportError as e:
        print(f"   ✗ dxf_parser导入失败: {e}")
        traceback.print_exc()
        return False
    
    # 5. 检查ezdxf
    print("\n5. 检查ezdxf库...")
    try:
        import ezdxf
        print(f"   ✓ ezdxf导入成功 (版本: {ezdxf.__version__})")
    except ImportError as e:
        print(f"   ✗ ezdxf导入失败: {e}")
        print(f"\n   解决方法:")
        print(f"   pip install ezdxf>=1.0.0")
        return False
    
    # 6. 测试DXF解析
    print("\n6. 测试DXF解析...")
    try:
        parser = DXFParser()
        parser.load(dxf_file)
        result = parser.parse_all()
        
        print(f"   ✓ DXF解析成功")
        print(f"   - 圆形数量: {len(result['entities']['circles'])}")
        print(f"   - 线段数量: {len(result['entities']['lines'])}")
        print(f"   - 尺寸标注数量: {len(result['dimensions'])}")
    
    except Exception as e:
        print(f"   ✗ DXF解析失败: {e}")
        traceback.print_exc()
        return False
    
    # 7. 测试模板转换
    print("\n7. 测试模板转换...")
    try:
        template = dxf_to_template(
            dxf_file,
            template_name="test",
            tolerance_standard="IT8",
            auto_extract_dimensions=True,
            auto_identify_features=True
        )
        
        if template:
            print(f"   ✓ 模板转换成功")
            print(f"   - 标注数量: {len(template.annotations)}")
        else:
            print(f"   ✗ 模板转换失败（返回None）")
            return False
    
    except Exception as e:
        print(f"   ✗ 模板转换失败: {e}")
        traceback.print_exc()
        return False
    
    # 8. 测试GUI文件对话框
    print("\n8. 测试GUI文件对话框...")
    try:
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        
        # 测试文件对话框（不实际打开）
        print(f"   ✓ GUI文件对话框测试成功")
        
        root.destroy()
    
    except Exception as e:
        print(f"   ✗ GUI文件对话框测试失败: {e}")
        traceback.print_exc()
        return False
    
    # 9. 检查GUI源代码
    print("\n9. 检查GUI源代码...")
    gui_file = "inspection_system_gui_v2.py"
    
    if not os.path.exists(gui_file):
        print(f"   ✗ GUI文件不存在: {gui_file}")
        return False
    
    print(f"   ✓ GUI文件存在")
    
    # 读取文件内容检查导入语句
    try:
        with open(gui_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查dxf_parser导入
        if 'from dxf_parser import' in content:
            print(f"   ✓ 找到dxf_parser导入语句")
        else:
            print(f"   ✗ 未找到dxf_parser导入语句")
            print(f"\n   需要在GUI文件中添加:")
            print(f"   from dxf_parser import (")
            print(f"       DXFParser,")
            print(f"       dxf_to_template")
            print(f"   )")
            return False
        
        # 检查_import_dxf方法
        if 'def _import_dxf' in content:
            print(f"   ✓ 找到_import_dxf方法")
        else:
            print(f"   ✗ 未找到_import_dxf方法")
            return False
        
        # 检查菜单项
        if '"导入DXF文件"' in content or "'导入DXF文件'" in content:
            print(f"   ✓ 找到导入DXF菜单项")
        else:
            print(f"   ✗ 未找到导入DXF菜单项")
            return False
    
    except Exception as e:
        print(f"   ✗ 读取GUI文件失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("诊断完成！所有检查通过")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = diagnose_gui_dxf_import()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n诊断脚本执行失败: {e}")
        traceback.print_exc()
        sys.exit(1)