# -*- coding: utf-8 -*-
"""
DXF文件加载测试脚本
用于诊断test.dxf无法加载的问题
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_dxf_loading():
    """测试DXF文件加载"""
    
    dxf_file = r"D:\Users\00596\Desktop\微小零件低精度视觉检测\参考程序\7.chiCunJianCe.demo\test.dxf"
    
    print("=" * 60)
    print("DXF文件加载测试")
    print("=" * 60)
    
    # 1. 检查文件是否存在
    print("\n1. 检查文件...")
    if not os.path.exists(dxf_file):
        print(f"   ✗ 文件不存在: {dxf_file}")
        return False
    print(f"   ✓ 文件存在")
    
    # 2. 检查文件大小
    file_size = os.path.getsize(dxf_file)
    print(f"\n2. 文件大小: {file_size} 字节 ({file_size / 1024:.2f} KB)")
    
    # 3. 检查文件前几行
    print("\n3. 文件前20行:")
    print("-" * 60)
    try:
        with open(dxf_file, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f, 1):
                if i > 20:
                    break
                print(f"   {line.rstrip()}")
    except Exception as e:
        print(f"   ✗ 读取文件失败: {e}")
        return False
    print("-" * 60)
    
    # 4. 检查ezdxf库
    print("\n4. 检查ezdxf库...")
    try:
        import ezdxf
        print(f"   ✓ ezdxf已安装")
        print(f"   版本: {ezdxf.__version__ if hasattr(ezdxf, '__version__') else '未知'}")
    except ImportError as e:
        print(f"   ✗ ezdxf未安装: {e}")
        print("\n解决方法:")
        print("   pip install ezdxf")
        return False
    
    # 5. 尝试加载DXF文件
    print("\n5. 尝试加载DXF文件...")
    try:
        doc = ezdxf.readfile(dxf_file)
        print(f"   ✓ DXF文件加载成功")
        
        # 获取文件信息
        print(f"\n   DXF文件信息:")
        print(f"   - DXF版本: {doc.dxfversion}")
        print(f"   - 单位: {doc.units}")
        print(f"   - 图层数量: {len(doc.layers)}")
        
        # 获取模型空间实体数量
        msp = doc.modelspace()
        entity_count = len(list(msp))
        print(f"   - 模型空间实体数量: {entity_count}")
        
        # 统计实体类型
        entity_types = {}
        for entity in msp:
            entity_type = entity.dxftype()
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        print(f"\n   实体类型统计:")
        for entity_type, count in sorted(entity_types.items()):
            print(f"   - {entity_type}: {count}")
        
        # 检查尺寸标注
        dimensions = []
        for entity in msp:
            if entity.dxftype() == 'DIMENSION':
                dimensions.append(entity)
        
        print(f"\n   尺寸标注数量: {len(dimensions)}")
        
        if dimensions:
            print(f"\n   前3个尺寸标注:")
            for i, dim in enumerate(dimensions[:3], 1):
                try:
                    text = getattr(dim, 'text', '') or getattr(dim.dxf, 'text', '')
                    dim_type = dim.dxf.dimtype if hasattr(dim.dxf, 'dimtype') else 'unknown'
                    print(f"   {i}. 类型={dim_type}, 文本='{text}'")
                except Exception as e:
                    print(f"   {i}. 读取失败: {e}")
        
        # 6. 使用DXFParser测试
        print("\n6. 使用DXFParser测试...")
        try:
            from dxf_parser import DXFParser, DXFToTemplateConverter, dxf_to_template
            
            parser = DXFParser()
            parser.load(dxf_file)
            result = parser.parse_all()
            
            print(f"   ✓ DXFParser解析成功")
            print(f"   - 圆形数量: {len(result['entities']['circles'])}")
            print(f"   - 线段数量: {len(result['entities']['lines'])}")
            print(f"   - 尺寸标注数量: {len(result['dimensions'])}")
            
            # 7. 尝试转换为模板
            print("\n7. 尝试转换为模板...")
            template = dxf_to_template(
                dxf_file,
                template_name="test模板",
                tolerance_standard="IT8",
                auto_extract_dimensions=True,
                auto_identify_features=True
            )
            
            if template:
                print(f"   ✓ 模板转换成功")
                print(f"   - 模板名称: {template.name}")
                print(f"   - 标注数量: {len(template.annotations)}")
                
                print(f"\n   标注列表:")
                for i, anno in enumerate(template.annotations, 1):
                    value_str = f"{anno.nominal_value:.3f}" if anno.nominal_value else "未设置"
                    print(f"   {i}. {anno.feature_type.value}: {value_str} mm")
            else:
                print(f"   ✗ 模板转换失败")
            
            return True
        
        except Exception as e:
            print(f"   ✗ DXFParser测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    except ezdxf.DXFStructureError as e:
        print(f"   ✗ DXF文件结构错误: {e}")
        print(f"\n   可能的原因:")
        print(f"   - DXF文件损坏")
        print(f"   - DXF版本过旧或过新")
        print(f"   - 文件编码问题")
        return False
    
    except Exception as e:
        print(f"   ✗ 加载DXF文件失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = test_dxf_loading()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n测试脚本执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)