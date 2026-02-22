# -*- coding: utf-8 -*-
"""
DWG文件转换模块
DWG File Converter Module

功能:
- DWG文件转换为DXF格式
- 批量转换支持
- 转换进度监控
- 转换结果验证
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging

from logger_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class ConversionResult:
    """转换结果"""
    success: bool
    input_file: str
    output_file: str
    message: str
    duration: float = 0.0


@dataclass
class BatchConversionResult:
    """批量转换结果"""
    total: int
    success: int
    failed: int
    results: List[ConversionResult]
    duration: float = 0.0


# =============================================================================
# DWG转换器类
# =============================================================================

class DWGConverter:
    """DWG文件转换器"""
    
    def __init__(self, teigha_converter_path: Optional[str] = None):
        """
        初始化DWG转换器
        
        Args:
            teigha_converter_path: ODA File Converter路径，如果为None则自动查找
        """
        self.teigha_converter_path = teigha_converter_path
        self._find_converter()
    
    def _find_converter(self) -> bool:
        """
        查找ODA File Converter
        
        Returns:
            是否找到转换器
        """
        # 常见安装路径
        possible_paths = [
            r"C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe",
            r"C:\Program Files (x86)\ODA\ODAFileConverter\ODAFileConverter.exe",
            r"D:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe",
            os.path.join(os.path.dirname(__file__), "ODAFileConverter.exe"),
        ]
        
        if self.teigha_converter_path:
            if os.path.exists(self.teigha_converter_path):
                logger.info(f"使用指定的转换器: {self.teigha_converter_path}")
                return True
            else:
                logger.warning(f"指定的转换器不存在: {self.teigha_converter_path}")
                self.teigha_converter_path = None
        
        for path in possible_paths:
            if os.path.exists(path):
                self.teigha_converter_path = path
                logger.info(f"找到ODA File Converter: {path}")
                return True
        
        logger.warning("未找到ODA File Converter，请手动安装或指定路径")
        return False
    
    def convert(self, dwg_file: str, output_dxf: Optional[str] = None,
                version: str = "ACAD2018") -> ConversionResult:
        """
        转换单个DWG文件为DXF
        
        Args:
            dwg_file: DWG文件路径
            output_dxf: 输出DXF文件路径，如果为None则自动生成
            version: 输出DXF版本 (ACAD2018, ACAD2013, ACAD2007, ACAD2004, ACAD2000)
        
        Returns:
            转换结果
        """
        import time
        start_time = time.time()
        
        # 检查输入文件
        if not os.path.exists(dwg_file):
            return ConversionResult(
                success=False,
                input_file=dwg_file,
                output_file="",
                message=f"输入文件不存在: {dwg_file}"
            )
        
        if not dwg_file.lower().endswith('.dwg'):
            return ConversionResult(
                success=False,
                input_file=dwg_file,
                output_file="",
                message=f"输入文件不是DWG格式: {dwg_file}"
            )
        
        # 生成输出文件名
        if output_dxf is None:
            output_dxf = os.path.splitext(dwg_file)[0] + '.dxf'
        
        # 检查转换器
        if not self.teigha_converter_path or not os.path.exists(self.teigha_converter_path):
            # 尝试使用其他方法
            return self._convert_alternative(dwg_file, output_dxf)
        
        # 使用ODA File Converter转换
        try:
            # 创建临时文件夹
            input_dir = os.path.dirname(dwg_file)
            output_dir = os.path.dirname(output_dxf)
            
            # 准备输入和输出列表文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                input_list = f.name
                f.write(dwg_file + '\n')
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                output_list = f.name
                f.write(output_dxf + '\n')
            
            # 构建命令
            cmd = [
                self.teigha_converter_path,
                input_dir,
                output_dir,
                version,
                'DXF',  # 输出格式
                '0',    # 递归: 0=否, 1=是
                '1'     # 详细输出: 0=否, 1=是
            ]
            
            logger.info(f"执行转换命令: {' '.join(cmd)}")
            
            # 执行转换
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                encoding='utf-8',
                errors='ignore'
            )
            
            # 清理临时文件
            try:
                os.remove(input_list)
                os.remove(output_list)
            except:
                pass
            
            # 检查结果
            if os.path.exists(output_dxf) and os.path.getsize(output_dxf) > 0:
                duration = time.time() - start_time
                logger.info(f"转换成功: {dwg_file} -> {output_dxf} (耗时: {duration:.2f}秒)")
                return ConversionResult(
                    success=True,
                    input_file=dwg_file,
                    output_file=output_dxf,
                    message="转换成功",
                    duration=duration
                )
            else:
                duration = time.time() - start_time
                logger.error(f"转换失败: {result.stderr}")
                return ConversionResult(
                    success=False,
                    input_file=dwg_file,
                    output_file=output_dxf,
                    message=f"转换失败: {result.stderr}",
                    duration=duration
                )
        
        except subprocess.TimeoutExpired:
            return ConversionResult(
                success=False,
                input_file=dwg_file,
                output_file=output_dxf,
                message="转换超时"
            )
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"转换异常: {str(e)}")
            return ConversionResult(
                success=False,
                input_file=dwg_file,
                output_file=output_dxf,
                message=f"转换异常: {str(e)}",
                duration=duration
            )
    
    def _convert_alternative(self, dwg_file: str, output_dxf: str) -> ConversionResult:
        """
        使用替代方法转换DWG（当ODA File Converter不可用时）
        
        Args:
            dwg_file: DWG文件路径
            output_dxf: 输出DXF文件路径
        
        Returns:
            转换结果
        """
        import time
        start_time = time.time()
        
        logger.warning("ODA File Converter不可用，尝试使用替代方法")
        
        # 方法1: 尝试使用ezdxf的DWG支持（如果可用）
        try:
            import ezdxf
            # ezdxf目前不支持直接读取DWG，但未来可能支持
            logger.info("ezdxf不支持DWG格式，需要手动转换")
        except ImportError:
            pass
        
        # 方法2: 返回提示信息，引导用户手动转换
        duration = time.time() - start_time
        message = """
        DWG转DXF需要以下步骤之一：
        
        1. 使用ODA File Converter（推荐）：
           - 下载地址: https://www.opendesign.com/guestfiles/oda_file_converter
           - 安装后系统会自动检测并使用
        
        2. 使用AutoCAD或其他CAD软件：
           - 打开DWG文件
           - 文件 -> 另存为
           - 选择DXF格式（建议ACAD2018或ACAD2013）
        
        3. 使用在线转换工具：
           - CloudConvert: https://cloudconvert.com/dwg-to-dxf
           - Zamzar: https://www.zamzar.com/convert/dwg-to-dxf/
        """
        
        return ConversionResult(
            success=False,
            input_file=dwg_file,
            output_file=output_dxf,
            message=message.strip(),
            duration=duration
        )
    
    def batch_convert(self, dwg_files: List[str], output_dir: str,
                     version: str = "ACAD2018") -> BatchConversionResult:
        """
        批量转换DWG文件
        
        Args:
            dwg_files: DWG文件列表
            output_dir: 输出目录
            version: 输出DXF版本
        
        Returns:
            批量转换结果
        """
        import time
        start_time = time.time()
        
        results = []
        success_count = 0
        failed_count = 0
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"开始批量转换，共 {len(dwg_files)} 个文件")
        
        for i, dwg_file in enumerate(dwg_files, 1):
            logger.info(f"转换进度: {i}/{len(dwg_files)} - {dwg_file}")
            
            # 生成输出文件路径
            output_dxf = os.path.join(
                output_dir,
                os.path.splitext(os.path.basename(dwg_file))[0] + '.dxf'
            )
            
            # 转换文件
            result = self.convert(dwg_file, output_dxf, version)
            results.append(result)
            
            if result.success:
                success_count += 1
            else:
                failed_count += 1
        
        duration = time.time() - start_time
        
        logger.info(f"批量转换完成: 成功 {success_count}, 失败 {failed_count}, 耗时 {duration:.2f}秒")
        
        return BatchConversionResult(
            total=len(dwg_files),
            success=success_count,
            failed=failed_count,
            results=results,
            duration=duration
        )
    
    def find_dwg_files(self, directory: str, recursive: bool = False) -> List[str]:
        """
        查找目录中的DWG文件
        
        Args:
            directory: 目录路径
            recursive: 是否递归查找子目录
        
        Returns:
            DWG文件列表
        """
        dwg_files = []
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith('.dwg'):
                        dwg_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory):
                if file.lower().endswith('.dwg'):
                    dwg_files.append(os.path.join(directory, file))
        
        logger.info(f"在 {directory} 中找到 {len(dwg_files)} 个DWG文件")
        return dwg_files
    
    def is_available(self) -> bool:
        """
        检查转换器是否可用
        
        Returns:
            转换器是否可用
        """
        return self.teigha_converter_path is not None and os.path.exists(self.teigha_converter_path)


# =============================================================================
# 便捷函数
# =============================================================================

def convert_dwg_to_dxf(dwg_file: str, output_dxf: Optional[str] = None,
                       version: str = "ACAD2018") -> ConversionResult:
    """
    便捷函数：转换单个DWG文件
    
    Args:
        dwg_file: DWG文件路径
        output_dxf: 输出DXF文件路径
        version: 输出DXF版本
    
    Returns:
        转换结果
    """
    converter = DWGConverter()
    return converter.convert(dwg_file, output_dxf, version)


def batch_convert_dwg(dwg_files: List[str], output_dir: str,
                     version: str = "ACAD2018") -> BatchConversionResult:
    """
    便捷函数：批量转换DWG文件
    
    Args:
        dwg_files: DWG文件列表
        output_dir: 输出目录
        version: 输出DXF版本
    
    Returns:
        批量转换结果
    """
    converter = DWGConverter()
    return converter.batch_convert(dwg_files, output_dir, version)


# =============================================================================
# 主函数（用于命令行测试）
# =============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python dwg_converter.py <dwg文件> [输出dxf文件] [版本]")
        print("  python dwg_converter.py --batch <输入目录> <输出目录> [版本]")
        print("\n示例:")
        print("  python dwg_converter.py drawing.dwg")
        print("  python dwg_converter.py drawing.dwg output.dxf ACAD2018")
        print("  python dwg_converter.py --batch ./input ./output ACAD2013")
        sys.exit(1)
    
    if sys.argv[1] == '--batch':
        # 批量转换模式
        if len(sys.argv) < 4:
            print("批量转换模式需要指定输入目录和输出目录")
            sys.exit(1)
        
        input_dir = sys.argv[2]
        output_dir = sys.argv[3]
        version = sys.argv[4] if len(sys.argv) > 4 else "ACAD2018"
        
        converter = DWGConverter()
        dwg_files = converter.find_dwg_files(input_dir, recursive=True)
        
        if not dwg_files:
            print(f"在 {input_dir} 中未找到DWG文件")
            sys.exit(1)
        
        print(f"找到 {len(dwg_files)} 个DWG文件")
        print(f"输出目录: {output_dir}")
        print(f"输出版本: {version}")
        print()
        
        result = converter.batch_convert(dwg_files, output_dir, version)
        
        print()
        print("=" * 60)
        print("批量转换结果")
        print("=" * 60)
        print(f"总数: {result.total}")
        print(f"成功: {result.success}")
        print(f"失败: {result.failed}")
        print(f"耗时: {result.duration:.2f}秒")
        print()
        
        if result.failed > 0:
            print("失败的文件:")
            for r in result.results:
                if not r.success:
                    print(f"  - {r.input_file}: {r.message}")
        
        sys.exit(0 if result.failed == 0 else 1)
    
    else:
        # 单文件转换模式
        dwg_file = sys.argv[1]
        output_dxf = sys.argv[2] if len(sys.argv) > 2 else None
        version = sys.argv[3] if len(sys.argv) > 3 else "ACAD2018"
        
        print(f"输入文件: {dwg_file}")
        print(f"输出文件: {output_dxf or '自动生成'}")
        print(f"输出版本: {version}")
        print()
        
        result = convert_dwg_to_dxf(dwg_file, output_dxf, version)
        
        print("=" * 60)
        print("转换结果")
        print("=" * 60)
        print(f"状态: {'成功' if result.success else '失败'}")
        print(f"输入: {result.input_file}")
        print(f"输出: {result.output_file}")
        print(f"耗时: {result.duration:.2f}秒")
        print(f"消息: {result.message}")
        
        sys.exit(0 if result.success else 1)
