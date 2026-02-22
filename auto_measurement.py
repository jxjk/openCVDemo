# -*- coding: utf-8 -*-
"""
自动测量模块
Auto Measurement Module

功能:
- 基于图纸标注的自动测量
- DWG/DXF图纸解析
- 图像配准和坐标转换
- 自动生成测量报告
"""

import os
import cv2
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

from logger_config import get_logger
from drawing_annotation import (
    InspectionTemplate,
    CircleAnnotation,
    LineAnnotation,
    FeatureType,
    ToleranceStandard,
    create_default_template
)
from dxf_parser import DXFParser, DXFDimension, DXFToTemplateConverter
from dwg_converter import DWGConverter, convert_dwg_to_dxf
from image_registration import ImageRegistration, TransformationMatrix
from inspection_system import InspectionEngine, InspectionConfig

logger = get_logger(__name__)


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class MeasurementResult:
    """测量结果"""
    annotation_id: str  # 标注ID
    feature_type: str  # 特征类型
    measured_value: float  # 测量值
    nominal_value: Optional[float] = None  # 标称值
    tolerance: Optional[float] = None  # 公差
    deviation: Optional[float] = None  # 偏差
    is_passed: bool = True  # 是否合格
    confidence: float = 0.0  # 置信度
    message: str = ""  # 消息


@dataclass
class AutoMeasurementReport:
    """自动测量报告"""
    template_name: str  # 模板名称
    dwg_file: str  # DWG文件路径
    image_file: str  # 图像文件路径
    timestamp: str  # 时间戳
    registration_success: bool  # 配准是否成功
    registration_confidence: float  # 配准置信度
    total_features: int  # 总特征数
    measured_features: int  # 已测量特征数
    passed_features: int  # 合格特征数
    failed_features: int  # 不合格特征数
    results: List[MeasurementResult] = field(default_factory=list)  # 测量结果列表
    duration: float = 0.0  # 总耗时


# =============================================================================
# 自动测量引擎
# =============================================================================

class AutoMeasurementEngine:
    """自动测量引擎"""
    
    def __init__(self, config: Optional[InspectionConfig] = None):
        """
        初始化自动测量引擎
        
        Args:
            config: 检测配置
        """
        self.config = config or InspectionConfig()
        self.inspection_engine = InspectionEngine(self.config)
        self.registration = ImageRegistration()
        self.dwg_converter = DWGConverter()
        self.dxf_parser = DXFParser()
    
    def measure_from_dwg(self, dwg_file: str, image_file: str,
                        output_dir: Optional[str] = None) -> AutoMeasurementReport:
        """
        从DWG文件自动测量
        
        Args:
            dwg_file: DWG文件路径
            image_file: 图像文件路径
            output_dir: 输出目录
        
        Returns:
            测量报告
        """
        import time
        start_time = time.time()
        
        logger.info(f"开始自动测量: DWG={dwg_file}, 图像={image_file}")
        
        # 创建报告
        report = AutoMeasurementReport(
            template_name=os.path.splitext(os.path.basename(dwg_file))[0],
            dwg_file=dwg_file,
            image_file=image_file,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            registration_success=False,
            registration_confidence=0.0
        )
        
        # 1. 转换DWG为DXF
        logger.info("步骤1: 转换DWG为DXF")
        temp_dxf = os.path.splitext(dwg_file)[0] + '_temp.dxf'
        conversion_result = convert_dwg_to_dxf(dwg_file, temp_dxf)
        
        if not conversion_result.success:
            report.message = f"DWG转换失败: {conversion_result.message}"
            logger.error(report.message)
            return report
        
        # 2. 解析DXF，提取标注
        logger.info("步骤2: 解析DXF文件")
        self.dxf_parser.load(temp_dxf)
        converter = DXFToTemplateConverter(self.dxf_parser)
        template = converter.convert(
            template_name=os.path.splitext(os.path.basename(dwg_file))[0],
            tolerance_standard=ToleranceStandard.IT8
        )
        
        if template is None:
            report.message = "DXF解析失败"
            logger.error(report.message)
            return report
        
        report.total_features = len(template.annotations)
        logger.info(f"提取到 {report.total_features} 个标注")
        
        # 3. 渲染DXF为模板图像
        logger.info("步骤3: 渲染DXF为模板图像")
        template_image = self._render_dxf_to_image(temp_dxf)
        
        if template_image is None:
            report.message = "DXF渲染失败"
            logger.error(report.message)
            return report
        
        # 4. 读取检测图像
        logger.info("步骤4: 读取检测图像")
        image = cv2.imread(image_file)
        
        if image is None:
            report.message = f"无法读取图像: {image_file}"
            logger.error(report.message)
            return report
        
        # 5. 图像配准
        logger.info("步骤5: 图像配准")
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
        
        reg_result = self.registration.register(image_gray, template_gray, method='homography')
        
        if not reg_result.success:
            report.message = f"图像配准失败: {reg_result.message}"
            logger.error(report.message)
            return report
        
        report.registration_success = True
        report.registration_confidence = reg_result.transformation.confidence
        logger.info(f"配准成功，置信度: {report.registration_confidence:.2f}")
        
        # 6. 自动测量
        logger.info("步骤6: 自动测量特征")
        report = self._measure_features(
            image, template, reg_result.transformation, report
        )
        
        # 7. 生成报告
        report.duration = time.time() - start_time
        logger.info(f"测量完成，耗时: {report.duration:.2f}秒")
        
        # 8. 保存报告
        if output_dir:
            self._save_report(report, output_dir)
        
        # 清理临时文件
        try:
            if os.path.exists(temp_dxf):
                os.remove(temp_dxf)
        except:
            pass
        
        return report
    
    def _render_dxf_to_image(self, dxf_file: str,
                            image_size: Tuple[int, int] = (2000, 2000),
                            background_color: Tuple[int, int, int] = (255, 255, 255),
                            line_color: Tuple[int, int, int] = (0, 0, 0),
                            line_width: int = 2) -> Optional[np.ndarray]:
        """
        渲染DXF为图像
        
        Args:
            dxf_file: DXF文件路径
            image_size: 图像大小
            background_color: 背景颜色 (0-255范围)
            line_color: 线条颜色 (0-255范围)
            line_width: 线条宽度
        
        Returns:
            渲染的图像
        """
        # 转换颜色值为matplotlib需要的0-1范围
        bg_color_norm = tuple(c / 255.0 for c in background_color)
        
        try:
            import ezdxf
            from ezdxf.addons.drawing import RenderContext, Frontend
            from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            
            # 加载DXF
            doc = ezdxf.readfile(dxf_file)
            msp = doc.modelspace()
            
            # 创建渲染上下文
            ctx = RenderContext(doc)
            
            # 获取DXF的边界
            extents = msp.extent()
            if extents is None:
                logger.warning("无法获取DXF边界，使用简化渲染")
                return self._simple_render_dxf(dxf_file, image_size, background_color, line_color, line_width)
            
            min_x, min_y, max_x, max_y = extents
            
            # 创建matplotlib图像
            fig, ax = plt.subplots(figsize=(image_size[0]/100, image_size[1]/100))
            ax.set_facecolor(bg_color_norm)
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.set_aspect('equal')
            
            # 渲染
            frontend = Frontend(ctx, MatplotlibBackend(ax))
            frontend.draw_layout(msp, finalize=True)
            
            # 转换为OpenCV图像
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            plt.close(fig)
            
            # 调整大小
            img = cv2.resize(img, image_size)
            
            return img
        
        except ImportError:
            logger.warning("matplotlib或ezdxf不可用，使用简化渲染")
            return self._simple_render_dxf(dxf_file, image_size, background_color, line_color, line_width)
        except Exception as e:
            logger.error(f"DXF渲染失败: {str(e)}")
            logger.info("尝试使用简化渲染方法")
            return self._simple_render_dxf(dxf_file, image_size, background_color, line_color, line_width)
    
    def _simple_render_dxf(self, dxf_file: str,
                          image_size: Tuple[int, int] = (2000, 2000),
                          background_color: Tuple[int, int, int] = (255, 255, 255),
                          line_color: Tuple[int, int, int] = (0, 0, 0),
                          line_width: int = 2) -> Optional[np.ndarray]:
        """
        简化的DXF渲染（不依赖matplotlib）
        
        Args:
            dxf_file: DXF文件路径
            image_size: 图像大小
            background_color: 背景颜色
            line_color: 线条颜色
            line_width: 线条宽度
        
        Returns:
            渲染的图像
        """
        try:
            import ezdxf
            
            # 加载DXF
            doc = ezdxf.readfile(dxf_file)
            msp = doc.modelspace()
            
            # 创建空白图像
            img = np.full((image_size[1], image_size[0], 3), background_color, dtype=np.uint8)
            
            # 获取边界
            extents = msp.extent()
            if extents is None:
                return img
            
            min_x, min_y, max_x, max_y = extents
            width = max_x - min_x
            height = max_y - min_y
            
            if width == 0 or height == 0:
                return img
            
            # 计算缩放比例
            scale_x = image_size[0] / width
            scale_y = image_size[1] / height
            scale = min(scale_x, scale_y) * 0.9  # 留10%边距
            
            # 偏移量（居中）
            offset_x = (image_size[0] - width * scale) / 2 - min_x * scale
            offset_y = (image_size[1] - height * scale) / 2 - min_y * scale
            
            # 绘制实体
            for entity in msp:
                if entity.dxftype() == 'LINE':
                    start = entity.start
                    end = entity.end
                    pt1 = (int(start[0] * scale + offset_x), int(image_size[1] - (start[1] * scale + offset_y)))
                    pt2 = (int(end[0] * scale + offset_x), int(image_size[1] - (end[1] * scale + offset_y)))
                    cv2.line(img, pt1, pt2, line_color, line_width)
                
                elif entity.dxftype() == 'CIRCLE':
                    center = entity.center
                    radius = entity.dxf.radius * scale
                    center_pt = (int(center[0] * scale + offset_x), int(image_size[1] - (center[1] * scale + offset_y)))
                    cv2.circle(img, center_pt, int(radius), line_color, line_width)
                
                elif entity.dxftype() == 'ARC':
                    center = entity.center
                    radius = entity.dxf.radius * scale
                    center_pt = (int(center[0] * scale + offset_x), int(image_size[1] - (center[1] * scale + offset_y)))
                    start_angle = entity.dxf.start_angle
                    end_angle = entity.dxf.end_angle
                    cv2.ellipse(img, center_pt, (int(radius), int(radius)), 0, start_angle, end_angle, line_color, line_width)
            
            return img
        
        except Exception as e:
            logger.error(f"简化渲染失败: {str(e)}")
            return None
    
    def _measure_features(self, image: np.ndarray, template: InspectionTemplate,
                         transformation: TransformationMatrix,
                         report: AutoMeasurementReport) -> AutoMeasurementReport:
        """
        测量所有特征
        
        Args:
            image: 输入图像
            template: 检测模板
            transformation: 变换矩阵
            report: 测量报告
        
        Returns:
            更新后的测量报告
        """
        for annotation in template.annotations:
            try:
                # 转换标注坐标到图像坐标系
                if isinstance(annotation, CircleAnnotation):
                    # 圆形特征
                    center = self.registration.transform_point(
                        (annotation.center.x, annotation.center.y),
                        transformation
                    )
                    
                    # 在图像中检测圆
                    measured_value = self._measure_circle(
                        image, center, annotation.nominal_radius
                    )
                    
                    # 计算偏差
                    deviation = measured_value * 2 - annotation.nominal_diameter
                    
                    # 判断是否合格
                    is_passed = abs(deviation) <= annotation.tolerance
                    
                    # 添加结果
                    result = MeasurementResult(
                        annotation_id=annotation.id,
                        feature_type=annotation.feature_type.value,
                        measured_value=measured_value * 2,  # 返回直径
                        nominal_value=annotation.nominal_diameter,
                        tolerance=annotation.tolerance,
                        deviation=deviation,
                        is_passed=is_passed,
                        confidence=0.9,
                        message=f"直径: {measured_value * 2:.3f}mm"
                    )
                
                elif isinstance(annotation, LineAnnotation):
                    # 线段特征
                    start = self.registration.transform_point(
                        (annotation.start.x, annotation.start.y),
                        transformation
                    )
                    end = self.registration.transform_point(
                        (annotation.end.x, annotation.end.y),
                        transformation
                    )
                    
                    # 测量距离
                    measured_value = np.sqrt(
                        (end[0] - start[0])**2 + (end[1] - start[1])**2
                    ) * self.config.PIXEL_TO_MM
                    
                    # 计算偏差
                    deviation = measured_value - annotation.nominal_length
                    
                    # 判断是否合格
                    is_passed = abs(deviation) <= annotation.tolerance
                    
                    # 添加结果
                    result = MeasurementResult(
                        annotation_id=annotation.id,
                        feature_type=annotation.feature_type.value,
                        measured_value=measured_value,
                        nominal_value=annotation.nominal_length,
                        tolerance=annotation.tolerance,
                        deviation=deviation,
                        is_passed=is_passed,
                        confidence=0.9,
                        message=f"长度: {measured_value:.3f}mm"
                    )
                
                else:
                    continue
                
                report.results.append(result)
                report.measured_features += 1
                
                if is_passed:
                    report.passed_features += 1
                else:
                    report.failed_features += 1
                
                logger.info(f"测量 {annotation.feature_type.value}: {result.message}")
            
            except Exception as e:
                logger.error(f"测量特征失败 {annotation.id}: {str(e)}")
                continue
        
        return report
    
    def _measure_circle(self, image: np.ndarray, center: Tuple[float, float],
                       nominal_radius: float) -> float:
        """
        测量圆的半径
        
        Args:
            image: 输入图像
            center: 圆心坐标（像素）
            nominal_radius: 标称半径（毫米）
        
        Returns:
            测量的半径（毫米）
        """
        # 转换为像素
        pixel_radius = nominal_radius / self.config.PIXEL_TO_MM
        
        # 定义ROI
        x, y = int(center[0]), int(center[1])
        r = int(pixel_radius * 1.5)  # 扩大ROI范围
        
        if x - r < 0 or y - r < 0 or x + r >= image.shape[1] or y + r >= image.shape[0]:
            return nominal_radius  # ROI超出图像范围，返回标称值
        
        roi = image[y-r:y+r, x-r:x+r]
        
        # 检测圆
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=r,
            param1=50,
            param2=30,
            minRadius=int(pixel_radius * 0.8),
            maxRadius=int(pixel_radius * 1.2)
        )
        
        if circles is not None and len(circles[0]) > 0:
            # 取第一个圆
            circle = circles[0][0]
            measured_radius_pixel = circle[2]
            measured_radius = measured_radius_pixel * self.config.PIXEL_TO_MM
            return measured_radius
        else:
            # 未检测到圆，返回标称值
            return nominal_radius
    
    def _save_report(self, report: AutoMeasurementReport, output_dir: str):
        """
        保存测量报告
        
        Args:
            report: 测量报告
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存JSON报告
        json_path = os.path.join(output_dir, f"{report.template_name}_report.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'template_name': report.template_name,
                'dwg_file': report.dwg_file,
                'image_file': report.image_file,
                'timestamp': report.timestamp,
                'registration_success': report.registration_success,
                'registration_confidence': report.registration_confidence,
                'total_features': report.total_features,
                'measured_features': report.measured_features,
                'passed_features': report.passed_features,
                'failed_features': report.failed_features,
                'duration': report.duration,
                'results': [
                    {
                        'annotation_id': r.annotation_id,
                        'feature_type': r.feature_type,
                        'measured_value': r.measured_value,
                        'nominal_value': r.nominal_value,
                        'tolerance': r.tolerance,
                        'deviation': r.deviation,
                        'is_passed': r.is_passed,
                        'confidence': r.confidence,
                        'message': r.message
                    }
                    for r in report.results
                ]
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"报告已保存到: {json_path}")
        
        # 保存文本报告
        txt_path = os.path.join(output_dir, f"{report.template_name}_report.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("自动测量报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"模板名称: {report.template_name}\n")
            f.write(f"DWG文件: {report.dwg_file}\n")
            f.write(f"图像文件: {report.image_file}\n")
            f.write(f"时间戳: {report.timestamp}\n")
            f.write(f"配准成功: {'是' if report.registration_success else '否'}\n")
            f.write(f"配准置信度: {report.registration_confidence:.2f}\n\n")
            f.write(f"总特征数: {report.total_features}\n")
            f.write(f"已测量特征数: {report.measured_features}\n")
            f.write(f"合格特征数: {report.passed_features}\n")
            f.write(f"不合格特征数: {report.failed_features}\n")
            f.write(f"总耗时: {report.duration:.2f}秒\n\n")
            f.write("-" * 60 + "\n")
            f.write("测量结果\n")
            f.write("-" * 60 + "\n\n")
            
            for result in report.results:
                f.write(f"特征ID: {result.annotation_id}\n")
                f.write(f"特征类型: {result.feature_type}\n")
                f.write(f"测量值: {result.measured_value:.3f}mm\n")
                if result.nominal_value is not None:
                    f.write(f"标称值: {result.nominal_value:.3f}mm\n")
                if result.tolerance is not None:
                    f.write(f"公差: ±{result.tolerance:.3f}mm\n")
                if result.deviation is not None:
                    f.write(f"偏差: {result.deviation:+.3f}mm\n")
                f.write(f"状态: {'合格' if result.is_passed else '不合格'}\n")
                f.write(f"消息: {result.message}\n")
                f.write("\n")
        
        logger.info(f"文本报告已保存到: {txt_path}")


# =============================================================================
# 便捷函数
# =============================================================================

def auto_measure(dwg_file: str, image_file: str,
                output_dir: Optional[str] = None) -> AutoMeasurementReport:
    """
    便捷函数：自动测量
    
    Args:
        dwg_file: DWG文件路径
        image_file: 图像文件路径
        output_dir: 输出目录
    
    Returns:
        测量报告
    """
    engine = AutoMeasurementEngine()
    return engine.measure_from_dwg(dwg_file, image_file, output_dir)


# =============================================================================
# 主函数（用于命令行测试）
# =============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("用法:")
        print("  python auto_measurement.py <dwg文件> <图像文件> [输出目录]")
        print("\n示例:")
        print("  python auto_measurement.py part.dwg part.jpg")
        print("  python auto_measurement.py part.dwg part.jpg ./reports")
        sys.exit(1)
    
    dwg_file = sys.argv[1]
    image_file = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "./reports"
    
    print(f"DWG文件: {dwg_file}")
    print(f"图像文件: {image_file}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 执行自动测量
    report = auto_measure(dwg_file, image_file, output_dir)
    
    # 打印结果
    print("=" * 60)
    print("自动测量报告")
    print("=" * 60)
    print(f"状态: {'成功' if report.registration_success else '失败'}")
    print(f"配准置信度: {report.registration_confidence:.2f}")
    print(f"总特征数: {report.total_features}")
    print(f"已测量特征数: {report.measured_features}")
    print(f"合格特征数: {report.passed_features}")
    print(f"不合格特征数: {report.failed_features}")
    print(f"总耗时: {report.duration:.2f}秒")
    print()
    
    if report.failed_features > 0:
        print("不合格特征:")
        for result in report.results:
            if not result.is_passed:
                print(f"  - {result.annotation_id}: {result.message}")
        print()
    
    print("详细报告已保存到输出目录")