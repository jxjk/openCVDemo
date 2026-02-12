# -*- coding: utf-8 -*-
"""
DXF文件解析器 - 增强版本
DXF File Parser - Enhanced Version

功能:
- 解析DXF文件中的几何实体
- 提取尺寸标注信息
- 自动识别检测特征
- 转换为标注模板
"""

import os
import re
import math
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    import ezdxf
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False
    print("警告: ezdxf库未安装，DXF解析功能不可用")

from drawing_annotation import (
    InspectionTemplate,
    CircleAnnotation,
    LineAnnotation,
    Point2D,
    FeatureType,
    ToleranceStandard,
    create_default_template
)


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class DXFDimension:
    """DXF尺寸标注"""
    dim_type: str  # 'linear', 'radius', 'diameter', 'angular'
    text: str  # 标注文本
    value: Optional[float] = None  # 数值
    unit: str = "mm"  # 单位
    points: List[Point2D] = None  # 标注点
    layer: str = ""  # 图层
    
    def __post_init__(self):
        if self.points is None:
            self.points = []
    
    def extract_value(self) -> Optional[float]:
        """从文本中提取数值"""
        if self.value is not None:
            return self.value
        
        # 移除单位符号
        text = self.text.replace('mm', '').replace('°', '').replace('°', '')
        text = text.strip()
        
        # 处理希腊字母
        text = text.replace('φ', '').replace('Φ', '').replace('Ø', '')
        text = text.replace('r', '').replace('R', '')
        text = text.replace('⌐', '')
        
        # 提取数字
        match = re.search(r'[-+]?\d*\.?\d+', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        
        return None


@dataclass
class DXFCircle:
    """DXF圆形实体"""
    center: Point2D
    radius: float
    layer: str = ""
    
    @property
    def diameter(self) -> float:
        return self.radius * 2


@dataclass
class DXFLine:
    """DXF线段实体"""
    start: Point2D
    end: Point2D
    layer: str = ""
    
    @property
    def length(self) -> float:
        return math.sqrt((self.end.x - self.start.x)**2 + (self.end.y - self.start.y)**2)


@dataclass
class DXFArc:
    """DXF圆弧实体"""
    center: Point2D
    radius: float
    start_angle: float  # 度
    end_angle: float  # 度
    layer: str = ""


@dataclass
class DXFPolyline:
    """DXF多段线实体"""
    points: List[Point2D]
    is_closed: bool = False
    layer: str = ""


# =============================================================================
# DXF解析器
# =============================================================================

class DXFParser:
    """DXF文件解析器"""
    
    def __init__(self):
        self.doc = None
        self.entities = []
        self.dimensions = []
        self.layers = set()
    
    def load(self, filepath: str) -> bool:
        """加载DXF文件"""
        if not EZDXF_AVAILABLE:
            raise ImportError("ezdxf库未安装，无法解析DXF文件")
        
        try:
            self.doc = ezdxf.readfile(filepath)
            return True
        except Exception as e:
            raise Exception(f"加载DXF文件失败: {e}")
    
    def parse_all(self) -> Dict:
        """解析所有内容"""
        if self.doc is None:
            raise Exception("请先加载DXF文件")
        
        result = {
            'entities': self.parse_entities(),
            'dimensions': self.parse_dimensions(),
            'layers': list(self.layers)
        }
        
        return result
    
    def parse_entities(self) -> Dict:
        """解析几何实体"""
        if self.doc is None:
            raise Exception("请先加载DXF文件")
        
        msp = self.doc.modelspace()
        
        entities = {
            'circles': [],
            'arcs': [],
            'lines': [],
            'polylines': [],
            'ellipses': [],
            'splines': []
        }
        
        for entity in msp:
            self.layers.add(entity.dxf.layer if hasattr(entity.dxf, 'layer') else '0')
            
            try:
                if entity.dxftype() == 'CIRCLE':
                    entities['circles'].append(self._parse_circle(entity))
                
                elif entity.dxftype() == 'ARC':
                    entities['arcs'].append(self._parse_arc(entity))
                
                elif entity.dxftype() == 'LINE':
                    entities['lines'].append(self._parse_line(entity))
                
                elif entity.dxftype() == 'LWPOLYLINE':
                    entities['polylines'].append(self._parse_polyline(entity))
                
                elif entity.dxftype() == 'POLYLINE':
                    entities['polylines'].append(self._parse_polyline_2d(entity))
                
                elif entity.dxftype() == 'ELLIPSE':
                    entities['ellipses'].append(self._parse_ellipse(entity))
                
                elif entity.dxftype() == 'SPLINE':
                    entities['splines'].append(self._parse_spline(entity))
            
            except Exception as e:
                print(f"解析实体失败 ({entity.dxftype()}): {e}")
                continue
        
        self.entities = entities
        return entities
    
    def parse_dimensions(self) -> List[DXFDimension]:
        """解析尺寸标注"""
        if self.doc is None:
            raise Exception("请先加载DXF文件")
        
        msp = self.doc.modelspace()
        dimensions = []
        
        for entity in msp:
            try:
                if entity.dxftype() == 'DIMENSION':
                    dimension = self._parse_dimension(entity)
                    if dimension:
                        dimensions.append(dimension)
            
            except Exception as e:
                print(f"解析尺寸标注失败: {e}")
                continue
        
        self.dimensions = dimensions
        return dimensions
    
    def _parse_circle(self, entity) -> DXFCircle:
        """解析圆形"""
        return DXFCircle(
            center=Point2D(entity.dxf.center.x, entity.dxf.center.y),
            radius=entity.dxf.radius,
            layer=entity.dxf.layer if hasattr(entity.dxf, 'layer') else '0'
        )
    
    def _parse_arc(self, entity) -> DXFArc:
        """解析圆弧"""
        return DXFArc(
            center=Point2D(entity.dxf.center.x, entity.dxf.center.y),
            radius=entity.dxf.radius,
            start_angle=math.degrees(entity.dxf.start_angle),
            end_angle=math.degrees(entity.dxf.end_angle),
            layer=entity.dxf.layer if hasattr(entity.dxf, 'layer') else '0'
        )
    
    def _parse_line(self, entity) -> DXFLine:
        """解析线段"""
        return DXFLine(
            start=Point2D(entity.dxf.start[0], entity.dxf.start[1]),
            end=Point2D(entity.dxf.end[0], entity.dxf.end[1]),
            layer=entity.dxf.layer if hasattr(entity.dxf, 'layer') else '0'
        )
    
    def _parse_polyline(self, entity) -> DXFPolyline:
        """解析轻量多段线（LWPOLYLINE）"""
        points = []
        for point in entity.get_points():
            points.append(Point2D(point[0], point[1]))
        
        return DXFPolyline(
            points=points,
            is_closed=entity.closed if hasattr(entity, 'closed') else False,
            layer=entity.dxf.layer if hasattr(entity.dxf, 'layer') else '0'
        )
    
    def _parse_polyline_2d(self, entity) -> DXFPolyline:
        """解析2D多段线（POLYLINE）"""
        points = []
        for point in entity.points:
            points.append(Point2D(point.dxf.location.x, point.dxf.location.y))
        
        return DXFPolyline(
            points=points,
            is_closed=entity.closed if hasattr(entity, 'closed') else False,
            layer=entity.dxf.layer if hasattr(entity.dxf, 'layer') else '0'
        )
    
    def _parse_ellipse(self, entity) -> Dict:
        """解析椭圆"""
        center = Point2D(entity.dxf.center.x, entity.dxf.center.y)
        major_axis = Point2D(entity.dxf.major_axis.x, entity.dxf.major_axis.y)
        ratio = entity.dxf.ratio
        
        # 计算长轴和短轴
        major_radius = math.sqrt(major_axis.x**2 + major_axis.y**2)
        minor_radius = major_radius * ratio
        
        return {
            'center': center,
            'major_radius': major_radius,
            'minor_radius': minor_radius,
            'start_angle': math.degrees(entity.dxf.start_angle),
            'end_angle': math.degrees(entity.dxf.end_angle),
            'layer': entity.dxf.layer if hasattr(entity.dxf, 'layer') else '0'
        }
    
    def _parse_spline(self, entity) -> Dict:
        """解析样条曲线"""
        control_points = []
        for point in entity.control_points:
            control_points.append(Point2D(point[0], point[1]))
        
        return {
            'control_points': control_points,
            'degree': entity.dxf.degree,
            'knots': entity.knots if hasattr(entity, 'knots') else [],
            'layer': entity.dxf.layer if hasattr(entity.dxf, 'layer') else '0'
        }
    
    def _parse_dimension(self, entity) -> Optional[DXFDimension]:
        """解析尺寸标注"""
        try:
            dim_type_code = entity.dxf.dimtype if hasattr(entity.dxf, 'dimtype') else 0
            
            # 确定标注类型
            if dim_type_code in [0, 1]:  # 旋转、对齐
                dim_type = 'linear'
            elif dim_type_code in [2]:  # 角度
                dim_type = 'angular'
            elif dim_type_code in [3]:  # 直径
                dim_type = 'diameter'
            elif dim_type_code in [4]:  # 半径
                dim_type = 'radius'
            else:
                dim_type = 'linear'
            
            # 获取标注文本
            text = ""
            if hasattr(entity, 'text'):
                text = entity.text
            elif hasattr(entity.dxf, 'text'):
                text = entity.dxf.text
            
            # 提取标注点
            points = []
            if hasattr(entity, 'defpoint'):
                defpoint = entity.defpoint
                points.append(Point2D(defpoint.x, defpoint.y))
            
            if hasattr(entity, 'insert'):
                insert = entity.insert
                points.append(Point2D(insert.x, insert.y))
            
            dimension = DXFDimension(
                dim_type=dim_type,
                text=text,
                points=points,
                layer=entity.dxf.layer if hasattr(entity.dxf, 'layer') else '0'
            )
            
            return dimension
        
        except Exception as e:
            print(f"解析尺寸标注失败: {e}")
            return None


# =============================================================================
# DXF到标注模板转换器
# =============================================================================

class DXFToTemplateConverter:
    """DXF文件到标注模板的转换器"""
    
    def __init__(self, dxf_parser: DXFParser):
        self.parser = dxf_parser
        self.template = None
    
    def convert(self, template_name: str = "DXF模板",
                tolerance_standard: ToleranceStandard = ToleranceStandard.IT8,
                auto_extract_dimensions: bool = True,
                auto_identify_features: bool = True) -> InspectionTemplate:
        """将DXF文件转换为标注模板"""
        
        # 创建模板
        self.template = create_default_template(template_name)
        self.template.tolerance_standard = tolerance_standard
        
        # 解析DXF内容
        entities = self.parser.parse_entities()
        dimensions = self.parser.parse_dimensions()
        
        # 方法1: 从尺寸标注自动提取
        if auto_extract_dimensions:
            self._extract_from_dimensions(dimensions)
        
        # 方法2: 自动识别几何特征
        if auto_identify_features:
            self._identify_features_from_entities(entities)
        
        return self.template
    
    def _extract_from_dimensions(self, dimensions: List[DXFDimension]):
        """从尺寸标注中提取检测特征"""
        
        for dim in dimensions:
            value = dim.extract_value()
            if value is None:
                continue
            
            import uuid
            
            if dim.dim_type == 'diameter':
                # 直径标注 - 寻找关联的圆
                associated_circle = self._find_associated_circle(dim, value)
                
                if associated_circle:
                    annotation = CircleAnnotation(
                        id=str(uuid.uuid4()),
                        center=associated_circle.center,
                        radius=associated_circle.radius,
                        feature_type=FeatureType.DIAMETER,
                        nominal_value=value,
                        tolerance=self.template.tolerance_standard,
                        description=f"直径标注: {dim.text}"
                    )
                    self.template.add_annotation(annotation)
            
            elif dim.dim_type == 'radius':
                # 半径标注 - 寻找关联的圆或圆弧
                associated_arc = self._find_associated_arc(dim, value)
                
                if associated_arc:
                    annotation = CircleAnnotation(
                        id=str(uuid.uuid4()),
                        center=associated_arc.center,
                        radius=associated_arc.radius,
                        feature_type=FeatureType.RADIUS,
                        nominal_value=value,
                        tolerance=self.template.tolerance_standard,
                        description=f"半径标注: {dim.text}"
                    )
                    self.template.add_annotation(annotation)
            
            elif dim.dim_type == 'linear':
                # 线性标注 - 寻找关联的线段
                associated_line = self._find_associated_line(dim, value)
                
                if associated_line:
                    # 判断是长度、宽度还是高度
                    feature_type = self._determine_linear_feature_type(associated_line)
                    
                    annotation = LineAnnotation(
                        id=str(uuid.uuid4()),
                        start=associated_line.start,
                        end=associated_line.end,
                        feature_type=feature_type,
                        nominal_value=value,
                        tolerance=self.template.tolerance_standard,
                        description=f"线性标注: {dim.text}"
                    )
                    self.template.add_annotation(annotation)
    
    def _identify_features_from_entities(self, entities: Dict):
        """从几何实体中自动识别检测特征"""
        
        import uuid
        
        # 识别圆形特征
        for circle in entities.get('circles', []):
            # 检查是否已经有该圆的标注
            if not self._has_circle_annotation(circle):
                annotation = CircleAnnotation(
                    id=str(uuid.uuid4()),
                    center=circle.center,
                    radius=circle.radius,
                    feature_type=FeatureType.DIAMETER,
                    nominal_value=None,  # 需要用户输入
                    tolerance=None,
                    description=f"圆形特征 (图层: {circle.layer})"
                )
                self.template.add_annotation(annotation)
        
        # 识别线段特征
        for line in entities.get('lines', []):
            # 检查是否已经有该线段的标注
            if not self._has_line_annotation(line):
                feature_type = self._determine_linear_feature_type(line)
                
                annotation = LineAnnotation(
                    id=str(uuid.uuid4()),
                    start=line.start,
                    end=line.end,
                    feature_type=feature_type,
                    nominal_value=None,  # 需要用户输入
                    tolerance=None,
                    description=f"线段特征 (图层: {line.layer})"
                )
                self.template.add_annotation(annotation)
    
    def _find_associated_circle(self, dimension: DXFDimension, 
                                nominal_value: float) -> Optional[DXFCircle]:
        """寻找与直径标注关联的圆"""
        
        if not hasattr(self.parser, 'entities') or 'circles' not in self.parser.entities:
            return None
        
        circles = self.parser.entities.get('circles', [])
        
        # 方法1: 根据标注点附近的圆
        if dimension.points:
            for point in dimension.points:
                for circle in circles:
                    distance = math.sqrt((point.x - circle.center.x)**2 + 
                                       (point.y - circle.center.y)**2)
                    if distance < circle.radius * 1.2:
                        # 检查直径是否匹配
                        if abs(circle.diameter - nominal_value) < nominal_value * 0.1:
                            return circle
        
        # 方法2: 根据直径匹配
        for circle in circles:
            if abs(circle.diameter - nominal_value) < nominal_value * 0.05:
                return circle
        
        return None
    
    def _find_associated_arc(self, dimension: DXFDimension, 
                            nominal_value: float) -> Optional[DXFArc]:
        """寻找与半径标注关联的圆弧"""
        
        if not hasattr(self.parser, 'entities') or 'arcs' not in self.parser.entities:
            return None
        
        arcs = self.parser.entities.get('arcs', [])
        
        # 根据标注点附近的圆弧
        if dimension.points:
            for point in dimension.points:
                for arc in arcs:
                    distance = math.sqrt((point.x - arc.center.x)**2 + 
                                       (point.y - arc.center.y)**2)
                    if distance < arc.radius * 1.2:
                        # 检查半径是否匹配
                        if abs(arc.radius - nominal_value) < nominal_value * 0.1:
                            return arc
        
        # 根据半径匹配
        for arc in arcs:
            if abs(arc.radius - nominal_value) < nominal_value * 0.05:
                return arc
        
        return None
    
    def _find_associated_line(self, dimension: DXFDimension, 
                             nominal_value: float) -> Optional[DXFLine]:
        """寻找与线性标注关联的线段"""
        
        if not hasattr(self.parser, 'entities') or 'lines' not in self.parser.entities:
            return None
        
        lines = self.parser.entities.get('lines', [])
        
        # 根据标注点附近的线段
        if dimension.points:
            for point in dimension.points:
                for line in lines:
                    # 检查点是否在线段附近
                    if self._is_point_near_line(point, line, tolerance=5.0):
                        # 检查长度是否匹配
                        if abs(line.length - nominal_value) < nominal_value * 0.1:
                            return line
        
        # 根据长度匹配
        for line in lines:
            if abs(line.length - nominal_value) < nominal_value * 0.05:
                return line
        
        return None
    
    def _is_point_near_line(self, point: Point2D, line: DXFLine, 
                           tolerance: float = 5.0) -> bool:
        """判断点是否在线段附近"""
        
        # 计算点到线段的距离
        line_vec = Point2D(line.end.x - line.start.x, line.end.y - line.start.y)
        point_vec = Point2D(point.x - line.start.x, point.y - line.start.y)
        
        line_length = line.length
        if line_length == 0:
            return False
        
        # 投影
        projection = (point_vec.x * line_vec.x + point_vec.y * line_vec.y) / line_length
        
        if projection < 0:
            # 点在线段起点之前
            distance = math.sqrt((point.x - line.start.x)**2 + (point.y - line.start.y)**2)
        elif projection > line_length:
            # 点在线段终点之后
            distance = math.sqrt((point.x - line.end.x)**2 + (point.y - line.end.y)**2)
        else:
            # 点在线段上
            closest_point = Point2D(
                line.start.x + line_vec.x * projection / line_length,
                line.start.y + line_vec.y * projection / line_length
            )
            distance = math.sqrt((point.x - closest_point.x)**2 + (point.y - closest_point.y)**2)
        
        return distance <= tolerance
    
    def _determine_linear_feature_type(self, line: DXFLine) -> FeatureType:
        """确定线性特征类型"""
        
        dx = abs(line.end.x - line.start.x)
        dy = abs(line.end.y - line.start.y)
        
        # 根据方向判断
        if dx > dy * 2:
            return FeatureType.LENGTH  # 水平方向
        elif dy > dx * 2:
            return FeatureType.HEIGHT  # 垂直方向
        else:
            return FeatureType.WIDTH  # 斜向
    
    def _has_circle_annotation(self, circle: DXFCircle) -> bool:
        """检查是否已经有该圆的标注"""
        
        for annotation in self.template.annotations:
            if isinstance(annotation, CircleAnnotation):
                distance = math.sqrt((annotation.center.x - circle.center.x)**2 + 
                                   (annotation.center.y - circle.center.y)**2)
                radius_diff = abs(annotation.radius - circle.radius)
                
                if distance < 5.0 and radius_diff < 2.0:
                    return True
        
        return False
    
    def _has_line_annotation(self, line: DXFLine) -> bool:
        """检查是否已经有该线段的标注"""
        
        for annotation in self.template.annotations:
            if isinstance(annotation, LineAnnotation):
                # 检查起点和终点是否接近
                start_dist = math.sqrt((annotation.start.x - line.start.x)**2 + 
                                      (annotation.start.y - line.start.y)**2)
                end_dist = math.sqrt((annotation.end.x - line.end.x)**2 + 
                                    (annotation.end.y - line.end.y)**2)
                
                if start_dist < 5.0 and end_dist < 5.0:
                    return True
        
        return False


# =============================================================================
# 工厂函数
# =============================================================================

def parse_dxf_file(filepath: str) -> Optional[Dict]:
    """解析DXF文件（便捷函数）"""
    
    try:
        parser = DXFParser()
        parser.load(filepath)
        return parser.parse_all()
    
    except Exception as e:
        print(f"解析DXF文件失败: {e}")
        return None


def dxf_to_template(filepath: str, 
                    template_name: str = "DXF模板",
                    tolerance_standard: str = "IT8",
                    auto_extract_dimensions: bool = True,
                    auto_identify_features: bool = True) -> Optional[InspectionTemplate]:
    """将DXF文件转换为标注模板（便捷函数）"""
    
    try:
        parser = DXFParser()
        parser.load(filepath)
        
        converter = DXFToTemplateConverter(parser)
        
        tolerance_std = ToleranceStandard(tolerance_standard)
        
        template = converter.convert(
            template_name=template_name,
            tolerance_standard=tolerance_std,
            auto_extract_dimensions=auto_extract_dimensions,
            auto_identify_features=auto_identify_features
        )
        
        return template
    
    except Exception as e:
        print(f"DXF转模板失败: {e}")
        return None


def extract_dimensions_from_dxf(filepath: str) -> List[Dict]:
    """从DXF文件中提取尺寸标注（便捷函数）"""
    
    try:
        parser = DXFParser()
        parser.load(filepath)
        
        dimensions = parser.parse_dimensions()
        
        result = []
        for dim in dimensions:
            result.append({
                'type': dim.dim_type,
                'text': dim.text,
                'value': dim.extract_value(),
                'unit': dim.unit,
                'layer': dim.layer,
                'points': [(p.x, p.y) for p in dim.points]
            })
        
        return result
    
    except Exception as e:
        print(f"提取尺寸标注失败: {e}")
        return []


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    # 测试DXF解析
    import sys
    
    if len(sys.argv) > 1:
        dxf_file = sys.argv[1]
        
        print(f"解析DXF文件: {dxf_file}")
        
        # 解析文件
        result = parse_dxf_file(dxf_file)
        
        if result:
            print("\n=== 解析结果 ===")
            print(f"图层数量: {len(result['layers'])}")
            print(f"圆形数量: {len(result['entities']['circles'])}")
            print(f"线段数量: {len(result['entities']['lines'])}")
            print(f"尺寸标注数量: {len(result['dimensions'])}")
            
            print("\n=== 尺寸标注 ===")
            for dim in result['dimensions']:
                value = dim.extract_value()
                print(f"{dim.dim_type}: {dim.text} -> {value}")
            
            # 转换为模板
            print("\n=== 转换为模板 ===")
            template = dxf_to_template(dxf_file, "测试模板")
            
            if template:
                print(f"模板名称: {template.name}")
                print(f"标注数量: {len(template.annotations)}")
                
                for anno in template.annotations:
                    print(f"  - {anno.feature_type.value}: {anno.nominal_value}")
    else:
        print("使用方法: python dxf_parser.py <dxf文件路径>")