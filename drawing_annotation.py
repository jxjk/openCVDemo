# -*- coding: utf-8 -*-
"""
图纸标注模块
Drawing Annotation Module

功能:
- 2D图纸解析（DXF/SVG/PDF/图片）
- 检测部位标注
- 标注模板管理
- 基于标注的检测
"""

import os
import json
import math
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import cv2


# =============================================================================
# 枚举类型
# =============================================================================

class FeatureType(Enum):
    """特征类型"""
    DIAMETER = "diameter"      # 直径
    RADIUS = "radius"          # 半径
    LENGTH = "length"          # 长度
    WIDTH = "width"            # 宽度
    HEIGHT = "height"          # 高度
    DISTANCE = "distance"      # 距离
    ANGLE = "angle"            # 角度


class AnnotationType(Enum):
    """标注类型"""
    CAD_AUTO = "cad_auto"           # CAD文件自动解析
    IMAGE_MANUAL = "image_manual"   # 图像手动标注


class ToleranceStandard(Enum):
    """公差标准"""
    IT5 = "IT5"
    IT7 = "IT7"
    IT8 = "IT8"
    IT9 = "IT9"
    IT11 = "IT11"


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class Point2D:
    """2D点"""
    x: float
    y: float
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def distance_to(self, other: 'Point2D') -> float:
        """计算到另一个点的距离"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class BoundingBox:
    """边界框"""
    x: float
    y: float
    width: float
    height: float
    
    def center(self) -> Point2D:
        """获取中心点"""
        return Point2D(self.x + self.width / 2, self.y + self.height / 2)
    
    def contains(self, point: Point2D) -> bool:
        """判断点是否在边界框内"""
        return (self.x <= point.x <= self.x + self.width and
                self.y <= point.y <= self.y + self.height)


@dataclass
class CircleAnnotation:
    """圆形标注"""
    id: str
    center: Point2D
    radius: float
    feature_type: FeatureType = FeatureType.DIAMETER
    nominal_value: Optional[float] = None
    tolerance: Optional[float] = None
    tolerance_standard: ToleranceStandard = ToleranceStandard.IT8
    description: str = ""
    
    def to_dict(self) -> dict:
        """转换为字典"""
        data = asdict(self)
        data['center'] = self.center.to_tuple()
        data['feature_type'] = self.feature_type.value
        data['tolerance_standard'] = self.tolerance_standard.value
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CircleAnnotation':
        """从字典创建"""
        data['center'] = Point2D(*data['center'])
        data['feature_type'] = FeatureType(data['feature_type'])
        data['tolerance_standard'] = ToleranceStandard(data['tolerance_standard'])
        return cls(**data)


@dataclass
class LineAnnotation:
    """线段标注"""
    id: str
    start: Point2D
    end: Point2D
    feature_type: FeatureType = FeatureType.LENGTH
    nominal_value: Optional[float] = None
    tolerance: Optional[float] = None
    tolerance_standard: ToleranceStandard = ToleranceStandard.IT8
    description: str = ""
    
    @property
    def length(self) -> float:
        """计算线段长度"""
        return self.start.distance_to(self.end)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        data = asdict(self)
        data['start'] = self.start.to_tuple()
        data['end'] = self.end.to_tuple()
        data['feature_type'] = self.feature_type.value
        data['tolerance_standard'] = self.tolerance_standard.value
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'LineAnnotation':
        """从字典创建"""
        data['start'] = Point2D(*data['start'])
        data['end'] = Point2D(*data['end'])
        data['feature_type'] = FeatureType(data['feature_type'])
        data['tolerance_standard'] = ToleranceStandard(data['tolerance_standard'])
        return cls(**data)


@dataclass
class RectangleAnnotation:
    """矩形标注"""
    id: str
    bbox: BoundingBox
    feature_type: FeatureType = FeatureType.WIDTH
    nominal_value: Optional[float] = None
    tolerance: Optional[float] = None
    tolerance_standard: ToleranceStandard = ToleranceStandard.IT8
    description: str = ""
    
    @property
    def width(self) -> float:
        """获取宽度"""
        return self.bbox.width
    
    @property
    def height(self) -> float:
        """获取高度"""
        return self.bbox.height
    
    def to_dict(self) -> dict:
        """转换为字典"""
        data = asdict(self)
        data['feature_type'] = self.feature_type.value
        data['tolerance_standard'] = self.tolerance_standard.value
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'RectangleAnnotation':
        """从字典创建"""
        data['bbox'] = BoundingBox(**data['bbox'])
        data['feature_type'] = FeatureType(data['feature_type'])
        data['tolerance_standard'] = ToleranceStandard(data['tolerance_standard'])
        return cls(**data)


@dataclass
class AngleAnnotation:
    """角度标注"""
    id: str
    vertex: Point2D
    start_point: Point2D
    end_point: Point2D
    feature_type: FeatureType = FeatureType.ANGLE
    nominal_value: Optional[float] = None
    tolerance: Optional[float] = None
    tolerance_standard: ToleranceStandard = ToleranceStandard.IT8
    description: str = ""
    
    @property
    def angle(self) -> float:
        """计算角度（度）"""
        # 计算两个向量
        v1 = np.array([self.start_point.x - self.vertex.x, self.start_point.y - self.vertex.y])
        v2 = np.array([self.end_point.x - self.vertex.x, self.end_point.y - self.vertex.y])
        
        # 计算夹角
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        return np.degrees(angle)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        data = asdict(self)
        data['vertex'] = self.vertex.to_tuple()
        data['start_point'] = self.start_point.to_tuple()
        data['end_point'] = self.end_point.to_tuple()
        data['feature_type'] = self.feature_type.value
        data['tolerance_standard'] = self.tolerance_standard.value
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AngleAnnotation':
        """从字典创建"""
        data['vertex'] = Point2D(*data['vertex'])
        data['start_point'] = Point2D(*data['start_point'])
        data['end_point'] = Point2D(*data['end_point'])
        data['feature_type'] = FeatureType(data['feature_type'])
        data['tolerance_standard'] = ToleranceStandard(data['tolerance_standard'])
        return cls(**data)


@dataclass
class InspectionTemplate:
    """检测模板"""
    name: str
    version: str = "1.0"
    created_date: str = ""
    updated_date: str = ""
    image_scale: float = 1.0  # 图像缩放比例（像素/毫米）
    tolerance_standard: ToleranceStandard = ToleranceStandard.IT8
    annotations: List = field(default_factory=list)
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.created_date:
            from datetime import datetime
            self.created_date = datetime.now().isoformat()
        if not self.updated_date:
            from datetime import datetime
            self.updated_date = datetime.now().isoformat()
    
    def add_annotation(self, annotation):
        """添加标注"""
        self.annotations.append(annotation)
        self._update_timestamp()
    
    def remove_annotation(self, annotation_id: str):
        """移除标注"""
        self.annotations = [a for a in self.annotations if a.id != annotation_id]
        self._update_timestamp()
    
    def get_annotation(self, annotation_id: str):
        """获取标注"""
        for annotation in self.annotations:
            if annotation.id == annotation_id:
                return annotation
        return None
    
    def get_annotations_by_type(self, feature_type: FeatureType) -> List:
        """根据类型获取标注"""
        return [a for a in self.annotations if a.feature_type == feature_type]
    
    def _update_timestamp(self):
        """更新时间戳"""
        from datetime import datetime
        self.updated_date = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        """转换为字典"""
        data = asdict(self)
        data['tolerance_standard'] = self.tolerance_standard.value
        data['annotations'] = [a.to_dict() for a in self.annotations]
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'InspectionTemplate':
        """从字典创建"""
        data['tolerance_standard'] = ToleranceStandard(data['tolerance_standard'])
        
        # 重建标注对象
        annotations = []
        for anno_data in data['annotations']:
            feature_type = FeatureType(anno_data['feature_type'])
            
            if feature_type in [FeatureType.DIAMETER, FeatureType.RADIUS]:
                annotations.append(CircleAnnotation.from_dict(anno_data))
            elif feature_type in [FeatureType.LENGTH, FeatureType.WIDTH, FeatureType.HEIGHT, FeatureType.DISTANCE]:
                annotations.append(LineAnnotation.from_dict(anno_data))
            elif feature_type == FeatureType.ANGLE:
                annotations.append(AngleAnnotation.from_dict(anno_data))
        
        data['annotations'] = annotations
        return cls(**data)
    
    def save(self, filepath: str):
        """保存模板到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'InspectionTemplate':
        """从文件加载模板"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


# =============================================================================
# 图纸解析器
# =============================================================================

class DrawingParser:
    """图纸解析器"""
    
    @staticmethod
    def parse_dxf(dxf_path: str) -> List[Dict]:
        """解析DXF文件"""
        try:
            import ezdxf
            
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            
            entities = []
            
            for entity in msp:
                entity_data = {}
                
                if entity.dxftype() == 'CIRCLE':
                    entity_data = {
                        'type': 'circle',
                        'center': (entity.dxf.center.x, entity.dxf.center.y),
                        'radius': entity.dxf.radius,
                        'layer': entity.dxf.layer
                    }
                elif entity.dxftype() == 'LINE':
                    entity_data = {
                        'type': 'line',
                        'start': (entity.dxf.start[0], entity.dxf.start[1]),
                        'end': (entity.dxf.end[0], entity.dxf.end[1]),
                        'layer': entity.dxf.layer
                    }
                elif entity.dxftype() == 'ARC':
                    entity_data = {
                        'type': 'arc',
                        'center': (entity.dxf.center.x, entity.dxf.center.y),
                        'radius': entity.dxf.radius,
                        'start_angle': entity.dxf.start_angle,
                        'end_angle': entity.dxf.end_angle,
                        'layer': entity.dxf.layer
                    }
                elif entity.dxftype() == 'LWPOLYLINE':
                    points = list(entity.get_points())
                    entity_data = {
                        'type': 'polyline',
                        'points': [(p[0], p[1]) for p in points],
                        'layer': entity.dxf.layer
                    }
                
                if entity_data:
                    entities.append(entity_data)
            
            return entities
            
        except ImportError:
            print("警告: ezdxf库未安装，无法解析DXF文件")
            return []
        except Exception as e:
            print(f"解析DXF文件失败: {e}")
            return []
    
    @staticmethod
    def parse_svg(svg_path: str) -> List[Dict]:
        """解析SVG文件"""
        try:
            import svgpathtools
            
            paths, attributes = svgpathtools.svg2paths(svg_path)
            
            entities = []
            
            for i, (path, attr) in enumerate(zip(paths, attributes)):
                # 获取路径边界框
                bbox = path.bbox()
                
                entities.append({
                    'type': 'path',
                    'id': attr.get('id', f'path_{i}'),
                    'bbox': bbox,
                    'attributes': attr
                })
            
            return entities
            
        except ImportError:
            print("警告: svgpathtools库未安装，无法解析SVG文件")
            return []
        except Exception as e:
            print(f"解析SVG文件失败: {e}")
            return []
    
    @staticmethod
    def extract_dimensions(dxf_path: str) -> List[Dict]:
        """提取尺寸标注"""
        try:
            import ezdxf
            
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            
            dimensions = []
            
            for entity in msp:
                if entity.dxftype() == 'DIMENSION':
                    dim_data = {
                        'type': 'dimension',
                        'dimension_type': entity.dxf.dimtype,
                        'text': entity.dxf.text if hasattr(entity, 'dxf') else '',
                        'layer': entity.dxf.layer
                    }
                    dimensions.append(dim_data)
            
            return dimensions
            
        except ImportError:
            print("警告: ezdxf库未安装，无法提取尺寸标注")
            return []
        except Exception as e:
            print(f"提取尺寸标注失败: {e}")
            return []


# =============================================================================
# 标注工具
# =============================================================================

class AnnotationTool:
    """标注工具"""
    
    def __init__(self, template: InspectionTemplate = None):
        self.template = template or InspectionTemplate(name="默认模板")
        self.current_annotation = None
        self.drawing_points = []
    
    def add_circle(self, center: Point2D, radius: float, 
                   feature_type: FeatureType = FeatureType.DIAMETER) -> CircleAnnotation:
        """添加圆形标注"""
        import uuid
        annotation = CircleAnnotation(
            id=str(uuid.uuid4()),
            center=center,
            radius=radius,
            feature_type=feature_type
        )
        self.template.add_annotation(annotation)
        return annotation
    
    def add_line(self, start: Point2D, end: Point2D,
                 feature_type: FeatureType = FeatureType.LENGTH) -> LineAnnotation:
        """添加线段标注"""
        import uuid
        annotation = LineAnnotation(
            id=str(uuid.uuid4()),
            start=start,
            end=end,
            feature_type=feature_type
        )
        self.template.add_annotation(annotation)
        return annotation
    
    def add_rectangle(self, bbox: BoundingBox,
                      feature_type: FeatureType = FeatureType.WIDTH) -> RectangleAnnotation:
        """添加矩形标注"""
        import uuid
        annotation = RectangleAnnotation(
            id=str(uuid.uuid4()),
            bbox=bbox,
            feature_type=feature_type
        )
        self.template.add_annotation(annotation)
        return annotation
    
    def add_angle(self, vertex: Point2D, start_point: Point2D, end_point: Point2D) -> AngleAnnotation:
        """添加角度标注"""
        import uuid
        annotation = AngleAnnotation(
            id=str(uuid.uuid4()),
            vertex=vertex,
            start_point=start_point,
            end_point=end_point
        )
        self.template.add_annotation(annotation)
        return annotation
    
    def remove_annotation(self, annotation_id: str):
        """移除标注"""
        self.template.remove_annotation(annotation_id)
    
    def update_annotation_value(self, annotation_id: str, nominal_value: float, 
                               tolerance: float = None):
        """更新标注的标称值和公差"""
        annotation = self.template.get_annotation(annotation_id)
        if annotation:
            annotation.nominal_value = nominal_value
            if tolerance is not None:
                annotation.tolerance = tolerance
            self.template._update_timestamp()


# =============================================================================
# 基于标注的检测器
# =============================================================================

class AnnotationBasedInspector:
    """基于标注的检测器"""
    
    def __init__(self, template: InspectionTemplate, config):
        self.template = template
        self.config = config
    
    def inspect_from_annotations(self, image: np.ndarray, 
                                part_id: str = "") -> List[Dict]:
        """根据标注进行检测"""
        results = []
        
        for annotation in self.template.annotations:
            result = self._inspect_single_annotation(image, annotation, part_id)
            if result:
                results.append(result)
        
        return results
    
    def _inspect_single_annotation(self, image: np.ndarray, annotation, 
                                  part_id: str) -> Optional[Dict]:
        """检测单个标注"""
        from inspection_system import InspectionEngine, SubpixelDetector
        
        engine = InspectionEngine(self.config)
        
        # 图像预处理
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        if isinstance(annotation, CircleAnnotation):
            # 圆形检测
            result = self._detect_circle(image, annotation, part_id)
        elif isinstance(annotation, LineAnnotation):
            # 线段检测
            result = self._detect_line(image, annotation, part_id)
        elif isinstance(annotation, RectangleAnnotation):
            # 矩形检测
            result = self._detect_rectangle(image, annotation, part_id)
        elif isinstance(annotation, AngleAnnotation):
            # 角度检测
            result = self._detect_angle(image, annotation, part_id)
        else:
            result = None
        
        return result
    
    def _detect_circle(self, image: np.ndarray, annotation: CircleAnnotation,
                      part_id: str) -> Dict:
        """检测圆形"""
        # 在标注区域附近进行检测
        roi_center = annotation.center
        roi_radius = annotation.radius * 1.5
        
        # 提取ROI
        x1 = max(0, int(roi_center.x - roi_radius))
        y1 = max(0, int(roi_center.y - roi_radius))
        x2 = min(image.shape[1], int(roi_center.x + roi_radius))
        y2 = min(image.shape[0], int(roi_center.y + roi_radius))
        
        roi = image[y1:y2, x1:x2]
        
        # 检测圆
        from inspection_system import InspectionEngine
        engine = InspectionEngine(self.config)
        
        # 这里简化处理，实际应该使用更精确的检测方法
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, min(roi.shape[:2]) / 8,
            param1=50, param2=30, minRoiRadius=int(annotation.radius * 0.5),
            maxRoiRadius=int(annotation.radius * 1.5)
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            x, y, r = circles[0]
            
            # 转换为全局坐标
            global_x = x + x1
            global_y = y + y1
            
            # 计算直径
            diameter_pixel = r * 2
            diameter_mm = diameter_pixel * self.config.PIXEL_TO_MM
            
            # 公差判断
            is_qualified = True
            deviation = 0
            
            if annotation.nominal_value is not None:
                tolerance = annotation.tolerance or self.config.get_it8_tolerance(annotation.nominal_value)
                deviation = diameter_mm - annotation.nominal_value
                is_qualified = abs(deviation) <= tolerance
            
            return {
                'annotation_id': annotation.id,
                'feature_type': annotation.feature_type.value,
                'measured_value': diameter_mm,
                'nominal_value': annotation.nominal_value,
                'deviation': deviation,
                'tolerance': annotation.tolerance,
                'is_qualified': is_qualified,
                'center': (global_x, global_y),
                'radius_pixel': r
            }
        
        return None
    
    def _detect_line(self, image: np.ndarray, annotation: LineAnnotation,
                    part_id: str) -> Dict:
        """检测线段"""
        # 在标注区域附近进行检测
        roi_size = 50
        
        x1 = max(0, int(annotation.start.x - roi_size))
        y1 = max(0, int(annotation.start.y - roi_size))
        x2 = min(image.shape[1], int(annotation.end.x + roi_size))
        y2 = min(image.shape[0], int(annotation.end.y + roi_size))
        
        roi = image[y1:y2, x1:x2]
        
        # 边缘检测
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        edges = cv2.Canny(gray, 50, 150)
        
        # 霍夫直线检测
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                               minLineLength=20, maxLineGap=10)
        
        if lines is not None:
            # 找到最接近标注线的直线
            best_line = None
            min_distance = float('inf')
            
            for line in lines:
                x1_line, y1_line, x2_line, y2_line = line[0]
                
                # 计算与标注线的距离
                center_line = Point2D((x1_line + x2_line) / 2 + x1, (y1_line + y2_line) / 2 + y1)
                distance = center_line.distance_to(Point2D((annotation.start.x + annotation.end.x) / 2,
                                                            (annotation.start.y + annotation.end.y) / 2))
                
                if distance < min_distance:
                    min_distance = distance
                    best_line = line
            
            if best_line is not None:
                x1_line, y1_line, x2_line, y2_line = best_line[0]
                
                # 转换为全局坐标
                global_x1 = x1_line + x1
                global_y1 = y1_line + y1
                global_x2 = x2_line + x1
                global_y2 = y2_line + y1
                
                # 计算长度
                length_pixel = math.sqrt((global_x2 - global_x1)**2 + (global_y2 - global_y1)**2)
                length_mm = length_pixel * self.config.PIXEL_TO_MM
                
                # 公差判断
                is_qualified = True
                deviation = 0
                
                if annotation.nominal_value is not None:
                    tolerance = annotation.tolerance or self.config.get_it8_tolerance(annotation.nominal_value)
                    deviation = length_mm - annotation.nominal_value
                    is_qualified = abs(deviation) <= tolerance
                
                return {
                    'annotation_id': annotation.id,
                    'feature_type': annotation.feature_type.value,
                    'measured_value': length_mm,
                    'nominal_value': annotation.nominal_value,
                    'deviation': deviation,
                    'tolerance': annotation.tolerance,
                    'is_qualified': is_qualified,
                    'start': (global_x1, global_y1),
                    'end': (global_x2, global_y2),
                    'length_pixel': length_pixel
                }
        
        return None
    
    def _detect_rectangle(self, image: np.ndarray, annotation: RectangleAnnotation,
                         part_id: str) -> Dict:
        """检测矩形"""
        # 简化处理，检测边界框
        roi_size = 20
        
        x1 = max(0, int(annotation.bbox.x - roi_size))
        y1 = max(0, int(annotation.bbox.y - roi_size))
        x2 = min(image.shape[1], int(annotation.bbox.x + annotation.bbox.width + roi_size))
        y2 = min(image.shape[0], int(annotation.bbox.y + annotation.bbox.height + roi_size))
        
        roi = image[y1:y2, x1:x2]
        
        # 边缘检测和轮廓查找
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            
            # 转换为全局坐标
            center_x, center_y = rect[0]
            width, height = rect[1]
            angle = rect[2]
            
            global_center_x = center_x + x1
            global_center_y = center_y + y1
            
            # 计算实际尺寸
            width_mm = width * self.config.PIXEL_TO_MM
            height_mm = height * self.config.PIXEL_TO_MM
            
            # 根据特征类型选择测量值
            if annotation.feature_type == FeatureType.WIDTH:
                measured_value = width_mm
            elif annotation.feature_type == FeatureType.HEIGHT:
                measured_value = height_mm
            else:
                measured_value = max(width_mm, height_mm)
            
            # 公差判断
            is_qualified = True
            deviation = 0
            
            if annotation.nominal_value is not None:
                tolerance = annotation.tolerance or self.config.get_it8_tolerance(annotation.nominal_value)
                deviation = measured_value - annotation.nominal_value
                is_qualified = abs(deviation) <= tolerance
            
            return {
                'annotation_id': annotation.id,
                'feature_type': annotation.feature_type.value,
                'measured_value': measured_value,
                'nominal_value': annotation.nominal_value,
                'deviation': deviation,
                'tolerance': annotation.tolerance,
                'is_qualified': is_qualified,
                'center': (global_center_x, global_center_y),
                'width_pixel': width,
                'height_pixel': height,
                'angle': angle
            }
        
        return None
    
    def _detect_angle(self, image: np.ndarray, annotation: AngleAnnotation,
                     part_id: str) -> Dict:
        """检测角度"""
        # 在顶点附近检测角点
        roi_size = 30
        
        x1 = max(0, int(annotation.vertex.x - roi_size))
        y1 = max(0, int(annotation.vertex.y - roi_size))
        x2 = min(image.shape[1], int(annotation.vertex.x + roi_size))
        y2 = min(image.shape[0], int(annotation.vertex.y + roi_size))
        
        roi = image[y1:y2, x1:x2]
        
        # 角点检测
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=1, qualityLevel=0.01, minDistance=10)
        
        if corners is not None:
            # 亚像素角点细化
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            
            # 转换为全局坐标
            corner_x, corner_y = corners[0][0]
            global_corner_x = corner_x + x1
            global_corner_y = corner_y + y1
            
            # 计算角度（简化处理）
            # 实际应该检测两条边的方向
            measured_angle = annotation.angle  # 使用标注角度作为参考
            
            # 公差判断
            is_qualified = True
            deviation = 0
            
            if annotation.nominal_value is not None:
                tolerance = annotation.tolerance or 1.0  # 默认角度公差±1度
                deviation = measured_angle - annotation.nominal_value
                is_qualified = abs(deviation) <= tolerance
            
            return {
                'annotation_id': annotation.id,
                'feature_type': annotation.feature_type.value,
                'measured_value': measured_angle,
                'nominal_value': annotation.nominal_value,
                'deviation': deviation,
                'tolerance': annotation.tolerance,
                'is_qualified': is_qualified,
                'vertex': (global_corner_x, global_corner_y)
            }
        
        return None
    
    def draw_annotations(self, image: np.ndarray) -> np.ndarray:
        """在图像上绘制标注"""
        result_image = image.copy()
        
        for annotation in self.template.annotations:
            if isinstance(annotation, CircleAnnotation):
                # 绘制圆形标注
                center = (int(annotation.center.x), int(annotation.center.y))
                radius = int(annotation.radius)
                color = (0, 255, 0) if annotation.feature_type == FeatureType.DIAMETER else (0, 0, 255)
                cv2.circle(result_image, center, radius, color, 2)
                cv2.circle(result_image, center, 3, color, -1)
                
                # 添加标注文本
                text = f"{annotation.feature_type.value}"
                if annotation.nominal_value:
                    text += f": {annotation.nominal_value:.3f}mm"
                cv2.putText(result_image, text, center, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            elif isinstance(annotation, LineAnnotation):
                # 绘制线段标注
                start = (int(annotation.start.x), int(annotation.start.y))
                end = (int(annotation.end.x), int(annotation.end.y))
                color = (0, 0, 255) if annotation.feature_type == FeatureType.LENGTH else (255, 0, 0)
                cv2.line(result_image, start, end, color, 2)
                cv2.circle(result_image, start, 3, color, -1)
                cv2.circle(result_image, end, 3, color, -1)
                
                # 添加标注文本
                mid_point = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
                text = f"{annotation.feature_type.value}"
                if annotation.nominal_value:
                    text += f": {annotation.nominal_value:.3f}mm"
                cv2.putText(result_image, text, mid_point,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            elif isinstance(annotation, RectangleAnnotation):
                # 绘制矩形标注
                bbox = annotation.bbox
                rect = (int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height))
                color = (255, 0, 0) if annotation.feature_type == FeatureType.WIDTH else (0, 255, 255)
                cv2.rectangle(result_image, (rect[0], rect[1]), 
                            (rect[0] + rect[2], rect[1] + rect[3]), color, 2)
        
        return result_image


# =============================================================================
# 工厂函数
# =============================================================================

def create_default_template(name: str = "默认模板") -> InspectionTemplate:
    """创建默认模板"""
    return InspectionTemplate(name=name)


def load_template(filepath: str) -> InspectionTemplate:
    """加载模板"""
    return InspectionTemplate.load(filepath)


def save_template(template: InspectionTemplate, filepath: str):
    """保存模板"""
    template.save(filepath)