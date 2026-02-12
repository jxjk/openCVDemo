# -*- coding: utf-8 -*-
"""
缺陷检测模块
Defect Detection Module

实现表面缺陷、边缘缺损、毛刺等缺陷检测算法

创建日期: 2026-02-10
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing

from exceptions import ImageProcessingException, DetectionException
from logger_config import get_logger
from config_manager import get_config


class DefectType(Enum):
    """缺陷类型"""
    SURFACE_DEFECT = "surface_defect"  # 表面缺陷（划痕、污渍、孔洞）
    EDGE_DEFECT = "edge_defect"  # 边缘缺损（缺口、崩边）
    BURR = "burr"  # 毛刺
    CRACK = "crack"  # 裂纹
    DENT = "dent"  # 凹陷
    BULGE = "bulge"  # 凸起
    UNKNOWN = "unknown"


@dataclass
class DefectInfo:
    """缺陷信息"""
    defect_type: DefectType
    location: Tuple[float, float]  # 中心坐标 (x, y)
    area: float  # 缺陷面积（像素）
    severity: float  # 严重程度 (0-1)
    confidence: float  # 置信度 (0-1)
    bounding_box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    features: Dict[str, Any] = field(default_factory=dict)  # 其他特征
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'defect_type': self.defect_type.value,
            'location': self.location,
            'area': self.area,
            'severity': self.severity,
            'confidence': self.confidence,
            'bounding_box': self.bounding_box,
            'features': self.features
        }


@dataclass
class DefectDetectionResult:
    """缺陷检测结果"""
    has_defect: bool  # 是否有缺陷
    defects: List[DefectInfo]  # 缺陷列表
    quality_score: float  # 质量评分 (0-1)
    timestamp: str
    defect_image: Optional[np.ndarray] = None  # 标注了缺陷的图像
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'has_defect': self.has_defect,
            'defects': [d.to_dict() for d in self.defects],
            'quality_score': self.quality_score,
            'timestamp': self.timestamp,
            'defect_count': len(self.defects)
        }


class DefectDetector:
    """缺陷检测器基类"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)
    
    def detect(self, image: np.ndarray) -> DefectDetectionResult:
        """
        检测缺陷
        
        Args:
            image: 输入图像
        
        Returns:
            缺陷检测结果
        """
        raise NotImplementedError


class SurfaceDefectDetector(DefectDetector):
    """
    表面缺陷检测器
    
    检测：划痕、污渍、孔洞、斑点等表面缺陷
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # 检测参数
        self.min_defect_area = 10  # 最小缺陷面积（像素）
        self.max_defect_area = 5000  # 最大缺陷面积（像素）
        self.defect_threshold = 30  # 缺陷检测阈值
        self.blob_min_size = 5  # 斑点最小尺寸
        self.blob_max_size = 100  # 斑点最大尺寸
    
    def detect(self, image: np.ndarray) -> DefectDetectionResult:
        """
        检测表面缺陷
        
        Args:
            image: 输入图像（灰度图）
        
        Returns:
            缺陷检测结果
        """
        try:
            # 确保是灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 方法1: 基于局部对比度的缺陷检测
            defects1 = self._detect_by_local_contrast(gray)
            
            # 方法2: 基于形态学的缺陷检测
            defects2 = self._detect_by_morphology(gray)
            
            # 方法3: 基于纹理分析的缺陷检测
            defects3 = self._detect_by_texture(gray)
            
            # 合并检测结果
            all_defects = defects1 + defects2 + defects3
            
            # 去重（基于位置）
            all_defects = self._remove_duplicates(all_defects)
            
            # 计算质量评分
            quality_score = self._calculate_quality_score(gray, all_defects)
            
            # 创建结果
            result = DefectDetectionResult(
                has_defect=len(all_defects) > 0,
                defects=all_defects,
                quality_score=quality_score,
                timestamp=cv2.getTickCount() / cv2.getTickFrequency() * 1000
            )
            
            self.logger.info(f"表面缺陷检测完成，发现 {len(all_defects)} 个缺陷")
            
            return result
        
        except Exception as e:
            self.logger.error(f"表面缺陷检测失败: {e}")
            raise ImageProcessingException("表面缺陷检测失败", details={'error': str(e)})
    
    def _detect_by_local_contrast(self, gray: np.ndarray) -> List[DefectInfo]:
        """基于局部对比度的缺陷检测"""
        defects = []
        
        try:
            # 高斯模糊
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            
            # 计算局部对比度
            local_contrast = cv2.absdiff(gray, blurred)
            
            # 阈值化
            _, binary = cv2.threshold(local_contrast, self.defect_threshold, 255, cv2.THRESH_BINARY)
            
            # 查找连通区域
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # 过滤面积
                if area < self.min_defect_area or area > self.max_defect_area:
                    continue
                
                # 获取边界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 计算中心
                center_x = x + w / 2
                center_y = y + h / 2
                
                # 计算严重程度
                severity = min(1.0, area / 1000.0)
                
                # 计算置信度
                confidence = 1.0 - (area / self.max_defect_area)
                
                defect = DefectInfo(
                    defect_type=DefectType.SURFACE_DEFECT,
                    location=(center_x, center_y),
                    area=area,
                    severity=severity,
                    confidence=confidence,
                    bounding_box=(x, y, x + w, y + h),
                    features={
                        'method': 'local_contrast',
                        'width': w,
                        'height': h,
                        'aspect_ratio': w / h if h > 0 else 0
                    }
                )
                defects.append(defect)
            
            return defects
        
        except Exception as e:
            self.logger.error(f"局部对比度检测失败: {e}")
            return defects
    
    def _detect_by_morphology(self, gray: np.ndarray) -> List[DefectInfo]:
        """基于形态学的缺陷检测"""
        defects = []
        
        try:
            # 形态学开运算（去除小噪声）
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            
            # 计算差异
            diff = cv2.absdiff(gray, opened)
            
            # 阈值化
            _, binary = cv2.threshold(diff, self.defect_threshold, 255, cv2.THRESH_BINARY)
            
            # 查找连通区域
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area < self.min_defect_area or area > self.max_defect_area:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w / 2
                center_y = y + h / 2
                
                severity = min(1.0, area / 1000.0)
                confidence = 1.0 - (area / self.max_defect_area)
                
                defect = DefectInfo(
                    defect_type=DefectType.SURFACE_DEFECT,
                    location=(center_x, center_y),
                    area=area,
                    severity=severity,
                    confidence=confidence,
                    bounding_box=(x, y, x + w, y + h),
                    features={
                        'method': 'morphology',
                        'width': w,
                        'height': h,
                        'aspect_ratio': w / h if h > 0 else 0
                    }
                )
                defects.append(defect)
            
            return defects
        
        except Exception as e:
            self.logger.error(f"形态学检测失败: {e}")
            return defects
    
    def _detect_by_texture(self, gray: np.ndarray) -> List[DefectInfo]:
        """基于纹理分析的缺陷检测"""
        defects = []
        
        try:
            # 检查图像是否为空
            if gray is None or gray.size == 0:
                return defects
            
            # 计算局部标准差（纹理粗糙度）
            kernel_size = 15
            mean = cv2.blur(gray, (kernel_size, kernel_size))
            mean_sq = cv2.blur(gray ** 2, (kernel_size, kernel_size))
            std_dev = np.sqrt(mean_sq - mean ** 2)
            
            # 检查std_dev是否有效
            if np.isnan(std_dev).any() or np.isinf(std_dev).any():
                return defects
            
            # 标准化
            std_normalized = cv2.normalize(std_dev, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # 阈值化
            _, binary = cv2.threshold(std_normalized, 50, 255, cv2.THRESH_BINARY)
            
            # 查找连通区域
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area < self.min_defect_area or area > self.max_defect_area:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w / 2
                center_y = y + h / 2
                
                severity = min(1.0, area / 1000.0)
                confidence = 1.0 - (area / self.max_defect_area)
                
                defect = DefectInfo(
                    defect_type=DefectType.SURFACE_DEFECT,
                    location=(center_x, center_y),
                    area=area,
                    severity=severity,
                    confidence=confidence,
                    bounding_box=(x, y, x + w, y + h),
                    features={
                        'method': 'texture',
                        'width': w,
                        'height': h,
                        'aspect_ratio': w / h if h > 0 else 0
                    }
                )
                defects.append(defect)
            
            return defects
        
        except Exception as e:
            self.logger.error(f"纹理分析检测失败: {e}")
            return defects
    
    def _remove_duplicates(self, defects: List[DefectInfo], 
                          distance_threshold: float = 10.0) -> List[DefectInfo]:
        """去除重复缺陷"""
        if len(defects) <= 1:
            return defects
        
        # 按严重程度排序
        defects_sorted = sorted(defects, key=lambda d: d.severity, reverse=True)
        
        unique_defects = [defects_sorted[0]]
        
        for defect in defects_sorted[1:]:
            # 检查与已保留缺陷的距离
            is_duplicate = False
            for unique_defect in unique_defects:
                dist = np.sqrt(
                    (defect.location[0] - unique_defect.location[0]) ** 2 +
                    (defect.location[1] - unique_defect.location[1]) ** 2
                )
                if dist < distance_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_defects.append(defect)
        
        return unique_defects
    
    def _calculate_quality_score(self, gray: np.ndarray, 
                                 defects: List[DefectInfo]) -> float:
        """计算质量评分"""
        if not defects:
            return 1.0
        
        # 基于缺陷总面积
        total_defect_area = sum(d.area for d in defects)
        image_area = gray.shape[0] * gray.shape[1]
        defect_ratio = total_defect_area / image_area
        
        # 基于缺陷严重程度
        avg_severity = sum(d.severity for d in defects) / len(defects)
        
        # 综合评分
        quality_score = 1.0 - (defect_ratio * 10 + avg_severity * 0.5)
        quality_score = max(0.0, min(1.0, quality_score))
        
        return quality_score


class EdgeDefectDetector(DefectDetector):
    """
    边缘缺陷检测器
    
    检测：缺口、崩边、锯齿等边缘缺陷
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        
        self.smoothing_kernel = 5  # 平滑核大小
        self.edge_threshold_low = 50  # 边缘检测低阈值
        self.edge_threshold_high = 150  # 边缘检测高阈值
        self.min_defect_length = 10  # 最小缺陷长度（像素）
        self.max_defect_length = 200  # 最大缺陷长度（像素）
        self.defect_depth_threshold = 5  # 缺陷深度阈值（像素）
    
    def detect(self, image: np.ndarray, contour: Optional[np.ndarray] = None) -> DefectDetectionResult:
        """
        检测边缘缺陷
        
        Args:
            image: 输入图像（灰度图）
            contour: 轮廓（可选，如果提供则检测该轮廓的缺陷）
        
        Returns:
            缺陷检测结果
        """
        try:
            # 确保是灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 如果没有提供轮廓，检测所有轮廓
            if contour is None:
                edges = cv2.Canny(gray, self.edge_threshold_low, self.edge_threshold_high)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                if not contours:
                    return DefectDetectionResult(
                        has_defect=False,
                        defects=[],
                        quality_score=1.0,
                        timestamp=cv2.getTickCount() / cv2.getTickFrequency() * 1000
                    )
                
                # 使用最大轮廓
                contour = max(contours, key=cv2.contourArea)
            
            # 分析轮廓缺陷
            defects = self._analyze_contour_defects(contour)
            
            # 计算质量评分
            quality_score = self._calculate_quality_score(contour, defects)
            
            # 创建结果
            result = DefectDetectionResult(
                has_defect=len(defects) > 0,
                defects=defects,
                quality_score=quality_score,
                timestamp=cv2.getTickCount() / cv2.getTickFrequency() * 1000
            )
            
            self.logger.info(f"边缘缺陷检测完成，发现 {len(defects)} 个缺陷")
            
            return result
        
        except Exception as e:
            self.logger.error(f"边缘缺陷检测失败: {e}")
            raise ImageProcessingException("边缘缺陷检测失败", details={'error': str(e)})
    
    def _analyze_contour_defects(self, contour: np.ndarray) -> List[DefectInfo]:
        """分析轮廓缺陷"""
        defects = []
        
        try:
            # 简化轮廓，避免自相交
            epsilon = 0.001 * cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # 计算凸包
            hull = cv2.convexHull(contour, returnPoints=False)
            
            if len(hull) < 3:
                return defects
            
            # 计算凸包缺陷
            defects_data = cv2.convexityDefects(contour, hull)
            
            if defects_data is None:
                return defects
            
            for i in range(defects_data.shape[0]):
                s, e, f, depth = defects_data[i, 0]
                
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                
                # 计算缺陷深度
                depth_value = depth / 256.0
                
                # 过滤浅缺陷
                if depth_value < self.defect_depth_threshold:
                    continue
                
                # 计算缺陷长度
                defect_length = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                
                # 过滤长度
                if defect_length < self.min_defect_length or defect_length > self.max_defect_length:
                    continue
                
                # 计算严重程度
                severity = min(1.0, depth_value / 20.0)
                
                # 计算置信度
                confidence = 1.0 - (depth_value / 50.0)
                
                # 计算边界框
                x1 = min(start[0], end[0], far[0])
                y1 = min(start[1], end[1], far[1])
                x2 = max(start[0], end[0], far[0])
                y2 = max(start[1], end[1], far[1])
                
                defect = DefectInfo(
                    defect_type=DefectType.EDGE_DEFECT,
                    location=far,
                    area=defect_length * depth_value,
                    severity=severity,
                    confidence=confidence,
                    bounding_box=(x1, y1, x2, y2),
                    features={
                        'start': start,
                        'end': end,
                        'depth': depth_value,
                        'length': defect_length,
                        'method': 'convexity'
                    }
                )
                defects.append(defect)
            
            return defects
        
        except Exception as e:
            self.logger.error(f"轮廓缺陷分析失败: {e}")
            return defects
    
    def _calculate_quality_score(self, contour: np.ndarray, 
                                 defects: List[DefectInfo]) -> float:
        """计算质量评分"""
        if not defects:
            return 1.0
        
        # 计算轮廓周长
        contour_length = cv2.arcLength(contour, True)
        
        # 计算总缺陷长度
        total_defect_length = sum(d.features.get('length', 0) for d in defects)
        
        # 缺陷比例
        defect_ratio = total_defect_length / contour_length if contour_length > 0 else 0
        
        # 平均严重程度
        avg_severity = sum(d.severity for d in defects) / len(defects)
        
        # 综合评分
        quality_score = 1.0 - (defect_ratio * 5 + avg_severity * 0.3)
        quality_score = max(0.0, min(1.0, quality_score))
        
        return quality_score


class BurrDetector(DefectDetector):
    """
    毛刺检测器
    
    检测边缘毛刺和突起
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        
        self.burr_threshold = 3.0  # 毛刺阈值（像素）
        self.burr_length_min = 2  # 最小毛刺长度（像素）
        self.burr_length_max = 30  # 最大毛刺长度（像素）
        self.smoothing_iterations = 3  # 平滑迭代次数
    
    def detect(self, image: np.ndarray, contour: Optional[np.ndarray] = None) -> DefectDetectionResult:
        """
        检测毛刺
        
        Args:
            image: 输入图像（灰度图）
            contour: 轮廓（可选）
        
        Returns:
            缺陷检测结果
        """
        try:
            # 确保是灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 如果没有提供轮廓，检测轮廓
            if contour is None:
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                if not contours:
                    return DefectDetectionResult(
                        has_defect=False,
                        defects=[],
                        quality_score=1.0,
                        timestamp=cv2.getTickCount() / cv2.getTickFrequency() * 1000
                    )
                
                contour = max(contours, key=cv2.contourArea)
            
            # 检测毛刺
            defects = self._detect_burrs(contour)
            
            # 计算质量评分
            quality_score = self._calculate_quality_score(contour, defects)
            
            # 创建结果
            result = DefectDetectionResult(
                has_defect=len(defects) > 0,
                defects=defects,
                quality_score=quality_score,
                timestamp=cv2.getTickCount() / cv2.getTickFrequency() * 1000
            )
            
            self.logger.info(f"毛刺检测完成，发现 {len(defects)} 个毛刺")
            
            return result
        
        except Exception as e:
            self.logger.error(f"毛刺检测失败: {e}")
            raise ImageProcessingException("毛刺检测失败", details={'error': str(e)})
    
    def _detect_burrs(self, contour: np.ndarray) -> List[DefectInfo]:
        """检测毛刺"""
        defects = []
        
        try:
            # 平滑轮廓
            smoothed = contour.copy().astype(np.float32)
            for _ in range(self.smoothing_iterations):
                smoothed = cv2.blur(smoothed, (3, 3))
            
            # 计算距离
            distances = []
            for i in range(len(contour)):
                # 找到平滑轮廓上最近的点
                p1 = contour[i][0]
                
                # 计算到平滑轮廓的距离
                min_dist = float('inf')
                for j in range(len(smoothed)):
                    p2 = smoothed[j][0]
                    dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                
                distances.append(min_dist)
            
            distances = np.array(distances)
            
            # 检测毛刺（距离突然增大的点）
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            threshold = mean_dist + self.burr_threshold * std_dist
            
            # 找到毛刺位置
            burr_indices = np.where(distances > threshold)[0]
            
            # 合并相邻的毛刺
            if len(burr_indices) > 0:
                burr_groups = [[burr_indices[0]]]
                for idx in burr_indices[1:]:
                    if idx - burr_groups[-1][-1] <= 5:  # 相邻5个点内
                        burr_groups[-1].append(idx)
                    else:
                        burr_groups.append([idx])
                
                # 为每个毛刺组创建缺陷信息
                for group in burr_groups:
                    if len(group) < self.burr_length_min:
                        continue
                    
                    # 计算毛刺位置
                    points = [contour[i][0] for i in group]
                    center_x = sum(p[0] for p in points) / len(points)
                    center_y = sum(p[1] for p in points) / len(points)
                    
                    # 计算毛刺长度
                    length = len(group)
                    
                    if length > self.burr_length_max:
                        continue
                    
                    # 计算严重程度
                    severity = min(1.0, length / 20.0)
                    
                    # 计算置信度
                    avg_distance = np.mean([distances[i] for i in group])
                    confidence = min(1.0, avg_distance / 10.0)
                    
                    # 计算边界框
                    x1 = min(p[0] for p in points)
                    y1 = min(p[1] for p in points)
                    x2 = max(p[0] for p in points)
                    y2 = max(p[1] for p in points)
                    
                    defect = DefectInfo(
                        defect_type=DefectType.BURR,
                        location=(center_x, center_y),
                        area=length * 2,
                        severity=severity,
                        confidence=confidence,
                        bounding_box=(x1, y1, x2, y2),
                        features={
                            'length': length,
                            'avg_distance': avg_distance,
                            'method': 'contour_distance'
                        }
                    )
                    defects.append(defect)
            
            return defects
        
        except Exception as e:
            self.logger.error(f"毛刺检测失败: {e}")
            return defects
    
    def _calculate_quality_score(self, contour: np.ndarray, 
                                 defects: List[DefectInfo]) -> float:
        """计算质量评分"""
        if not defects:
            return 1.0
        
        # 计算轮廓周长
        contour_length = cv2.arcLength(contour, True)
        
        # 计算总毛刺长度
        total_burr_length = sum(d.features.get('length', 0) for d in defects)
        
        # 毛刺比例
        burr_ratio = total_burr_length / contour_length if contour_length > 0 else 0
        
        # 平均严重程度
        avg_severity = sum(d.severity for d in defects) / len(defects)
        
        # 综合评分
        quality_score = 1.0 - (burr_ratio * 10 + avg_severity * 0.5)
        quality_score = max(0.0, min(1.0, quality_score))
        
        return quality_score


class ComprehensiveDefectDetector:
    """
    综合缺陷检测器
    
    整合多种缺陷检测方法，提供完整的缺陷检测报告
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)
        
        # 初始化各个检测器
        self.surface_detector = SurfaceDefectDetector(config)
        self.edge_detector = EdgeDefectDetector(config)
        self.burr_detector = BurrDetector(config)
    
    def detect_all(self, image: np.ndarray, 
                   contour: Optional[np.ndarray] = None) -> DefectDetectionResult:
        """
        执行所有缺陷检测
        
        Args:
            image: 输入图像
            contour: 轮廓（可选）
        
        Returns:
            综合缺陷检测结果
        """
        try:
            # 确保是灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 检测表面缺陷
            surface_result = self.surface_detector.detect(gray)
            
            # 检测边缘缺陷
            edge_result = self.edge_detector.detect(gray, contour)
            
            # 检测毛刺
            burr_result = self.burr_detector.detect(gray, contour)
            
            # 合并所有缺陷
            all_defects = []
            all_defects.extend(surface_result.defects)
            all_defects.extend(edge_result.defects)
            all_defects.extend(burr_result.defects)
            
            # 去重
            all_defects = self._remove_duplicates(all_defects)
            
            # 综合质量评分（取最小值）
            quality_score = min(
                surface_result.quality_score,
                edge_result.quality_score,
                burr_result.quality_score
            )
            
            # 创建结果
            result = DefectDetectionResult(
                has_defect=len(all_defects) > 0,
                defects=all_defects,
                quality_score=quality_score,
                timestamp=cv2.getTickCount() / cv2.getTickFrequency() * 1000
            )
            
            self.logger.info(f"综合缺陷检测完成，发现 {len(all_defects)} 个缺陷")
            
            return result
        
        except Exception as e:
            self.logger.error(f"综合缺陷检测失败: {e}")
            raise ImageProcessingException("综合缺陷检测失败", details={'error': str(e)})
    
    def _remove_duplicates(self, defects: List[DefectInfo], 
                          distance_threshold: float = 15.0) -> List[DefectInfo]:
        """去除重复缺陷"""
        if len(defects) <= 1:
            return defects
        
        # 按严重程度排序
        defects_sorted = sorted(defects, key=lambda d: d.severity, reverse=True)
        
        unique_defects = [defects_sorted[0]]
        
        for defect in defects_sorted[1:]:
            # 检查与已保留缺陷的距离
            is_duplicate = False
            for unique_defect in unique_defects:
                dist = np.sqrt(
                    (defect.location[0] - unique_defect.location[0]) ** 2 +
                    (defect.location[1] - unique_defect.location[1]) ** 2
                )
                if dist < distance_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_defects.append(defect)
        
        return unique_defects
    
    def draw_defects(self, image: np.ndarray, result: DefectDetectionResult) -> np.ndarray:
        """
        在图像上绘制缺陷
        
        Args:
            image: 输入图像
            result: 缺陷检测结果
        
        Returns:
            标注了缺陷的图像
        """
        result_image = image.copy()
        
        if len(image.shape) == 2:
            result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
        
        # 缺陷颜色映射
        color_map = {
            DefectType.SURFACE_DEFECT: (0, 0, 255),  # 红色
            DefectType.EDGE_DEFECT: (0, 255, 255),  # 黄色
            DefectType.BURR: (255, 0, 255),  # 紫色
            DefectType.CRACK: (255, 0, 0),  # 蓝色
        }
        
        for defect in result.defects:
            color = color_map.get(defect.defect_type, (0, 255, 0))
            
            # 绘制边界框
            x1, y1, x2, y2 = defect.bounding_box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制中心点
            cx, cy = int(defect.location[0]), int(defect.location[1])
            cv2.circle(result_image, (cx, cy), 3, color, -1)
            
            # 绘制缺陷信息
            text = f"{defect.defect_type.value}"
            cv2.putText(result_image, text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 绘制质量评分
        score_text = f"Quality: {result.quality_score:.2f}"
        status_text = "NG" if result.has_defect else "OK"
        status_color = (0, 0, 255) if result.has_defect else (0, 255, 0)
        
        cv2.putText(result_image, score_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result_image, status_text, (result_image.shape[1] - 100, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, status_color, 3)
        
        return result_image


# ============================================================================
# 便捷函数
# ============================================================================

def detect_defects(image: np.ndarray, 
                   detect_surface: bool = True,
                   detect_edge: bool = True,
                   detect_burr: bool = True) -> DefectDetectionResult:
    """
    检测缺陷（便捷函数）
    
    Args:
        image: 输入图像
        detect_surface: 是否检测表面缺陷
        detect_edge: 是否检测边缘缺陷
        detect_burr: 是否检测毛刺
    
    Returns:
        缺陷检测结果
    """
    detector = ComprehensiveDefectDetector()
    return detector.detect_all(image)