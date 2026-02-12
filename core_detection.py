# -*- coding: utf-8 -*-
"""
核心检测模块
Core Detection Module

提取自 inspection_system.py 的核心检测功能
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from exceptions import (
    ImageProcessingException,
    DetectionException,
    NoFeaturesFoundError,
    CircleDetectionError,
    CalibrationException
)
from logger_config import get_logger
from config_manager import get_config, InspectionConfig


class FeatureType(Enum):
    """特征类型枚举"""
    CIRCLE = "circle"
    LINE = "line"
    RECTANGLE = "rectangle"
    ANGLE = "angle"
    POINT = "point"
    UNKNOWN = "unknown"


@dataclass
class DetectionResult:
    """检测结果"""
    feature_type: FeatureType
    center: Tuple[float, float]  # (x, y) 像素坐标
    radius: Optional[float] = None  # 半径（像素）
    area: Optional[float] = None  # 面积（像素）
    confidence: float = 0.0  # 置信度
    parameters: Dict[str, Any] = field(default_factory=dict)  # 其他参数
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'feature_type': self.feature_type.value,
            'center': self.center,
            'radius': self.radius,
            'area': self.area,
            'confidence': self.confidence,
            'parameters': self.parameters
        }


@dataclass
class InspectionResult:
    """检测结果（含尺寸和公差）"""
    feature_type: FeatureType
    measured_value: float  # 实测值（mm）
    nominal_value: float  # 标称值（mm）
    tolerance: float  # 公差（mm）
    is_passed: bool  # 是否合格
    deviation: float  # 偏差（mm）
    confidence: float  # 置信度
    timestamp: str  # 时间戳
    image_path: Optional[str] = None  # 图像路径
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'feature_type': self.feature_type.value,
            'measured_value': self.measured_value,
            'nominal_value': self.nominal_value,
            'tolerance': self.tolerance,
            'is_passed': self.is_passed,
            'deviation': self.deviation,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'image_path': self.image_path
        }


class SubpixelDetector:
    """亚像素边缘检测器"""
    
    def __init__(self, config: Optional[InspectionConfig] = None):
        self.config = config or get_config()
        self.logger = get_logger(__name__)
    
    def refine_corner(self, image: np.ndarray, corner: Tuple[float, float], 
                      window_size: int = 10) -> Tuple[float, float]:
        """
        亚像素角点细化
        
        Args:
            image: 输入图像
            corner: 初始角点坐标
            window_size: 搜索窗口大小
        
        Returns:
            亚像素精度的角点坐标
        """
        try:
            # 转换为浮点型
            image_float = image.astype(np.float32)
            
            # 定义搜索窗口
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            
            # 亚像素细化
            corners = np.array([[corner[0], corner[1]]], dtype=np.float32)
            corners_refined = cv2.cornerSubPix(
                image_float, 
                corners, 
                (window_size, window_size), 
                (-1, -1), 
                criteria
            )
            
            refined_point = (corners_refined[0][0], corners_refined[0][1])
            
            self.logger.debug(f"亚像素细化: {corner} -> {refined_point}")
            
            return refined_point
        
        except Exception as e:
            self.logger.error(f"亚像素细化失败: {e}")
            raise ImageProcessingException("亚像素细化失败", details={'error': str(e)})
    
    def refine_edge_point(self, edge_image: np.ndarray, point: Tuple[float, float],
                         gradient_direction: Tuple[float, float], window_size: int = 5) -> Tuple[float, float]:
        """
        亚像素边缘点细化（多项式拟合）
        
        Args:
            edge_image: 边缘图像
            point: 初始边缘点
            gradient_direction: 梯度方向
            window_size: 搜索窗口大小
        
        Returns:
            亚像素精度的边缘点坐标
        """
        try:
            x, y = int(point[0]), int(point[1])
            
            # 提取沿梯度方向的强度分布
            dx, dy = gradient_direction
            samples = []
            intensities = []
            
            for i in range(-window_size, window_size + 1):
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < edge_image.shape[1] and 0 <= ny < edge_image.shape[0]:
                    samples.append(i)
                    intensities.append(float(edge_image[ny, nx]))
            
            if len(samples) < 3:
                return point
            
            # 多项式拟合（二次或三次）
            degree = min(3, len(samples) - 1)
            coeffs = np.polyfit(samples, intensities, degree)
            
            # 找到拟合曲线的最大梯度点
            derivative = np.polyder(coeffs)
            max_grad_idx = None
            
            for i in range(len(samples) - 1):
                if derivative(samples[i]) * derivative(samples[i + 1]) <= 0:
                    max_grad_idx = i
                    break
            
            if max_grad_idx is not None:
                refined_idx = samples[max_grad_idx]
                refined_x = x + refined_idx * dx
                refined_y = y + refined_idx * dy
                return (refined_x, refined_y)
            
            return point
        
        except Exception as e:
            self.logger.error(f"亚像素边缘细化失败: {e}")
            return point


class GeometryFitter:
    """几何拟合器"""
    
    def __init__(self, config: Optional[InspectionConfig] = None):
        self.config = config or get_config()
        self.logger = get_logger(__name__)
    
    def fit_circle(self, points: np.ndarray) -> Dict[str, Any]:
        """
        拟合圆形（最小二乘法）
        
        Args:
            points: 点集，形状为(N, 2)
        
        Returns:
            拟合结果字典: {'center': (x, y), 'radius': r, 'score': float}
        """
        try:
            if len(points) < 3:
                raise ValueError("至少需要3个点才能拟合圆")
            
            # 提取坐标
            x = points[:, 0]
            y = points[:, 1]
            
            # 最小二乘拟合圆
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            
            # 计算中心坐标
            D = (x - x_mean)**2 + (y - y_mean)**2
            S_xx = np.sum(D * (x - x_mean)**2) / np.sum(D)
            S_yy = np.sum(D * (y - y_mean)**2) / np.sum(D)
            S_xy = np.sum(D * (x - x_mean) * (y - y_mean)) / np.sum(D)
            
            center_x = x_mean + S_xx * np.sum((x - x_mean) * D) - S_xy * np.sum((y - y_mean) * D)
            center_y = y_mean + S_yy * np.sum((y - y_mean) * D) - S_xy * np.sum((x - x_mean) * D)
            
            # 计算半径
            radius = np.mean(np.sqrt((x - center_x)**2 + (y - center_y)**2))
            
            # 计算拟合误差
            distances = np.abs(np.sqrt((x - center_x)**2 + (y - center_y)**2) - radius)
            score = 1.0 - np.mean(distances) / radius if radius > 0 else 0.0
            
            result = {
                'center': (float(center_x), float(center_y)),
                'radius': float(radius),
                'score': float(score),
                'points': points.tolist()
            }
            
            self.logger.debug(f"圆形拟合: 中心={result['center']}, 半径={result['radius']:.2f}")
            
            return result
        
        except Exception as e:
            self.logger.error(f"圆形拟合失败: {e}")
            raise CircleDetectionError(details={'error': str(e)})
    
    def fit_circle_ransac(self, points: np.ndarray, iterations: int = 1000,
                         threshold: float = 0.01) -> Dict[str, Any]:
        """
        使用RANSAC拟合圆形（鲁棒拟合）
        
        Args:
            points: 点集，形状为(N, 2)
            iterations: RANSAC迭代次数
            threshold: 内点阈值
        
        Returns:
            拟合结果字典
        """
        try:
            if len(points) < 3:
                raise ValueError("至少需要3个点才能拟合圆")
            
            best_result = None
            best_inliers = 0
            
            for _ in range(iterations):
                # 随机选择3个点
                idx = np.random.choice(len(points), 3, replace=False)
                sample = points[idx]
                
                # 计算通过这3个点的圆
                result = self._fit_circle_3points(sample)
                
                if result is None:
                    continue
                
                # 计算内点
                distances = np.abs(np.sqrt(
                    (points[:, 0] - result['center'][0])**2 + 
                    (points[:, 1] - result['center'][1])**2
                ) - result['radius'])
                
                inliers = np.sum(distances < threshold)
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_result = result
            
            if best_result is None:
                raise CircleDetectionError("RANSAC拟合失败")
            
            # 使用所有内点重新拟合
            inlier_mask = np.sqrt(
                (points[:, 0] - best_result['center'][0])**2 + 
                (points[:, 1] - best_result['center'][1])**2
            ) - best_result['radius'] < threshold
            
            if np.sum(inlier_mask) >= 3:
                best_result = self.fit_circle(points[inlier_mask])
            
            return best_result
        
        except Exception as e:
            self.logger.error(f"RANSAC圆形拟合失败: {e}")
            raise CircleDetectionError(details={'error': str(e)})
    
    def _fit_circle_3points(self, points: np.ndarray) -> Optional[Dict[str, Any]]:
        """通过3个点拟合圆"""
        try:
            x1, y1 = points[0]
            x2, y2 = points[1]
            x3, y3 = points[2]
            
            # 计算中垂线交点（圆心）
            D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
            
            if abs(D) < 1e-10:
                return None  # 3点共线
            
            center_x = ((x1**2 + y1**2) * (y2 - y3) + 
                        (x2**2 + y2**2) * (y3 - y1) + 
                        (x3**2 + y3**2) * (y1 - y2)) / D
            
            center_y = ((x1**2 + y1**2) * (x3 - x2) + 
                        (x2**2 + y2**2) * (x1 - x3) + 
                        (x3**2 + y3**2) * (x2 - x1)) / D
            
            radius = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
            
            return {
                'center': (float(center_x), float(center_y)),
                'radius': float(radius),
                'score': 1.0
            }
        
        except Exception:
            return None
    
    def fit_line(self, points: np.ndarray) -> Dict[str, Any]:
        """
        拟合直线（最小二乘法）
        
        Args:
            points: 点集，形状为(N, 2)
        
        Returns:
            拟合结果字典: {'start': (x1, y1), 'end': (x2, y2), 'length': float, 'score': float}
        """
        try:
            if len(points) < 2:
                raise ValueError("至少需要2个点才能拟合直线")
            
            x = points[:, 0]
            y = points[:, 1]
            
            # 最小二乘拟合
            coeffs = np.polyfit(x, y, 1)
            
            # 计算起点和终点
            x_start, x_end = np.min(x), np.max(x)
            y_start = coeffs[0] * x_start + coeffs[1]
            y_end = coeffs[0] * x_end + coeffs[1]
            
            # 计算长度
            length = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
            
            # 计算拟合误差
            y_pred = coeffs[0] * x + coeffs[1]
            errors = np.abs(y - y_pred)
            score = 1.0 - np.mean(errors) / length if length > 0 else 0.0
            
            result = {
                'start': (float(x_start), float(y_start)),
                'end': (float(x_end), float(y_end)),
                'length': float(length),
                'angle': float(np.degrees(np.arctan2(y_end - y_start, x_end - x_start))),
                'score': float(score)
            }
            
            self.logger.debug(f"直线拟合: 长度={result['length']:.2f}")
            
            return result
        
        except Exception as e:
            self.logger.error(f"直线拟合失败: {e}")
            raise DetectionException("直线拟合失败", details={'error': str(e)})
    
    def fit_rectangle(self, points: np.ndarray) -> Dict[str, Any]:
        """
        拟合矩形（凸包+最小外接矩形）
        
        Args:
            points: 点集，形状为(N, 2)
        
        Returns:
            拟合结果字典
        """
        try:
            if len(points) < 4:
                raise ValueError("至少需要4个点才能拟合矩形")
            
            # 转换为OpenCV格式
            contour = points.reshape((-1, 1, 2)).astype(np.int32)
            
            # 计算最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # 计算长宽
            width = min(rect[1])
            height = max(rect[1])
            
            result = {
                'center': (float(rect[0][0]), float(rect[0][1])),
                'width': float(width),
                'height': float(height),
                'angle': float(rect[2]),
                'corners': box.tolist()
            }
            
            self.logger.debug(f"矩形拟合: 宽={result['width']:.2f}, 高={result['height']:.2f}")
            
            return result
        
        except Exception as e:
            self.logger.error(f"矩形拟合失败: {e}")
            raise DetectionException("矩形拟合失败", details={'error': str(e)})


class ImagePreprocessor:
    """图像预处理器"""
    
    def __init__(self, config: Optional[InspectionConfig] = None):
        self.config = config or get_config()
        self.logger = get_logger(__name__)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 输入图像
        
        Returns:
            预处理后的图像
        """
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 高斯滤波去噪
            blurred = cv2.GaussianBlur(gray, (self.config.gaussian_kernel, self.config.gaussian_kernel), 0)
            
            # 中值滤波去除椒盐噪声
            denoised = cv2.medianBlur(blurred, self.config.median_kernel)
            
            # 自适应直方图均衡化
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            self.logger.debug("图像预处理完成")
            
            return enhanced
        
        except Exception as e:
            self.logger.error(f"图像预处理失败: {e}")
            raise ImageProcessingException("图像预处理失败", details={'error': str(e)})
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        边缘检测（Canny）
        
        Args:
            image: 输入图像（预处理后的灰度图）
        
        Returns:
            边缘图像
        """
        try:
            edges = cv2.Canny(
                image,
                self.config.canny_threshold1,
                self.config.canny_threshold2,
                apertureSize=self.config.aperture_size
            )
            
            self.logger.debug(f"边缘检测完成，边缘点数: {np.sum(edges > 0)}")
            
            return edges
        
        except Exception as e:
            self.logger.error(f"边缘检测失败: {e}")
            raise ImageProcessingException("边缘检测失败", details={'error': str(e)})


class CircleDetector:
    """圆形检测器"""
    
    def __init__(self, config: Optional[InspectionConfig] = None):
        self.config = config or get_config()
        self.logger = get_logger(__name__)
        self.fitter = GeometryFitter(config)
    
    def detect(self, image: np.ndarray, 
               min_radius: Optional[int] = None,
               max_radius: Optional[int] = None) -> List[DetectionResult]:
        """
        检测圆形（Hough变换）
        
        Args:
            image: 输入图像
            min_radius: 最小半径（像素）
            max_radius: 最大半径（像素）
        
        Returns:
            检测结果列表
        """
        try:
            if min_radius is None:
                min_radius = self.config.min_radius
            if max_radius is None:
                max_radius = self.config.max_radius
            
            # Hough圆检测
            circles = cv2.HoughCircles(
                image,
                cv2.HOUGH_GRADIENT,
                dp=self.config.dp,
                minDist=self.config.min_dist,
                param1=self.config.param1,
                param2=self.config.param2,
                minRadius=min_radius,
                maxRadius=max_radius
            )
            
            results = []
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                for (x, y, r) in circles:
                    # 计算圆的面积
                    area = np.pi * r ** 2
                    
                    results.append(DetectionResult(
                        feature_type=FeatureType.CIRCLE,
                        center=(float(x), float(y)),
                        radius=float(r),
                        area=float(area),
                        confidence=1.0  # Hough变换的置信度
                    ))
                
                self.logger.info(f"检测到 {len(results)} 个圆形")
            
            else:
                self.logger.warning("未检测到圆形")
            
            return results
        
        except Exception as e:
            self.logger.error(f"圆形检测失败: {e}")
            raise CircleDetectionError(details={'error': str(e)})
    
    def detect_by_contour(self, image: np.ndarray,
                         min_area: Optional[int] = None,
                         max_area: Optional[int] = None) -> List[DetectionResult]:
        """
        通过轮廓检测圆形
        
        Args:
            image: 输入图像
            min_area: 最小面积
            max_area: 最大面积
        
        Returns:
            检测结果列表
        """
        try:
            if min_area is None:
                min_area = self.config.min_contour_area
            if max_area is None:
                max_area = self.config.max_contour_area
            
            # 查找轮廓
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            results = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area < min_area or area > max_area:
                    continue
                
                # 拟合圆
                (x, y), radius = cv2.minEnclosingCircle(contour)
                area_ratio = area / (np.pi * radius ** 2)
                
                # 检查圆形度
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2)
                
                if circularity >= self.config.min_circularity:
                    results.append(DetectionResult(
                        feature_type=FeatureType.CIRCLE,
                        center=(float(x), float(y)),
                        radius=float(radius),
                        area=float(area),
                        confidence=circularity,
                        parameters={
                            'circularity': circularity,
                            'area_ratio': area_ratio
                        }
                    ))
            
            self.logger.info(f"轮廓检测到 {len(results)} 个圆形")
            
            return results
        
        except Exception as e:
            self.logger.error(f"轮廓圆形检测失败: {e}")
            raise CircleDetectionError(details={'error': str(e)})


class ContourDetector:
    """轮廓检测器"""
    
    def __init__(self, config: Optional[InspectionConfig] = None):
        self.config = config or get_config()
        self.logger = get_logger(__name__)
    
    def find_contours(self, image: np.ndarray,
                     min_area: Optional[int] = None,
                     max_area: Optional[int] = None) -> List[np.ndarray]:
        """
        查找轮廓
        
        Args:
            image: 输入图像（二值图）
            min_area: 最小面积
            max_area: 最大面积
        
        Returns:
            轮廓列表
        """
        try:
            if min_area is None:
                min_area = self.config.min_contour_area
            if max_area is None:
                max_area = self.config.max_contour_area
            
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 过滤轮廓
            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    filtered_contours.append(contour)
            
            self.logger.debug(f"找到 {len(filtered_contours)} 个轮廓")
            
            return filtered_contours
        
        except Exception as e:
            self.logger.error(f"轮廓查找失败: {e}")
            raise ImageProcessingException("轮廓查找失败", details={'error': str(e)})