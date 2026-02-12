# -*- coding: utf-8 -*-
"""
核心检测模块测试
"""

import unittest
import sys
import os
import numpy as np
import cv2

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_detection import (
    FeatureType,
    DetectionResult,
    InspectionResult,
    SubpixelDetector,
    GeometryFitter,
    ImagePreprocessor,
    CircleDetector,
    ContourDetector
)
from config_manager import InspectionConfig


class TestFeatureType(unittest.TestCase):
    """特征类型测试"""
    
    def test_feature_type_enum(self):
        """测试特征类型枚举"""
        self.assertEqual(FeatureType.CIRCLE.value, "circle")
        self.assertEqual(FeatureType.LINE.value, "line")
        self.assertEqual(FeatureType.RECTANGLE.value, "rectangle")


class TestDetectionResult(unittest.TestCase):
    """检测结果测试"""
    
    def test_detection_result(self):
        """测试检测结果"""
        result = DetectionResult(
            feature_type=FeatureType.CIRCLE,
            center=(100.0, 100.0),
            radius=50.0,
            area=7853.98,
            confidence=0.95
        )
        
        self.assertEqual(result.feature_type, FeatureType.CIRCLE)
        self.assertEqual(result.center, (100.0, 100.0))
        self.assertEqual(result.radius, 50.0)
        
        data = result.to_dict()
        self.assertEqual(data['feature_type'], "circle")
        self.assertEqual(data['center'], (100.0, 100.0))
    
    def test_inspection_result(self):
        """测试检测结果（含公差）"""
        result = InspectionResult(
            feature_type=FeatureType.CIRCLE,
            measured_value=10.05,
            nominal_value=10.0,
            tolerance=0.018,
            is_passed=True,
            deviation=0.05,
            confidence=0.95,
            timestamp="2026-02-10 10:00:00"
        )
        
        self.assertTrue(result.is_passed)
        self.assertEqual(result.measured_value, 10.05)


class TestGeometryFitter(unittest.TestCase):
    """几何拟合器测试"""
    
    def setUp(self):
        self.config = InspectionConfig()
        self.fitter = GeometryFitter(self.config)
    
    def test_fit_circle(self):
        """测试圆形拟合"""
        # 生成圆上的点
        center = (100, 100)
        radius = 50
        angles = np.linspace(0, 2*np.pi, 20)
        points = np.array([
            [center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)]
            for a in angles
        ])
        
        result = self.fitter.fit_circle(points)
        
        self.assertIn('center', result)
        self.assertIn('radius', result)
        self.assertIn('score', result)
        
        # 验证拟合结果接近原始值
        fitted_center = result['center']
        fitted_radius = result['radius']
        
        self.assertAlmostEqual(fitted_center[0], center[0], delta=1.0)
        self.assertAlmostEqual(fitted_center[1], center[1], delta=1.0)
        self.assertAlmostEqual(fitted_radius, radius, delta=1.0)
    
    def test_fit_circle_ransac(self):
        """测试RANSAC圆形拟合"""
        # 生成圆上的点，添加一些噪声点
        center = (100, 100)
        radius = 50
        angles = np.linspace(0, 2*np.pi, 15)
        points = np.array([
            [center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)]
            for a in angles
        ])
        
        # 添加噪声点
        noise = np.array([[50, 50], [150, 150], [50, 150], [150, 50]])
        all_points = np.vstack([points, noise])
        
        result = self.fitter.fit_circle_ransac(all_points, iterations=100, threshold=2.0)
        
        self.assertIn('center', result)
        self.assertIn('radius', result)
        
        # RANSAC应该能够排除噪声点
        fitted_radius = result['radius']
        self.assertAlmostEqual(fitted_radius, radius, delta=5.0)
    
    def test_fit_line(self):
        """测试直线拟合"""
        # 生成直线上的点
        points = np.array([
            [0, 0],
            [10, 10],
            [20, 20],
            [30, 30],
            [40, 40]
        ], dtype=float)
        
        result = self.fitter.fit_line(points)
        
        self.assertIn('start', result)
        self.assertIn('end', result)
        self.assertIn('length', result)
        self.assertIn('angle', result)
        
        # 验证角度接近45度
        self.assertAlmostEqual(result['angle'], 45.0, delta=1.0)
    
    def test_fit_rectangle(self):
        """测试矩形拟合"""
        # 生成矩形的点
        points = np.array([
            [50, 50],
            [150, 50],
            [150, 100],
            [100, 100],
            [50, 100]
        ], dtype=float)
        
        result = self.fitter.fit_rectangle(points)
        
        self.assertIn('center', result)
        self.assertIn('width', result)
        self.assertIn('height', result)


class TestImagePreprocessor(unittest.TestCase):
    """图像预处理器测试"""
    
    def setUp(self):
        self.config = InspectionConfig()
        self.preprocessor = ImagePreprocessor(self.config)
        
        # 创建测试图像
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(self.test_image, (320, 240), 100, (255, 255, 255), -1)
    
    def test_preprocess(self):
        """测试图像预处理"""
        processed = self.preprocessor.preprocess(self.test_image)
        
        self.assertEqual(len(processed.shape), 2)  # 应该是灰度图
        self.assertEqual(processed.shape, (480, 640))
    
    def test_detect_edges(self):
        """测试边缘检测"""
        # 先预处理
        gray = self.preprocessor.preprocess(self.test_image)
        
        # 边缘检测
        edges = self.preprocessor.detect_edges(gray)
        
        self.assertEqual(edges.shape, (480, 640))
        self.assertEqual(edges.dtype, np.uint8)


class TestCircleDetector(unittest.TestCase):
    """圆形检测器测试"""
    
    def setUp(self):
        self.config = InspectionConfig()
        self.detector = CircleDetector(self.config)
        self.preprocessor = ImagePreprocessor(self.config)
        
        # 创建测试图像（包含一个圆）
        self.test_image = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(self.test_image, (320, 240), 80, 255, 2)
    
    def test_detect_circles(self):
        """测试圆形检测"""
        processed = self.preprocessor.preprocess(self.test_image)
        results = self.detector.detect(processed, min_radius=50, max_radius=100)
        
        # 应该检测到至少一个圆
        self.assertGreater(len(results), 0)
        
        # 检查检测结果
        result = results[0]
        self.assertEqual(result.feature_type, FeatureType.CIRCLE)
        self.assertIsNotNone(result.center)
        self.assertIsNotNone(result.radius)
    
    def test_detect_by_contour(self):
        """测试通过轮廓检测圆形"""
        processed = self.preprocessor.preprocess(self.test_image)
        results = self.detector.detect_by_contour(processed)
        
        # 应该检测到至少一个圆
        self.assertGreater(len(results), 0)


class TestContourDetector(unittest.TestCase):
    """轮廓检测器测试"""
    
    def setUp(self):
        self.config = InspectionConfig()
        self.detector = ContourDetector(self.config)
        
        # 创建测试图像
        self.test_image = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(self.test_image, (100, 100), (300, 300), 255, -1)
    
    def test_find_contours(self):
        """测试查找轮廓"""
        contours = self.detector.find_contours(self.test_image)
        
        # 应该找到至少一个轮廓
        self.assertGreater(len(contours), 0)


if __name__ == '__main__':
    unittest.main()