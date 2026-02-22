# -*- coding: utf-8 -*-
"""
自动测量功能测试
Auto Measurement Feature Tests
"""

import unittest
import os
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_measurement import AutoMeasurementEngine, AutoMeasurementReport, MeasurementResult
from dwg_converter import DWGConverter, ConversionResult
from image_registration import ImageRegistration, TransformationMatrix, MatchResult
from drawing_annotation import InspectionTemplate, CircleAnnotation, FeatureType, ToleranceStandard
from logger_config import get_logger

logger = get_logger(__name__)


class TestDWGConverter(unittest.TestCase):
    """DWG转换器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.converter = DWGConverter()
    
    def test_converter_initialization(self):
        """测试转换器初始化"""
        self.assertIsNotNone(self.converter)
    
    def test_find_converter(self):
        """测试查找转换器"""
        result = self.converter._find_converter()
        # 即使找不到转换器，也应该返回False而不是报错
        self.assertIsInstance(result, bool)
    
    def test_convert_nonexistent_file(self):
        """测试转换不存在的文件"""
        result = self.converter.convert("nonexistent.dwg")
        self.assertFalse(result.success)
        self.assertIn("不存在", result.message)
    
    def test_convert_invalid_file(self):
        """测试转换无效文件"""
        # 创建一个临时文本文件
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test")
            temp_file = f.name
        
        try:
            result = self.converter.convert(temp_file)
            self.assertFalse(result.success)
            self.assertIn("不是DWG格式", result.message)
        finally:
            os.remove(temp_file)
    
    def test_batch_convert_empty_list(self):
        """测试批量转换空列表"""
        import tempfile
        temp_dir = tempfile.mkdtemp()
        try:
            result = self.converter.batch_convert([], temp_dir)
            self.assertEqual(result.total, 0)
            self.assertEqual(result.success, 0)
            self.assertEqual(result.failed, 0)
        finally:
            os.rmdir(temp_dir)


class TestImageRegistration(unittest.TestCase):
    """图像配准测试"""
    
    def setUp(self):
        """测试前准备"""
        self.registration = ImageRegistration(method='ORB')
    
    def test_registration_initialization(self):
        """测试配准器初始化"""
        self.assertIsNotNone(self.registration)
        self.assertEqual(self.registration.method, 'ORB')
    
    def test_register_with_none_images(self):
        """测试使用None图像注册"""
        result = self.registration.register(None, None)
        self.assertFalse(result.success)
        self.assertIn("空", result.message)
    
    def test_register_with_empty_images(self):
        """测试使用空图像注册"""
        empty_image = np.array([])
        result = self.registration.register(empty_image, empty_image)
        self.assertFalse(result.success)
    
    def test_register_with_small_images(self):
        """测试使用小图像注册"""
        small_template = np.ones((10, 10), dtype=np.uint8) * 128
        small_image = np.ones((10, 10), dtype=np.uint8) * 128
        
        result = self.registration.register(small_image, small_template)
        # 小图像可能无法检测到足够的特征点
        self.assertIsNotNone(result)
    
    def test_transform_point(self):
        """测试点变换"""
        # 创建简单的平移变换矩阵
        matrix = np.array([
            [1, 0, 10],
            [0, 1, 20],
            [0, 0, 1]
        ])
        transformation = TransformationMatrix(matrix, 'translation', 1.0, 10)
        
        point = (5.0, 5.0)
        transformed = self.registration.transform_point(point, transformation)
        
        self.assertEqual(transformed[0], 15.0)  # 5 + 10
        self.assertEqual(transformed[1], 25.0)  # 5 + 20
    
    def test_transform_points(self):
        """测试多点变换"""
        matrix = np.array([
            [1, 0, 10],
            [0, 1, 20],
            [0, 0, 1]
        ])
        transformation = TransformationMatrix(matrix, 'translation', 1.0, 10)
        
        points = [(0.0, 0.0), (5.0, 5.0), (10.0, 10.0)]
        transformed = self.registration.transform_points(points, transformation)
        
        self.assertEqual(len(transformed), 3)
        self.assertEqual(transformed[0], (10.0, 20.0))
        self.assertEqual(transformed[1], (15.0, 25.0))
        self.assertEqual(transformed[2], (20.0, 30.0))


class TestAutoMeasurementEngine(unittest.TestCase):
    """自动测量引擎测试"""
    
    def setUp(self):
        """测试前准备"""
        self.engine = AutoMeasurementEngine()
    
    def test_engine_initialization(self):
        """测试引擎初始化"""
        self.assertIsNotNone(self.engine)
        self.assertIsNotNone(self.engine.inspection_engine)
        self.assertIsNotNone(self.engine.registration)
        self.assertIsNotNone(self.engine.dwg_converter)
        self.assertIsNotNone(self.engine.dxf_parser)
    
    def test_measure_from_nonexistent_dwg(self):
        """测试测量不存在的DWG文件"""
        report = self.engine.measure_from_dwg("nonexistent.dwg", "nonexistent.jpg")
        self.assertFalse(report.registration_success)
        self.assertIsNotNone(report.message)
    
    @patch('auto_measurement.DWGConverter')
    @patch('auto_measurement.DXFParser')
    def test_measure_with_mocked_conversion(self, mock_parser_class, mock_converter_class):
        """测试使用模拟转换的测量"""
        # 模拟转换结果
        mock_conversion_result = ConversionResult(
            success=True,
            input_file="test.dwg",
            output_file="test.dxf",
            message="成功"
        )
        mock_converter = Mock()
        mock_converter.convert.return_value = mock_conversion_result
        mock_converter_class.return_value = mock_converter
        
        # 模拟解析结果
        mock_template = InspectionTemplate(
            name="test",
            annotations=[]
        )
        mock_parser = Mock()
        mock_parser.parse_to_template.return_value = mock_template
        mock_parser_class.return_value = mock_parser
        
        # 创建临时图像
        import tempfile
        temp_image = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        try:
            # 创建一个简单的测试图像
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
            cv2.imwrite(temp_image.name, test_image)
            
            # 执行测量
            report = self.engine.measure_from_dwg("test.dwg", temp_image.name)
            
            # 验证结果
            self.assertIsNotNone(report)
            self.assertEqual(report.dwg_file, "test.dwg")
            self.assertEqual(report.image_file, temp_image.name)
        
        finally:
            if os.path.exists(temp_image.name):
                os.remove(temp_image.name)
    
    def test_simple_render_dxf(self):
        """测试简化DXF渲染"""
        # 创建临时DXF文件
        import tempfile
        temp_dxf = tempfile.NamedTemporaryFile(suffix=".dxf", delete=False)
        try:
            # 写入一个空的DXF文件（简化版）
            temp_dxf.write(b"0\nSECTION\n2\nHEADER\n0\nENDSEC\n")
            temp_dxf.close()
            
            # 尝试渲染
            image = self.engine._simple_render_dxf(temp_dxf.name)
            
            # 应该返回一个图像（即使是空白的）
            self.assertIsNotNone(image)
            self.assertEqual(len(image.shape), 3)  # 应该是彩色图像
        
        finally:
            if os.path.exists(temp_dxf.name):
                os.remove(temp_dxf.name)


class TestMeasurementResult(unittest.TestCase):
    """测量结果测试"""
    
    def test_measurement_result_creation(self):
        """测试创建测量结果"""
        result = MeasurementResult(
            annotation_id="test_id",
            feature_type="diameter",
            measured_value=10.0,
            nominal_value=10.0,
            tolerance=0.1,
            deviation=0.0,
            is_passed=True,
            confidence=0.9,
            message="测试成功"
        )
        
        self.assertEqual(result.annotation_id, "test_id")
        self.assertEqual(result.feature_type, "diameter")
        self.assertEqual(result.measured_value, 10.0)
        self.assertTrue(result.is_passed)
    
    def test_measurement_result_defaults(self):
        """测试测量结果默认值"""
        result = MeasurementResult(
            annotation_id="test_id",
            feature_type="diameter",
            measured_value=10.0
        )
        
        self.assertIsNone(result.nominal_value)
        self.assertIsNone(result.tolerance)
        self.assertIsNone(result.deviation)
        self.assertTrue(result.is_passed)  # 默认为合格
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.message, "")


class TestAutoMeasurementReport(unittest.TestCase):
    """自动测量报告测试"""
    
    def test_report_creation(self):
        """测试创建报告"""
        report = AutoMeasurementReport(
            template_name="test_template",
            dwg_file="test.dwg",
            image_file="test.jpg",
            timestamp="2026-02-22 12:00:00",
            registration_success=True,
            registration_confidence=0.95,
            total_features=10,
            measured_features=10,
            passed_features=9,
            failed_features=1
        )
        
        self.assertEqual(report.template_name, "test_template")
        self.assertEqual(report.total_features, 10)
        self.assertEqual(report.measured_features, 10)
        self.assertEqual(report.passed_features, 9)
        self.assertEqual(report.failed_features, 1)
        self.assertEqual(len(report.results), 0)
        self.assertEqual(report.duration, 0.0)
    
    def test_report_with_results(self):
        """测试带结果的报告"""
        report = AutoMeasurementReport(
            template_name="test_template",
            dwg_file="test.dwg",
            image_file="test.jpg",
            timestamp="2026-02-22 12:00:00",
            registration_success=True,
            registration_confidence=0.95,
            total_features=2,
            measured_features=2,
            passed_features=2,
            failed_features=0
        )
        
        # 添加结果
        result1 = MeasurementResult(
            annotation_id="id1",
            feature_type="diameter",
            measured_value=10.0,
            is_passed=True
        )
        
        result2 = MeasurementResult(
            annotation_id="id2",
            feature_type="length",
            measured_value=20.0,
            is_passed=True
        )
        
        report.results = [result1, result2]
        
        self.assertEqual(len(report.results), 2)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestDWGConverter))
    suite.addTests(loader.loadTestsFromTestCase(TestImageRegistration))
    suite.addTests(loader.loadTestsFromTestCase(TestAutoMeasurementEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestMeasurementResult))
    suite.addTests(loader.loadTestsFromTestCase(TestAutoMeasurementReport))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"测试总数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("=" * 60)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    import sys
    sys.exit(run_tests())
