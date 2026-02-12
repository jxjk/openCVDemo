# -*- coding: utf-8 -*-
"""
异常类测试
"""

import unittest
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exceptions import (
    InspectionSystemException,
    CameraException,
    CameraConnectionError,
    ImageProcessingException,
    DetectionException,
    CalibrationException,
    DXFException,
    StorageException,
    TemplateException,
    ConfigurationException,
    handle_exception
)
from logger_config import get_logger


class TestExceptions(unittest.TestCase):
    """异常类测试"""
    
    def test_base_exception(self):
        """测试基础异常"""
        exc = InspectionSystemException("测试消息", error_code="TEST001", details={'key': 'value'})
        
        self.assertEqual(str(exc), "[TEST001] 测试消息")
        self.assertEqual(exc.message, "测试消息")
        self.assertEqual(exc.error_code, "TEST001")
        self.assertEqual(exc.details, {'key': 'value'})
        
        result = exc.to_dict()
        self.assertEqual(result['error_code'], "TEST001")
        self.assertEqual(result['message'], "测试消息")
    
    def test_camera_exception(self):
        """测试相机异常"""
        exc = CameraConnectionError("camera_01")
        
        self.assertEqual(exc.error_code, "CAM001")
        self.assertEqual(exc.camera_id, "camera_01")
        self.assertIn("camera_01", str(exc))
    
    def test_detection_exception(self):
        """测试检测异常"""
        exc = DetectionException("圆形")
        
        self.assertEqual(exc.error_code, "DET001")
        self.assertEqual(exc.feature_type, "圆形")
    
    def test_handle_exception(self):
        """测试异常处理函数"""
        logger = get_logger()
        
        # 测试自定义异常
        custom_exc = CameraConnectionError("test_camera")
        result = handle_exception(custom_exc, logger=logger, reraise=False)
        
        self.assertEqual(result['error_code'], "CAM001")
        self.assertIn('test_camera', result['message'])
        
        # 测试普通异常
        generic_exc = ValueError("普通错误")
        result = handle_exception(generic_exc, logger=logger, reraise=False)
        
        self.assertEqual(result['error_code'], "UNK001")
        self.assertEqual(result['message'], "普通错误")
        self.assertEqual(result['details']['type'], "ValueError")


if __name__ == '__main__':
    unittest.main()