# -*- coding: utf-8 -*-
"""
配置管理器测试
"""

import unittest
import sys
import os
import json
import tempfile
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_manager import (
    ToleranceStandard,
    ToleranceTable,
    InspectionConfig,
    ConfigManager,
    get_config,
    get_tolerance,
    update_config
)


class TestToleranceTable(unittest.TestCase):
    """公差表测试"""
    
    def setUp(self):
        self.tolerance_table = ToleranceTable()
    
    def test_get_tolerance_it8(self):
        """测试IT8级公差"""
        tolerance = self.tolerance_table.get_tolerance("IT8", 5.0)
        self.assertEqual(tolerance, 0.018)
        
        tolerance = self.tolerance_table.get_tolerance("IT8", 15.0)
        self.assertEqual(tolerance, 0.027)
    
    def test_get_tolerance_it7(self):
        """测试IT7级公差"""
        tolerance = self.tolerance_table.get_tolerance("IT7", 5.0)
        self.assertEqual(tolerance, 0.012)
    
    def test_get_tolerance_out_of_range(self):
        """测试超出范围的公差"""
        tolerance = self.tolerance_table.get_tolerance("IT8", 50.0)
        # 应返回最大范围的公差
        self.assertEqual(tolerance, 0.039)


class TestInspectionConfig(unittest.TestCase):
    """检测配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = InspectionConfig()
        
        self.assertEqual(config.pixel_to_mm, 0.098)
        self.assertEqual(config.default_tolerance_standard, "IT8")
        self.assertEqual(config.subpixel_precision, 1.0 / 20.0)
    
    def test_get_tolerance(self):
        """测试获取公差"""
        config = InspectionConfig()
        
        tolerance = config.get_tolerance(5.0, "IT8")
        self.assertEqual(tolerance, 0.018)
        
        # 使用默认标准
        tolerance = config.get_tolerance(5.0)
        self.assertEqual(tolerance, 0.018)
    
    def test_to_dict(self):
        """测试转换为字典"""
        config = InspectionConfig()
        data = config.to_dict()
        
        self.assertIn('pixel_to_mm', data)
        self.assertIn('tolerance_table', data)
        self.assertIn('it8', data['tolerance_table'])


class TestConfigManager(unittest.TestCase):
    """配置管理器测试"""
    
    def setUp(self):
        # 使用临时文件
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.temp_file.close()
        self.temp_path = self.temp_file.name
    
    def tearDown(self):
        # 删除临时文件
        if os.path.exists(self.temp_path):
            os.remove(self.temp_path)
    
    def test_save_and_load_config(self):
        """测试保存和加载配置"""
        config = InspectionConfig()
        config.pixel_to_mm = 0.1
        config.min_circularity = 0.8
        
        # 保存
        ConfigManager.save_config(config, self.temp_path)
        
        # 加载
        loaded_config = ConfigManager.load_config(self.temp_path)
        
        self.assertEqual(loaded_config.pixel_to_mm, 0.1)
        self.assertEqual(loaded_config.min_circularity, 0.8)
    
    def test_update_config(self):
        """测试更新配置"""
        # 先保存一个配置
        config = InspectionConfig()
        ConfigManager.save_config(config, self.temp_path)
        
        # 更新
        ConfigManager.update_config(pixel_to_mm=0.12, min_radius=10)
        
        # 重新加载验证
        updated_config = ConfigManager.load_config(self.temp_path)
        self.assertEqual(updated_config.pixel_to_mm, 0.12)
        self.assertEqual(updated_config.min_radius, 10)
    
    def test_get_tolerance(self):
        """测试获取公差"""
        tolerance = ConfigManager.get_tolerance(5.0, "IT8")
        self.assertEqual(tolerance, 0.018)


if __name__ == '__main__':
    unittest.main()