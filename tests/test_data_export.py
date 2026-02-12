# -*- coding: utf-8 -*-
"""
数据导出模块测试
"""

import unittest
import sys
import os
import tempfile
import json
import pandas as pd

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_export import (
    DataExporter,
    InspectionDataExporter,
    StatisticsCalculator,
    export_data,
    calculate_statistics
)


class TestDataExporter(unittest.TestCase):
    """数据导出器测试"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = DataExporter(self.temp_dir)
        
        self.sample_data = [
            {'id': 1, 'name': 'Test1', 'value': 10.5},
            {'id': 2, 'name': 'Test2', 'value': 20.3},
            {'id': 3, 'name': 'Test3', 'value': 15.7}
        ]
    
    def tearDown(self):
        # 删除临时目录
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_export_to_csv(self):
        """测试导出CSV"""
        filepath = self.exporter.export_to_csv(self.sample_data, 'test.csv')
        
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith('test.csv'))
        
        # 验证CSV内容
        df = pd.read_csv(filepath)
        self.assertEqual(len(df), 3)
    
    def test_export_to_json(self):
        """测试导出JSON"""
        filepath = self.exporter.export_to_json(self.sample_data, 'test.json')
        
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith('test.json'))
        
        # 验证JSON内容
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 3)
    
    def test_export_to_excel(self):
        """测试导出Excel"""
        filepath = self.exporter.export_to_excel(self.sample_data, 'test.xlsx')
        
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith('test.xlsx'))
        
        # 验证Excel内容
        df = pd.read_excel(filepath, engine='openpyxl')
        self.assertEqual(len(df), 3)
    
    def test_export_empty_data(self):
        """测试导出空数据"""
        with self.assertRaises(Exception):
            self.exporter.export_to_csv([])
    
    def test_calculate_statistics(self):
        """测试计算统计"""
        stats = self.exporter.calculate_statistics(self.sample_data)
        
        self.assertIn('summary', stats)
        self.assertIn('detailed', stats)
        
        self.assertEqual(stats['summary']['总检测数'], 3)
        self.assertIn('value', stats['detailed'][0])


class TestInspectionDataExporter(unittest.TestCase):
    """检测数据导出器测试"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = InspectionDataExporter(self.temp_dir)
        
        self.sample_results = [
            {
                'id': 1,
                'feature_type': 'circle',
                'measured_value': 10.05,
                'nominal_value': 10.0,
                'tolerance': 0.018,
                'is_passed': True,
                'deviation': 0.05
            },
            {
                'id': 2,
                'feature_type': 'circle',
                'measured_value': 10.03,
                'nominal_value': 10.0,
                'tolerance': 0.018,
                'is_passed': True,
                'deviation': 0.03
            },
            {
                'id': 3,
                'feature_type': 'circle',
                'measured_value': 10.08,
                'nominal_value': 10.0,
                'tolerance': 0.018,
                'is_passed': True,
                'deviation': 0.08
            }
        ]
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_export_results(self):
        """测试导出检测结果"""
        filepath = self.exporter.export_results(self.sample_results)
        
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith('.csv'))
    
    def test_generate_report(self):
        """测试生成报告"""
        filepath = self.exporter.generate_report(self.sample_results)
        
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith('.xlsx'))


class TestStatisticsCalculator(unittest.TestCase):
    """统计计算器测试"""
    
    def setUp(self):
        self.calculator = StatisticsCalculator()
    
    def test_calculate_pass_rate(self):
        """测试计算合格率"""
        data = [
            {'id': 1, 'is_passed': True},
            {'id': 2, 'is_passed': True},
            {'id': 3, 'is_passed': False},
            {'id': 4, 'is_passed': True}
        ]
        
        result = self.calculator.calculate_pass_rate(data)
        
        self.assertEqual(result['total'], 4)
        self.assertEqual(result['passed'], 3)
        self.assertEqual(result['failed'], 1)
        self.assertAlmostEqual(result['pass_rate'], 75.0)
    
    def test_calculate_statistics_by_field(self):
        """测试按字段计算统计"""
        data = [
            {'value': 10.5},
            {'value': 20.3},
            {'value': 15.7},
            {'value': 18.2}
        ]
        
        result = self.calculator.calculate_statistics_by_field(data, 'value')
        
        self.assertIn('mean', result)
        self.assertIn('std', result)
        self.assertIn('min', result)
        self.assertIn('max', result)
        
        # 验证平均值
        expected_mean = (10.5 + 20.3 + 15.7 + 18.2) / 4
        self.assertAlmostEqual(result['mean'], expected_mean, places=2)
    
    def test_calculate_cp_cpk(self):
        """测试计算过程能力指数"""
        data = [
            {'diameter': 10.02},
            {'diameter': 10.05},
            {'diameter': 9.98},
            {'diameter': 10.03},
            {'diameter': 10.00}
        ]
        
        result = self.calculator.calculate_cp_cpk(data, 'diameter', 10.0, 0.05)
        
        self.assertIn('cp', result)
        self.assertIn('cpk', result)
        self.assertIn('mean', result)
        self.assertIn('std', result)


class TestConvenienceFunctions(unittest.TestCase):
    """便捷函数测试"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = [{'id': 1, 'value': 10.5}]
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_export_data(self):
        """测试导出数据便捷函数"""
        from data_export import DataExporter
        
        exporter = DataExporter(self.temp_dir)
        filepath = exporter.export_to_csv(self.sample_data, 'test.csv')
        
        self.assertTrue(os.path.exists(filepath))


if __name__ == '__main__':
    unittest.main()