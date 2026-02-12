# -*- coding: utf-8 -*-
"""
单元测试运行脚本
Unit Test Runner
"""

import sys
import os
import unittest
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logger_config import get_logger

logger = get_logger(__name__)


def run_tests(test_dir='tests', pattern='test_*.py'):
    """
    运行单元测试
    
    Args:
        test_dir: 测试目录
        pattern: 测试文件模式
    """
    logger.info("=" * 60)
    logger.info("开始运行单元测试")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # 发现测试
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(start_dir, test_dir)
    
    suite = loader.discover(test_path, pattern=pattern)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # 打印总结
    logger.info("=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)
    logger.info(f"测试总数: {result.testsRun}")
    logger.info(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    logger.info(f"失败: {len(result.failures)}")
    logger.info(f"错误: {len(result.errors)}")
    logger.info(f"跳过: {len(result.skipped)}")
    logger.info(f"耗时: {duration:.2f}秒")
    
    if result.wasSuccessful():
        logger.info("所有测试通过! ✓")
        return 0
    else:
        logger.error("测试失败! ✗")
        return 1


def run_specific_test(test_class_name):
    """
    运行特定测试类
    
    Args:
        test_class_name: 测试类名
    """
    logger.info(f"运行测试类: {test_class_name}")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    try:
        # 动态导入测试模块
        module = __import__(f'tests.{test_class_name}', fromlist=[test_class_name])
        test_class = getattr(module, test_class_name)
        
        suite.addTests(loader.loadTestsFromTestCase(test_class))
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return 0 if result.wasSuccessful() else 1
    
    except Exception as e:
        logger.error(f"运行测试失败: {e}")
        return 1


def main():
    """主函数"""
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name.endswith('.py'):
            test_name = test_name[:-3]
        return run_specific_test(test_name)
    else:
        return run_tests()


if __name__ == '__main__':
    sys.exit(main())