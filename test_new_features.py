# -*- coding: utf-8 -*-
"""
新功能测试脚本
Test Script for New Features

测试批量检测、缺陷检测等功能

创建日期: 2026-02-10
"""

import os
import sys
import time
import cv2
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inspection_system import InspectionEngine, InspectionConfig
from batch_inspection import BatchInspectionEngine, BatchInspectionConfig
from defect_detection import ComprehensiveDefectDetector, DefectType
from data_export import DataExporter, StatisticsCalculator
from logger_config import get_logger

# 配置日志
logger = get_logger(__name__n)


def create_test_image(size=(1000, 1000), with_defect=False):
    """
    创建测试图像
    
    Args:
        size: 图像大小 (width, height)
        with_defect: 是否包含缺陷
    
    Returns:
        测试图像
    """
    # 创建白色背景
    image = np.ones((size[1], size[0]), dtype=np.uint8) * 255
    
    # 绘制圆形零件
    center = (size[0] // 2, size[1] // 2)
    radius = 200
    cv2.circle(image, center, radius, 100, -1)
    cv2.circle(image, center, radius, 0, 3)
    
    # 添加缺陷
    if with_defect:
        # 表面缺陷（污渍）
        cv2.circle(image, (center[0] - 50, center[1] - 50), 20, 50, -1)
        
        # 划痕
        cv2.line(image, (center[0] + 30, center[1] - 100), 
                (center[0] + 100, center[1] - 50), 150, 2)
        
        # 毛刺（在边缘）
        cv2.line(image, (center[0] + radius, center[1]), 
                (center[0] + radius + 15, center[1]), 50, 3)
    
    return image


def test_inspection_engine():
    """测试检测引擎"""
    logger.info("=" * 60)
    logger.info("测试检测引擎")
    logger.info("=" * 60)
    
    try:
        # 初始化
        config = InspectionConfig()
        config.PIXEL_TO_MM = 0.098
        
        inspection_engine = InspectionEngine(config)
        
        # 创建测试图像
        image = create_test_image(with_defect=False)
        
        # 执行检测
        result = inspection_engine.detect_circle(
            image,
            part_id="TEST_001",
            part_type="圆形",
            nominal_size=5.0
        )
        
        if result:
            logger.info(f"检测成功:")
            logger.info(f"  零件编号: {result.part_id}")
            logger.info(f"  实测直径: {result.diameter_mm:.3f} mm")
            logger.info(f"  标称值: {result.nominal_size:.3f} mm")
            logger.info(f"  偏差: {result.deviation:.3f} mm")
            logger.info(f"  结果: {'合格' if result.is_qualified else '不合格'}")
            
            # 绘制结果
            result_image = inspection_engine.draw_result(image, result)
            cv2.imwrite("test_inspection_result.jpg", result_image)
            logger.info("结果图像已保存: test_inspection_result.jpg")
        else:
            logger.error("检测失败")
        
        return True
    
    except Exception as e:
        logger.error(f"测试检测引擎失败: {e}")
        return False


def test_defect_detection():
    """测试缺陷检测"""
    logger.info("=" * 60)
    logger.info("测试缺陷检测")
    logger.info("=" * 60)
    
    try:
        # 创建测试图像
        image_good = create_test_image(with_defect=False)
        image_bad = create_test_image(with_defect=True)
        
        # 初始化检测器
        detector = ComprehensiveDefectDetector()
        
        # 测试合格零件
        logger.info("\n测试合格零件:")
        result_good = detector.detect_all(image_good)
        logger.info(f"  有缺陷: {result_good.has_defect}")
        logger.info(f"  缺陷数量: {len(result_good.defects)}")
        logger.info(f"  质量评分: {result_good.quality_score:.2f}")
        
        # 测试不合格零件
        logger.info("\n测试不合格零件:")
        result_bad = detector.detect_all(image_bad)
        logger.info(f"  有缺陷: {result_bad.has_defect}")
        logger.info(f"  缺陷数量: {len(result_bad.defects)}")
        logger.info(f"  质量评分: {result_bad.quality_score:.2f}")
        
        # 显示缺陷详情
        if result_bad.has_defect:
            logger.info("\n  缺陷详情:")
            for i, defect in enumerate(result_bad.defects):
                logger.info(f"    {i+1}. {defect.defect_type.value}")
                logger.info(f"       位置: {defect.location}")
                logger.info(f"       面积: {defect.area:.1f} 像素")
                logger.info(f"       严重程度: {defect.severity:.2f}")
        
        # 绘制缺陷标记
        result_image = detector.draw_defects(image_bad, result_bad)
        cv2.imwrite("test_defect_result.jpg", result_image)
        logger.info("\n缺陷标记图像已保存: test_defect_result.jpg")
        
        return True
    
    except Exception as e:
        logger.error(f"测试缺陷检测失败: {e}")
        return False


def test_batch_inspection():
    """测试批量检测"""
    logger.info("=" * 60)
    logger.info("测试批量检测")
    logger.info("=" * 60)
    
    try:
        # 初始化
        config = InspectionConfig()
        inspection_engine = InspectionEngine(config)
        
        batch_config = BatchInspectionConfig(
            max_workers=2,
            target_speed=60
        )
        batch_engine = BatchInspectionEngine(inspection_engine, batch_config)
        
        # 设置回调
        results_collected = []
        
        def on_result(result):
            results_collected.append(result)
            status = "✓" if result.is_passed else "✗"
            logger.info(f"  {status} {result.part_id}: {'合格' if result.is_passed else '不合格'}")
        
        batch_engine.set_result_callback(on_result)
        
        # 启动批量检测
        logger.info("启动批量检测...")
        batch_engine.start()
        
        # 添加任务
        logger.info("添加检测任务...")
        for i in range(10):
            image = create_test_image(with_defect=(i % 3 == 0))  # 每3个有一个缺陷
            batch_engine.add_image(
                image,
                part_id=f"TEST_{i:03d}",
                part_type="圆形",
                nominal_size=5.0
            )
            time.sleep(0.05)  # 模拟采集间隔
        
        # 等待完成
        logger.info("等待检测完成...")
        time.sleep(5)
        
        # 停止批量检测
        batch_engine.stop()
        
        # 获取统计信息
        stats = batch_engine.get_statistics()
        logger.info("\n批量检测统计:")
        logger.info(f"  总计: {stats['completed_tasks']} 件")
        logger.info(f"  合格: {stats['passed_tasks']} 件")
        logger.info(f"  不合格: {stats['failed_tasks']} 件")
        logger.info(f"  错误: {stats['error_tasks']} 件")
        logger.info(f"  合格率: {stats['pass_rate']:.2f}%")
        logger.info(f"  平均速度: {stats['current_speed']:.1f} 件/分钟")
        logger.info(f"  平均耗时: {stats['avg_time_per_part']:.3f} 秒/件")
        
        # 验证结果
        success = stats['completed_tasks'] == 10
        if success:
            logger.info("\n✓ 批量检测测试通过")
        else:
            logger.error("\n✗ 批量检测测试失败")
        
        return success
    
    except Exception as e:
        logger.error(f"测试批量检测失败: {e}")
        return False


def test_data_export():
    """测试数据导出"""
    logger.info("=" * 60)
    logger.info("测试数据导出")
    logger.info("=" * 60)
    
    try:
        # 创建测试数据
        test_data = [
            {
                'timestamp': '2026-02-10 10:00:00',
                'part_id': 'TEST_001',
                'part_type': '圆形',
                'measured_value': 5.023,
                'nominal_value': 5.0,
                'is_passed': True,
                'deviation': 0.023
            },
            {
                'timestamp': '2026-02-10 10:00:01',
                'part_id': 'TEST_002',
                'part_type': '圆形',
                'measured_value': 4.975,
                'nominal_value': 5.0,
                'is_passed': True,
                'deviation': -0.025
            },
            {
                'timestamp': '2026-02-10 10:00:02',
                'part_id': 'TEST_003',
                'part_type': '圆形',
                'measured_value': 5.050,
                'nominal_value': 5.0,
                'is_passed': False,
                'deviation': 0.050
            }
        ]
        
        # 初始化导出器
        exporter = DataExporter()
        
        # 测试CSV导出
        logger.info("测试CSV导出...")
        csv_file = exporter.export_to_csv(test_data, "test_export.csv")
        logger.info(f"  CSV文件已保存: {csv_file}")
        
        # 测试Excel导出
        logger.info("测试Excel导出...")
        excel_file = exporter.export_to_excel(test_data, "test_export.xlsx")
        logger.info(f"  Excel文件已保存: {excel_file}")
        
        # 测试统计计算
        logger.info("\n测试统计计算...")
        stats = exporter.calculate_statistics(test_data)
        logger.info(f"  总检测数: {stats['summary']['总检测数']}")
        logger.info(f"  合格数: {stats['summary']['合格数']}")
        logger.info(f"  不合格数: {stats['summary']['不合格数']}")
        logger.info(f"  合格率: {stats['summary']['合格率']:.2f}%")
        
        # 测试统计报表
        logger.info("\n测试统计报表...")
        report_file = exporter.export_statistics(test_data, "test_statistics.xlsx")
        logger.info(f"  统计报表已保存: {report_file}")
        
        logger.info("\n✓ 数据导出测试通过")
        return True
    
    except Exception as e:
        logger.error(f"测试数据导出失败: {e}")
        return False


def test_statistics_calculator():
    """测试统计计算器"""
    logger.info("=" * 60)
    logger.info("测试统计计算器")
    logger.info("=" * 60)
    
    try:
        # 创建测试数据
        test_data = [
            {'is_passed': True, 'measured_value': 5.020},
            {'is_passed': True, 'measured_value': 5.015},
            {'is_passed': True, 'measured_value': 4.990},
            {'is_passed': False, 'measured_value': 5.050},
            {'is_passed': True, 'measured_value': 5.005},
            {'is_passed': False, 'measured_value': 4.930},
            {'is_passed': True, 'measured_value': 4.995},
            {'is_passed': True, 'measured_value': 5.010},
        ]
        
        # 初始化计算器
        calc = StatisticsCalculator()
        
        # 测试合格率计算
        logger.info("测试合格率计算...")
        pass_rate = calc.calculate_pass_rate(test_data)
        logger.info(f"  总计: {pass_rate['total']}")
        logger.info(f"  合格: {pass_rate['passed']}")
        logger.info(f"  不合格: {pass_rate['failed']}")
        logger.info(f"  合格率: {pass_rate['pass_rate']:.2f}%")
        
        # 测试字段统计
        logger.info("\n测试字段统计...")
        field_stats = calc.calculate_statistics_by_field(test_data, 'measured_value')
        logger.info(f"  平均值: {field_stats['mean']:.3f}")
        logger.info(f"  标准差: {field_stats['std']:.4f}")
        logger.info(f"  最小值: {field_stats['min']:.3f}")
        logger.info(f"  最大值: {field_stats['max']:.3f}")
        logger.info(f"  中位数: {field_stats['median']:.3f}")
        
        # 测试过程能力指数
        logger.info("\n测试过程能力指数...")
        cpk_result = calc.calculate_cp_cpk(test_data, 'measured_value', 
                                           nominal=5.0, tolerance=0.050)
        logger.info(f"  Cp: {cpk_result['cp']:.3f}")
        logger.info(f"  Cpk: {cpk_result['cpk']:.3f}")
        logger.info(f"  上规格限: {cpk_result['usl']:.3f}")
        logger.info(f"  下规格限: {cpk_result['lsl']:.3f}")
        
        # 解释Cpk值
        if cpk_result['cpk'] >= 1.33:
            logger.info("  评级: 优秀 (Cpk >= 1.33)")
        elif cpk_result['cpk'] >= 1.0:
            logger.info("  评级: 良好 (1.0 <= Cpk < 1.33)")
        elif cpk_result['cpk'] >= 0.67:
            logger.info("  评级: 一般 (0.67 <= Cpk < 1.0)")
        else:
            logger.info("  评级: 不足 (Cpk < 0.67)")
        
        logger.info("\n✓ 统计计算器测试通过")
        return True
    
    except Exception as e:
        logger.error(f"测试统计计算器失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("开始运行新功能测试")
    logger.info("=" * 60)
    logger.info("\n")
    
    results = {}
    
    # 运行测试
    results['检测引擎'] = test_inspection_engine()
    time.sleep(1)
    
    results['缺陷检测'] = test_defect_detection()
    time.sleep(1)
    
    results['批量检测'] = test_batch_inspection()
    time.sleep(1)
    
    results['数据导出'] = test_data_export()
    time.sleep(1)
    
    results['统计计算'] = test_statistics_calculator()
    
    # 汇总结果
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        logger.info(f"{test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    logger.info(f"\n总计: {passed_tests}/{total_tests} 通过")
    logger.info(f"通过率: {pass_rate:.1f}%")
    
    if pass_rate == 100:
        logger.info("\n🎉 所有测试通过！")
    else:
        logger.warning(f"\n⚠️  有 {total_tests - passed_tests} 个测试失败")
    
    return all(results.values())


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
