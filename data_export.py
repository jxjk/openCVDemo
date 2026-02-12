# -*- coding: utf-8 -*-
"""
数据导出和统计模块
Data Export and Statistics Module
"""

import os
import csv
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from exceptions import StorageException, FileSaveError
from logger_config import get_logger
from config_manager import get_config


class DataExporter:
    """数据导出器"""
    
    def __init__(self, data_dir: Optional[str] = None):
        self.config = get_config()
        self.data_dir = data_dir or self.config.data_dir
        self.logger = get_logger(self.__class__.__name__)
        
        # 确保目录存在
        os.makedirs(self.data_dir, exist_ok=True)
    
    def export_to_csv(self, data: List[Dict], filename: Optional[str] = None) -> str:
        """
        导出数据到CSV文件
        
        Args:
            data: 数据列表
            filename: 文件名（可选）
        
        Returns:
            文件路径
        """
        try:
            if not data:
                raise StorageException("没有数据可导出")
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d")
                filename = f"export_{timestamp}.csv"
            
            filepath = os.path.join(self.data_dir, filename)
            
            # 获取所有字段名
            fieldnames = set()
            for item in data:
                fieldnames.update(item.keys())
            fieldnames = sorted(fieldnames)
            
            # 写入CSV
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            
            self.logger.info(f"数据已导出到: {filepath}")
            
            return filepath
        
        except StorageException:
            raise
        except Exception as e:
            self.logger.error(f"导出CSV失败: {e}")
            raise FileSaveError(filepath, details={'error': str(e)})
    
    def export_to_excel(self, data: List[Dict], filename: Optional[str] = None) -> str:
        """
        导出数据到Excel文件
        
        Args:
            data: 数据列表
            filename: 文件名（可选）
        
        Returns:
            文件路径
        """
        try:
            if not data:
                raise StorageException("没有数据可导出")
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d")
                filename = f"export_{timestamp}.xlsx"
            
            filepath = os.path.join(self.data_dir, filename)
            
            # 使用pandas导出
            df = pd.DataFrame(data)
            df.to_excel(filepath, index=False, engine='openpyxl')
            
            self.logger.info(f"数据已导出到: {filepath}")
            
            return filepath
        
        except StorageException:
            raise
        except Exception as e:
            self.logger.error(f"导出Excel失败: {e}")
            raise FileSaveError(filepath, details={'error': str(e)})
    
    def export_to_json(self, data: List[Dict], filename: Optional[str] = None) -> str:
        """
        导出数据到JSON文件
        
        Args:
            data: 数据列表
            filename: 文件名（可选）
        
        Returns:
            文件路径
        """
        try:
            if not data:
                raise StorageException("没有数据可导出")
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d")
                filename = f"export_{timestamp}.json"
            
            filepath = os.path.join(self.data_dir, filename)
            
            # 写入JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"数据已导出到: {filepath}")
            
            return filepath
        
        except StorageException:
            raise
        except Exception as e:
            self.logger.error(f"导出JSON失败: {e}")
            raise FileSaveError(filepath, details={'error': str(e)})
    
    def export_statistics(self, data: List[Dict], filename: Optional[str] = None) -> str:
        """
        导出统计报告
        
        Args:
            data: 数据列表
            filename: 文件名（可选）
        
        Returns:
            文件路径
        """
        try:
            if not data:
                raise StorageException("没有数据可导出")
            
            # 计算统计数据
            stats = self.calculate_statistics(data)
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d")
                filename = f"statistics_{timestamp}.xlsx"
            
            filepath = os.path.join(self.data_dir, filename)
            
            # 导出为Excel，多个工作表
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # 原始数据
                df_data = pd.DataFrame(data)
                df_data.to_excel(writer, sheet_name='原始数据', index=False)
                
                # 统计汇总
                df_summary = pd.DataFrame([stats['summary']])
                df_summary.to_excel(writer, sheet_name='统计汇总', index=False)
                
                # 详细统计
                df_detailed = pd.DataFrame(stats['detailed'])
                df_detailed.to_excel(writer, sheet_name='详细统计', index=False)
            
            self.logger.info(f"统计报告已导出到: {filepath}")
            
            return filepath
        
        except StorageException:
            raise
        except Exception as e:
            self.logger.error(f"导出统计报告失败: {e}")
            raise FileSaveError(filepath, details={'error': str(e)})
    
    def calculate_statistics(self, data: List[Dict]) -> Dict:
        """
        计算统计数据
        
        Args:
            data: 数据列表
        
        Returns:
            统计结果字典
        """
        try:
            if not data:
                return {
                    'summary': {
                        '总检测数': 0,
                        '合格数': 0,
                        '不合格数': 0,
                        '合格率': 0.0,
                        '导出时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'error': '数据为空'
                    },
                    'detailed': [],
                    'by_type': None
                }
            
            df = pd.DataFrame(data)
            
            summary = {
                '总检测数': len(df),
                '合格数': len(df[df.get('is_passed', False) == True]),
                '不合格数': len(df[df.get('is_passed', False) == False]),
                '合格率': 0.0,
                '导出时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if summary['总检测数'] > 0:
                summary['合格率'] = summary['合格数'] / summary['总检测数'] * 100
            
            # 数值型字段的统计
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            detailed_stats = []
            
            for col in numeric_columns:
                col_stats = {
                    '字段': col,
                    '平均值': float(df[col].mean()) if not df[col].isna().all() else None,
                    '标准差': float(df[col].std()) if not df[col].isna().all() else None,
                    '最小值': float(df[col].min()) if not df[col].isna().all() else None,
                    '最大值': float(df[col].max()) if not df[col].isna().all() else None,
                    '中位数': float(df[col].median()) if not df[col].isna().all() else None
                }
                detailed_stats.append(col_stats)
            
            # 按特征类型统计
            if 'feature_type' in df.columns:
                type_stats = df.groupby('feature_type').agg({
                    'is_passed': ['count', 'sum']
                }).reset_index()
                type_stats.columns = ['特征类型', '总数', '合格数']
                type_stats['合格率'] = (type_stats['合格数'] / type_stats['总数'] * 100).round(2)
            
            return {
                'summary': summary,
                'detailed': detailed_stats,
                'by_type': type_stats if 'feature_type' in df.columns else None
            }
        
        except Exception as e:
            self.logger.error(f"计算统计数据失败: {e}")
            return {
                'summary': {'error': str(e)},
                'detailed': [],
                'by_type': None
            }


class InspectionDataExporter(DataExporter):
    """检测数据导出器"""
    
    def export_results(self, results: List[Dict], filename: Optional[str] = None) -> str:
        """
        导出检测结果
        
        Args:
            results: 检测结果列表
            filename: 文件名（可选）
        
        Returns:
            文件路径
        """
        return self.export_to_csv(results, filename)
    
    def export_errors(self, errors: List[Dict], filename: Optional[str] = None) -> str:
        """
        导出错误记录
        
        Args:
            errors: 错误记录列表
            filename: 文件名（可选）
        
        Returns:
            文件路径
        """
        return self.export_to_csv(errors, filename)
    
    def generate_report(self, results: List[Dict], filename: Optional[str] = None) -> str:
        """
        生成检测报告
        
        Args:
            results: 检测结果列表
            filename: 文件名（可选）
        
        Returns:
            文件路径
        """
        return self.export_statistics(results, filename)


class StatisticsCalculator:
    """统计计算器"""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def calculate_pass_rate(self, data: List[Dict]) -> Dict[str, float]:
        """
        计算合格率
        
        Args:
            data: 数据列表
        
        Returns:
            合格率字典
        """
        try:
            total = len(data)
            
            if total == 0:
                return {
                    'total': 0,
                    'passed': 0,
                    'failed': 0,
                    'pass_rate': 0.0
                }
            
            passed = sum(1 for item in data if item.get('is_passed', False))
            failed = total - passed
            
            return {
                'total': total,
                'passed': passed,
                'failed': failed,
                'pass_rate': (passed / total * 100) if total > 0 else 0.0
            }
        
        except Exception as e:
            self.logger.error(f"计算合格率失败: {e}")
            return {'error': str(e)}
    
    def calculate_statistics_by_field(self, data: List[Dict], field: str) -> Dict:
        """
        按字段计算统计
        
        Args:
            data: 数据列表
            field: 字段名
        
        Returns:
            统计结果
        """
        try:
            values = [item.get(field) for item in data if item.get(field) is not None]
            
            if not values:
                return {'error': f'没有找到有效数据: {field}'}
            
            return {
                'field': field,
                'count': len(values),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75))
            }
        
        except Exception as e:
            self.logger.error(f"计算统计失败: {e}")
            return {'error': str(e)}
    
    def calculate_cp_cpk(self, data: List[Dict], field: str, 
                        nominal: float, tolerance: float) -> Dict:
        """
        计算过程能力指数
        
        Args:
            data: 数据列表
            field: 字段名
            nominal: 标称值
            tolerance: 公差
        
        Returns:
            Cp, Cpk值
        """
        try:
            values = [item.get(field) for item in data if item.get(field) is not None]
            
            if len(values) < 2:
                return {'error': '数据不足，至少需要2个数据点'}
            
            mean = np.mean(values)
            std = np.std(values)
            
            # 规格限
            usl = nominal + tolerance  # 上规格限
            lsl = nominal - tolerance  # 下规格限
            
            # Cp = (USL - LSL) / (6 * σ)
            cp = (usl - lsl) / (6 * std) if std > 0 else 0.0
            
            # Cpk = min((USL - μ) / (3σ), (μ - LSL) / (3σ))
            cpk_upper = (usl - mean) / (3 * std) if std > 0 else 0.0
            cpk_lower = (mean - lsl) / (3 * std) if std > 0 else 0.0
            cpk = min(cpk_upper, cpk_lower)
            
            return {
                'field': field,
                'nominal': nominal,
                'tolerance': tolerance,
                'usl': usl,
                'lsl': lsl,
                'mean': float(mean),
                'std': float(std),
                'cp': float(cp),
                'cpk': float(cpk),
                'cpk_upper': float(cpk_upper),
                'cpk_lower': float(cpk_lower),
                'sample_size': len(values)
            }
        
        except Exception as e:
            self.logger.error(f"计算过程能力指数失败: {e}")
            return {'error': str(e)}


# 便捷函数
def export_data(data: List[Dict], format: str = 'csv', filename: Optional[str] = None) -> str:
    """
    导出数据（便捷函数）
    
    Args:
        data: 数据列表
        format: 格式 (csv, excel, json)
        filename: 文件名
    
    Returns:
        文件路径
    """
    exporter = DataExporter()
    
    if format == 'csv':
        return exporter.export_to_csv(data, filename)
    elif format == 'excel':
        return exporter.export_to_excel(data, filename)
    elif format == 'json':
        return exporter.export_to_json(data, filename)
    else:
        raise StorageException(f"不支持的格式: {format}")


def calculate_statistics(data: List[Dict]) -> Dict:
    """计算统计数据（便捷函数）"""
    exporter = DataExporter()
    return exporter.calculate_statistics(data)