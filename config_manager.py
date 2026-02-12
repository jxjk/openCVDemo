# -*- coding: utf-8 -*-
"""
配置管理模块
Configuration Management Module
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

from exceptions import ConfigLoadError, ConfigValidationError


class ToleranceStandard(Enum):
    """公差标准枚举"""
    IT5 = "IT5"
    IT7 = "IT7"
    IT8 = "IT8"
    IT9 = "IT9"
    IT11 = "IT11"


@dataclass
class ToleranceTable:
    """公差表"""
    it5: Dict[tuple, float] = field(default_factory=lambda: {
        (1, 3): 0.004, (3, 6): 0.005, (6, 10): 0.006, (10, 18): 0.008,
        (18, 30): 0.009, (30, 40): 0.011
    })
    it7: Dict[tuple, float] = field(default_factory=lambda: {
        (1, 3): 0.010, (3, 6): 0.012, (6, 10): 0.015, (10, 18): 0.018,
        (18, 30): 0.021, (30, 40): 0.025
    })
    it8: Dict[tuple, float] = field(default_factory=lambda: {
        (1, 3): 0.014, (3, 6): 0.018, (6, 10): 0.022, (10, 18): 0.027,
        (18, 30): 0.033, (30, 40): 0.039
    })
    it9: Dict[tuple, float] = field(default_factory=lambda: {
        (1, 3): 0.025, (3, 6): 0.030, (6, 10): 0.036, (10, 18): 0.043,
        (18, 30): 0.052, (30, 40): 0.062
    })
    it11: Dict[tuple, float] = field(default_factory=lambda: {
        (1, 3): 0.060, (3, 6): 0.075, (6, 10): 0.090, (10, 18): 0.110,
        (18, 30): 0.130, (30, 40): 0.160
    })
    
    def get_tolerance(self, standard: str, nominal_value: float) -> float:
        """
        获取公差值
        
        Args:
            standard: 公差标准 (IT5, IT7, IT8, IT9, IT11)
            nominal_value: 标称值
        
        Returns:
            公差值（mm）
        """
        standard_enum = ToleranceStandard(standard)
        tolerance_dict = getattr(self, standard_enum.value.lower())
        
        for (min_val, max_val), tolerance in tolerance_dict.items():
            if min_val < nominal_value <= max_val:
                return tolerance
        
        # 如果找不到匹配的范围，使用最大范围的公差
        if tolerance_dict:
            return max(tolerance_dict.values())
        
        return 0.1  # 默认公差


@dataclass
class InspectionConfig:
    """检测配置"""
    # 像素到毫米转换比例
    pixel_to_mm: float = 0.098
    
    # 默认公差标准
    default_tolerance_standard: str = "IT8"
    
    # 亚像素检测精度
    subpixel_precision: float = 1.0 / 20.0  # 1/20像素
    
    # 检测参数
    min_circularity: float = 0.7
    min_contour_area: int = 100
    max_contour_area: int = 100000
    
    # 边缘检测参数
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    aperture_size: int = 3
    
    # 圆检测参数
    dp: float = 1.0
    min_dist: int = 20
    param1: int = 50
    param2: int = 30
    min_radius: int = 5
    max_radius: int = 100
    
    # RANSAC参数
    ransac_iterations: int = 1000
    ransac_threshold: float = 0.01
    
    # 公差表
    tolerance_table: ToleranceTable = field(default_factory=ToleranceTable)
    
    # 路径配置
    data_dir: str = "data"
    template_dir: str = "templates"
    image_dir: str = "data/images"
    
    # 文件名模式
    result_csv_pattern: str = "inspection_results_{date}.csv"
    error_csv_pattern: str = "inspection_errors_{date}.csv"
    calibration_file: str = "calibration_data.json"
    
    def get_tolerance(self, nominal_value: float, 
                     standard: Optional[str] = None) -> float:
        """
        获取公差值
        
        Args:
            nominal_value: 标称值
            standard: 公差标准（默认使用默认标准）
        
        Returns:
            公差值（mm）
        """
        if standard is None:
            standard = self.default_tolerance_standard
        
        return self.tolerance_table.get_tolerance(standard, nominal_value)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        data = asdict(self)
        # 将公差表转换为可序列化的格式
        # 将元组键转换为字符串键，如 (1, 3) -> "1-3"
        data['tolerance_table'] = {}
        for std_key in ['it5', 'it7', 'it8', 'it9', 'it11']:
            tolerance_dict = getattr(self.tolerance_table, std_key)
            serializable_dict = {}
            for (min_val, max_val), tolerance in tolerance_dict.items():
                # 将元组键转换为字符串键 "min-max"
                serializable_dict[f"{min_val}-{max_val}"] = tolerance
            data['tolerance_table'][std_key] = serializable_dict
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'InspectionConfig':
        """从字典创建配置"""
        # 转换公差表
        tolerance_table_data = data.get('tolerance_table', {})
        tolerance_table = ToleranceTable()
        
        for key in ['it5', 'it7', 'it8', 'it9', 'it11']:
            if key in tolerance_table_data:
                # 将字符串键转换为元组键，如 "1-3" -> (1, 3)
                tolerance_dict = {}
                for key_str, value in tolerance_table_data[key].items():
                    # key_str格式为"1-3"
                    min_val, max_val = map(float, key_str.split('-'))
                    tolerance_dict[(min_val, max_val)] = value
                setattr(tolerance_table, key, tolerance_dict)
        
        data['tolerance_table'] = tolerance_table
        return cls(**data)


class ConfigManager:
    """配置管理器"""
    
    _instance = None
    _config: Optional[InspectionConfig] = None
    _config_file: Optional[Path] = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self._config = None
        self._config_file = None
    
    @classmethod
    def load_config(cls, config_file: Optional[str] = None) -> InspectionConfig:
        """
        加载配置文件
        
        Args:
            config_file: 配置文件路径（默认：config.json）
        
        Returns:
            InspectionConfig实例
        
        Raises:
            ConfigLoadError: 配置加载失败
        """
        if config_file is None:
            config_file = "config.json"
        
        config_path = Path(config_file)
        
        if not config_path.exists():
            # 创建默认配置文件
            default_config = InspectionConfig()
            cls.save_config(default_config, str(config_path))
            cls._config = default_config
            cls._config_file = config_path
            return cls._config
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            cls._config = InspectionConfig.from_dict(data)
            cls._config_file = config_path
            
            return cls._config
        
        except json.JSONDecodeError as e:
            raise ConfigLoadError(
                f"配置文件格式错误: {config_file}",
                details={'error': str(e)}
            )
        except Exception as e:
            raise ConfigLoadError(
                f"加载配置文件失败: {config_file}",
                details={'error': str(e)}
            )
    
    @classmethod
    def save_config(cls, config: InspectionConfig, 
                    config_file: Optional[str] = None) -> bool:
        """
        保存配置文件
        
        Args:
            config: 配置对象
            config_file: 配置文件路径
        
        Returns:
            是否保存成功
        """
        if config_file is None:
            config_file = "config.json"
        
        config_path = Path(config_file)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
            
            cls._config = config
            cls._config_file = config_path
            
            return True
        
        except Exception as e:
            from logger_config import get_logger
            logger = get_logger(__name__)
            logger.error(f"保存配置文件失败: {e}")
            return False
    
    @classmethod
    def get_config(cls) -> InspectionConfig:
        """
        获取当前配置
        
        Returns:
            InspectionConfig实例
        """
        if cls._config is None:
            cls.load_config()
        
        return cls._config
    
    @classmethod
    def reload_config(cls) -> InspectionConfig:
        """
        重新加载配置文件
        
        Returns:
            InspectionConfig实例
        """
        if cls._config_file:
            return cls.load_config(cls._config_file)
        else:
            return cls.load_config()
    
    @classmethod
    def update_config(cls, **kwargs) -> bool:
        """
        更新配置项
        
        Args:
            **kwargs: 要更新的配置项
        
        Returns:
            是否更新成功
        """
        config = cls.get_config()
        
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return cls.save_config(config)
    
    @classmethod
    def get_tolerance(cls, nominal_value: float, 
                      standard: Optional[str] = None) -> float:
        """
        获取公差值（便捷方法）
        
        Args:
            nominal_value: 标称值
            standard: 公差标准
        
        Returns:
            公差值（mm）
        """
        config = cls.get_config()
        return config.get_tolerance(nominal_value, standard)


# =============================================================================
# 便捷函数
# =============================================================================

def get_config() -> InspectionConfig:
    """获取配置实例（便捷函数）"""
    return ConfigManager.get_config()


def get_tolerance(nominal_value: float, 
                 standard: Optional[str] = None) -> float:
    """获取公差值（便捷函数）"""
    return ConfigManager.get_tolerance(nominal_value, standard)


def update_config(**kwargs) -> bool:
    """更新配置（便捷函数）"""
    return ConfigManager.update_config(**kwargs)


def reload_config() -> InspectionConfig:
    """重新加载配置（便捷函数）"""
    return ConfigManager.reload_config()


def save_default_config() -> bool:
    """保存默认配置（便捷函数）"""
    config = InspectionConfig()
    return ConfigManager.save_config(config)