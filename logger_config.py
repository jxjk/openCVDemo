# -*- coding: utf-8 -*-
"""
日志配置模块
Logging Configuration Module
"""

import os
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class InspectionSystemLogger:
    """检测系统日志管理器"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.logger = None
        self.log_dir = None
        self.log_file = None
    
    @classmethod
    def get_logger(cls, name: str = "inspection_system") -> logging.Logger:
        """
        获取日志记录器
        
        Args:
            name: 日志记录器名称
        
        Returns:
            logging.Logger实例
        """
        if not cls._initialized:
            cls._initialize()
        
        return logging.getLogger(name)
    
    @classmethod
    def _initialize(cls):
        """初始化日志系统"""
        if cls._initialized:
            return
        
        # 创建日志目录
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 设置日志文件名
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"inspection_system_{timestamp}.log"
        
        # 配置根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # 清除现有处理器
        root_logger.handlers.clear()
        
        # 创建格式化器
        detailed_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器（INFO级别）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
        
        # 文件处理器（DEBUG级别）
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        # 错误日志文件处理器
        error_log_file = log_dir / f"inspection_system_errors_{timestamp}.log"
        error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
        
        cls._instance = cls()
        cls._instance.logger = root_logger
        cls._instance.log_dir = log_dir
        cls._instance.log_file = log_file
        cls._initialized = True
        
        logging.info("=" * 60)
        logging.info("检测系统日志系统初始化完成")
        logging.info(f"日志目录: {log_dir}")
        logging.info(f"日志文件: {log_file}")
        logging.info("=" * 60)
    
    @classmethod
    def set_level(cls, level: str):
        """
        设置日志级别
        
        Args:
            level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        log_level = level_map.get(level.upper(), logging.INFO)
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # 同时设置控制台处理器级别
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(log_level)
        
        logging.info(f"日志级别已设置为: {level}")
    
    @classmethod
    def get_log_files(cls) -> list:
        """
        获取所有日志文件
        
        Returns:
            日志文件列表
        """
        if cls._instance and cls._instance.log_dir:
            return list(cls._instance.log_dir.glob("*.log"))
        return []


# ============================================================================
# 便捷函数
# ============================================================================

def get_logger(name: str = "inspection_system") -> logging.Logger:
    """
    获取日志记录器（便捷函数）
    
    Args:
        name: 日志记录器名称
    
    Returns:
        logging.Logger实例
    """
    return InspectionSystemLogger.get_logger(name)


def setup_logging(level: str = "INFO"):
    """
    设置日志系统
    
    Args:
        level: 日志级别
    """
    InspectionSystemLogger.set_level(level)


def log_exception(logger: logging.Logger, exception: Exception, context: str = ""):
    """
    记录异常
    
    Args:
        logger: 日志记录器
        exception: 异常对象
        context: 上下文信息
    """
    import traceback
    
    error_msg = f"{context} - {type(exception).__name__}: {str(exception)}"
    
    logger.error(error_msg, exc_info=True)
    
    # 如果是自定义异常，记录详细信息
    if hasattr(exception, 'details') and exception.details:
        logger.error(f"异常详情: {exception.details}")


def log_performance(logger: logging.Logger, operation: str, duration: float, 
                    success: bool = True, details: Optional[dict] = None):
    """
    记录性能日志
    
    Args:
        logger: 日志记录器
        operation: 操作名称
        duration: 持续时间（秒）
        success: 是否成功
        details: 额外详情
    """
    status = "成功" if success else "失败"
    perf_msg = f"[性能] {operation} - {status} - 耗时: {duration:.3f}秒"
    
    if details:
        perf_msg += f" - 详情: {details}"
    
    logger.info(perf_msg)


def log_api_call(logger: logging.Logger, endpoint: str, method: str, 
                  params: Optional[dict] = None, result: Optional[Any] = None,
                  success: bool = True, duration: Optional[float] = None):
    """
    记录API调用日志
    
    Args:
        logger: 日志记录器
        endpoint: API端点
        method: HTTP方法
        params: 请求参数
        result: 返回结果
        success: 是否成功
        duration: 持续时间
    """
    status = "成功" if success else "失败"
    api_msg = f"[API] {method} {endpoint} - {status}"
    
    if params:
        api_msg += f" - 参数: {params}"
    
    if result and success:
        api_msg += f" - 结果: {result}"
    
    if duration:
        api_msg += f" - 耗时: {duration:.3f}秒"
    
    logger.info(api_msg)


# 自动初始化日志系统
try:
    setup_logging("INFO")
except Exception as e:
    print(f"日志系统初始化失败: {e}")
    # 使用基本配置
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')