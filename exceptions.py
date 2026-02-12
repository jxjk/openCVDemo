# -*- coding: utf-8 -*-
"""
异常类定义
Exception Classes for Micro Part Inspection System
"""

from typing import Optional, Any


class InspectionSystemException(Exception):
    """基础异常类"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details
        }


# ============================================================================
# 相机相关异常
# ============================================================================

class CameraException(InspectionSystemException):
    """相机异常"""
    pass


class CameraConnectionError(CameraException):
    """相机连接失败"""
    def __init__(self, camera_id: str, details: Optional[dict] = None):
        super().__init__(
            f"相机连接失败: {camera_id}",
            error_code="CAM001",
            details=details
        )
        self.camera_id = camera_id


class CameraDisconnectedError(CameraException):
    """相机断开"""
    def __init__(self, camera_id: str, details: Optional[dict] = None):
        super().__init__(
            f"相机已断开: {camera_id}",
            error_code="CAM002",
            details=details
        )
        self.camera_id = camera_id


class CameraCaptureError(CameraException):
    """图像采集失败"""
    def __init__(self, details: Optional[dict] = None):
        super().__init__(
            "图像采集失败",
            error_code="CAM003",
            details=details
        )


class CameraNotFoundError(CameraException):
    """相机未找到"""
    def __init__(self, camera_id: str, details: Optional[dict] = None):
        super().__init__(
            f"相机未找到: {camera_id}",
            error_code="CAM004",
            details=details
        )
        self.camera_id = camera_id


# ============================================================================
# 图像处理异常
# ============================================================================

class ImageProcessingException(InspectionSystemException):
    """图像处理异常"""
    pass


class ImageLoadError(ImageProcessingException):
    """图像加载失败"""
    def __init__(self, filepath: str, details: Optional[dict] = None):
        super().__init__(
            f"图像加载失败: {filepath}",
            error_code="IMG001",
            details=details
        )
        self.filepath = filepath


class ImagePreprocessingError(ImageProcessingException):
    """图像预处理失败"""
    def __init__(self, details: Optional[dict] = None):
        super().__init__(
            "图像预处理失败",
            error_code="IMG002",
            details=details
        )


class EdgeDetectionError(ImageProcessingException):
    """边缘检测失败"""
    def __init__(self, details: Optional[dict] = None):
        super().__init__(
            "边缘检测失败",
            error_code="IMG003",
            details=details
        )


class SubpixelDetectionError(ImageProcessingException):
    """亚像素检测失败"""
    def __init__(self, details: Optional[dict] = None):
        super().__init__(
            "亚像素检测失败",
            error_code="IMG004",
            details=details
        )


# ============================================================================
# 检测算法异常
# ============================================================================

class DetectionException(InspectionSystemException):
    """检测算法异常"""
    pass


class FeatureDetectionError(DetectionException):
    """特征检测失败"""
    def __init__(self, feature_type: str, details: Optional[dict] = None):
        super().__init__(
            f"特征检测失败: {feature_type}",
            error_code="DET001",
            details=details
        )
        self.feature_type = feature_type


class NoFeaturesFoundError(DetectionException):
    """未检测到特征"""
    def __init__(self, feature_type: str, details: Optional[dict] = None):
        super().__init__(
            f"未检测到{feature_type}特征",
            error_code="DET002",
            details=details
        )
        self.feature_type = feature_type


class CircleDetectionError(FeatureDetectionError):
    """圆形检测失败"""
    def __init__(self, details: Optional[dict] = None):
        super().__init__("圆形", details)


class LineDetectionError(FeatureDetectionError):
    """线段检测失败"""
    def __init__(self, details: Optional[dict] = None):
        super().__init__("线段", details)


class RectangleDetectionError(FeatureDetectionError):
    """矩形检测失败"""
    def __init__(self, details: Optional[dict] = None):
        super().__init__("矩形", details)


# ============================================================================
# 标定异常
# ============================================================================

class CalibrationException(InspectionSystemException):
    """标定异常"""
    pass


class CalibrationBoardNotFoundError(CalibrationException):
    """标定板未找到"""
    def __init__(self, details: Optional[dict] = None):
        super().__init__(
            "未找到标定板图像",
            error_code="CAL001",
            details=details
        )


class CalibrationFailedError(CalibrationException):
    """标定失败"""
    def __init__(self, details: Optional[dict] = None):
        super().__init__(
            "相机标定失败",
            error_code="CAL002",
            details=details
        )


class PixelToMmCalibrationError(CalibrationException):
    """像素-毫米标定失败"""
    def __init__(self, details: Optional[dict] = None):
        super().__init__(
            "像素-毫米转换标定失败",
            error_code="CAL003",
            details=details
        )


# ============================================================================
# DXF解析异常
# ============================================================================

class DXFException(InspectionSystemException):
    """DXF解析异常"""
    pass


class DXFLoadError(DXFException):
    """DXF文件加载失败"""
    def __init__(self, filepath: str, details: Optional[dict] = None):
        super().__init__(
            f"DXF文件加载失败: {filepath}",
            error_code="DXF001",
            details=details
        )
        self.filepath = filepath


class DXFParseError(DXFException):
    """DXF文件解析失败"""
    def __init__(self, details: Optional[dict] = None):
        super().__init__(
            "DXF文件解析失败",
            error_code="DXF002",
            details=details
        )


class DXFEntityNotFoundError(DXFException):
    """DXF实体未找到"""
    def __init__(self, entity_type: str, details: Optional[dict] = None):
        super().__init__(
            f"DXF中未找到实体: {entity_type}",
            error_code="DXF003",
            details=details
        )
        self.entity_type = entity_type


class DXFLibraryNotFoundError(DXFException):
    """ezdxf库未安装"""
    def __init__(self, details: Optional[dict] = None):
        super().__init__(
            "ezdxf库未安装，无法解析DXF文件",
            error_code="DXF004",
            details=details
        )


# ============================================================================
# 数据存储异常
# ============================================================================

class StorageException(InspectionSystemException):
    """数据存储异常"""
    pass


class FileSaveError(StorageException):
    """文件保存失败"""
    def __init__(self, filepath: str, details: Optional[dict] = None):
        super().__init__(
            f"文件保存失败: {filepath}",
            error_code="STO001",
            details=details
        )
        self.filepath = filepath


class FileLoadError(StorageException):
    """文件加载失败"""
    def __init__(self, filepath: str, details: Optional[dict] = None):
        super().__init__(
            f"文件加载失败: {filepath}",
            error_code="STO002",
            details=details
        )
        self.filepath = filepath


class DatabaseError(StorageException):
    """数据库错误"""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            f"数据库错误: {message}",
            error_code="STO003",
            details=details
        )


# ============================================================================
# 模板异常
# ============================================================================

class TemplateException(InspectionSystemException):
    """模板异常"""
    pass


class TemplateLoadError(TemplateException):
    """模板加载失败"""
    def __init__(self, filepath: str, details: Optional[dict] = None):
        super().__init__(
            f"模板加载失败: {filepath}",
            error_code="TPL001",
            details=details
        )
        self.filepath = filepath


class TemplateSaveError(TemplateException):
    """模板保存失败"""
    def __init__(self, filepath: str, details: Optional[dict] = None):
        super().__init__(
            f"模板保存失败: {filepath}",
            error_code="TPL002",
            details=details
        )
        self.filepath = filepath


class TemplateValidationError(TemplateException):
    """模板验证失败"""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            f"模板验证失败: {message}",
            error_code="TPL003",
            details=details
        )


# ============================================================================
# 配置异常
# ============================================================================

class ConfigurationException(InspectionSystemException):
    """配置异常"""
    pass


class ConfigLoadError(ConfigurationException):
    """配置加载失败"""
    def __init__(self, filepath: str, details: Optional[dict] = None):
        super().__init__(
            f"配置加载失败: {filepath}",
            error_code="CFG001",
            details=details
        )
        self.filepath = filepath


class ConfigValidationError(ConfigurationException):
    """配置验证失败"""
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            f"配置验证失败: {message}",
            error_code="CFG002",
            details=details
        )


# ============================================================================
# 工具函数
# ============================================================================

def handle_exception(exception: Exception, logger=None, reraise: bool = False) -> dict:
    """
    统一异常处理
    
    Args:
        exception: 捕获的异常
        logger: 日志记录器
        reraise: 是否重新抛出异常
    
    Returns:
        异常信息字典
    """
    if isinstance(exception, InspectionSystemException):
        error_info = exception.to_dict()
    else:
        error_info = {
            'error_code': 'UNK001',
            'message': str(exception),
            'details': {'type': type(exception).__name__}
        }
    
    if logger:
        logger.error(f"异常发生: {error_info}")
    
    if reraise:
        raise exception
    
    return error_info


def log_exception(logger, exception: Exception, context: str = ""):
    """
    记录异常到日志
    
    Args:
        logger: 日志记录器
        exception: 异常对象
        context: 上下文信息
    """
    import traceback
    
    error_msg = f"{context} - {type(exception).__name__}: {str(exception)}"
    
    if isinstance(exception, InspectionSystemException):
        logger.error(error_msg, extra=exception.details)
    else:
        logger.error(error_msg, exc_info=True)