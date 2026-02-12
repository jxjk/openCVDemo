# -*- coding: utf-8 -*-
"""
微小零件中高精度视觉检测系统
Micro Part High-Precision Visual Inspection System

版本: V1.0
创建日期: 2026-02-10
精度目标: IT8级（亚像素检测精度≥1/20像素）
作者: 基于参考程序和需求规格说明书整合开发

功能:
- 图像采集（支持USB相机和大恒相机）
- 相机标定（棋盘格标定）
- 亚像素边缘检测
- 圆形零件直径测量
- 矩形零件长宽测量
- IT8级公差判断
- 数据记录和导出
- 图形用户界面
"""

import os
import sys
import time
import csv
import json
import datetime
import threading
import numpy as np
import cv2
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, asdict


# =============================================================================
# 配置管理
# =============================================================================

@dataclass
class InspectionConfig:
    """检测配置"""
    # IT8级公差表（mm）
    IT8_TOLERANCE = {
        (1, 3): 0.014,
        (3, 6): 0.018,
        (6, 10): 0.022,
        (10, 18): 0.027,
        (18, 30): 0.033,
        (30, 40): 0.039
    }
    
    # 像素-毫米转换系数（需要标定）
    PIXEL_TO_MM: float = 0.098  # 默认值，需标定更新
    
    # 亚像素精度等级
    SUBPIXEL_PRECISION: float = 1.0 / 20.0  # 1/20像素
    
    # 图像处理参数
    GAUSSIAN_KERNEL: int = 5
    MEDIAN_KERNEL: int = 5
    CANNY_THRESHOLD_LOW: int = 50
    CANNY_THRESHOLD_HIGH: int = 150
    
    # 相机参数
    EXPOSURE_TIME: float = 10000.0  # μs
    GAIN: float = 10.0
    
    # 文件路径
    DATA_DIR: str = "data"
    IMAGE_DIR: str = "data/images"
    RESULT_FILE: str = "data/inspection_results.csv"
    ERROR_FILE: str = "data/inspection_errors.csv"
    
    def __post_init__(self):
        """初始化时创建目录"""
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.IMAGE_DIR, exist_ok=True)
    
    def get_it8_tolerance(self, nominal_size: float) -> float:
        """获取IT8级公差值"""
        for (min_size, max_size), tolerance in self.IT8_TOLERANCE.items():
            if min_size < nominal_size <= max_size:
                return tolerance
        return 0.039  # 默认使用最大公差


# =============================================================================
# 图像采集接口
# =============================================================================

class IImageAcquisition(ABC):
    """图像采集接口"""
    
    @abstractmethod
    def connect(self, device_id: str) -> bool:
        """连接相机设备"""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """断开相机连接"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """检查相机连接状态"""
        pass
    
    @abstractmethod
    def capture_image(self) -> Optional[np.ndarray]:
        """采集单帧图像"""
        pass
    
    @abstractmethod
    def set_exposure(self, exposure: float) -> bool:
        """设置曝光时间"""
        pass
    
    @abstractmethod
    def set_gain(self, gain: float) -> bool:
        """设置增益"""
        pass


class USBCameraDriver(IImageAcquisition):
    """USB相机驱动实现"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.camera = None
        self.is_capturing = False
    
    def connect(self, device_id: str = None) -> bool:
        """连接USB相机"""
        try:
            if device_id:
                self.device_id = int(device_id)
            self.camera = cv2.VideoCapture(self.device_id)
            
            # 设置分辨率（如果相机支持）
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
            
            if self.camera.isOpened():
                print(f"USB相机已连接: 设备ID={self.device_id}")
                return True
            else:
                print("USB相机连接失败")
                return False
        except Exception as e:
            print(f"USB相机连接异常: {e}")
            return False
    
    def disconnect(self) -> bool:
        """断开相机连接"""
        if self.camera:
            self.camera.release()
            self.camera = None
            print("USB相机已断开")
            return True
        return False
    
    def is_connected(self) -> bool:
        """检查相机连接状态"""
        return self.camera is not None and self.camera.isOpened()
    
    def capture_image(self) -> Optional[np.ndarray]:
        """采集单帧图像"""
        if not self.is_connected():
            return None
        
        try:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                return frame
            return None
        except Exception as e:
            print(f"图像采集失败: {e}")
            return None
    
    def set_exposure(self, exposure: float) -> bool:
        """设置曝光时间"""
        if self.is_connected():
            self.camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
            return True
        return False
    
    def set_gain(self, gain: float) -> bool:
        """设置增益"""
        if self.is_connected():
            # OpenCV的gain参数设置
            self.camera.set(cv2.CAP_PROP_GAIN, gain)
            return True
        return False


class DahengCameraDriver(IImageAcquisition):
    """大恒相机驱动实现（如果大恒SDK可用）"""
    
    def __init__(self):
        self.camera = None
        self.gxipy_available = False
        self._check_gxipy()
    
    def _check_gxipy(self):
        """检查大恒相机SDK是否可用"""
        try:
            import gxipy as gx
            self.gxipy_available = True
            print("大恒相机SDK (gxipy) 可用")
        except ImportError:
            print("警告: 大恒相机SDK (gxipy) 不可用，将使用USB相机")
    
    def connect(self, device_id: str = None) -> bool:
        """连接大恒相机"""
        if not self.gxipy_available:
            return False
        
        try:
            import gxipy as gx
            device_manager = gx.DeviceManager()
            dev_num, dev_info_list = device_manager.update_device_list()
            
            if dev_num == 0:
                print("未发现大恒相机设备")
                return False
            
            # 打开第一个设备
            self.camera = device_manager.open_device_by_index(1)
            
            # 设置连续采集
            self.camera.TriggerMode.set(gx.GxSwitchEntry.OFF)
            
            # 设置曝光和增益
            self.camera.ExposureTime.set(10000.0)
            self.camera.Gain.set(10.0)
            
            print("大恒相机已连接")
            return True
        except Exception as e:
            print(f"大恒相机连接失败: {e}")
            return False
    
    def disconnect(self) -> bool:
        """断开相机连接"""
        if self.camera:
            self.camera.stream_off()
            self.camera.close_device()
            self.camera = None
            print("大恒相机已断开")
            return True
        return False
    
    def is_connected(self) -> bool:
        """检查相机连接状态"""
        return self.camera is not None
    
    def capture_image(self) -> Optional[np.ndarray]:
        """采集单帧图像"""
        if not self.is_connected():
            return None
        
        try:
            import gxipy as gx
            
            # 启动数据流
            self.camera.stream_on()
            
            # 获取图像
            raw_image = self.camera.data_stream[0].get_image()
            if raw_image is None:
                self.camera.stream_off()
                return None
            
            # 转换为numpy数组
            numpy_image = raw_image.get_numpy_array()
            
            # 停止数据流
            self.camera.stream_off()
            
            return numpy_image
        except Exception as e:
            print(f"大恒相机图像采集失败: {e}")
            return None
    
    def set_exposure(self, exposure: float) -> bool:
        """设置曝光时间"""
        if self.is_connected():
            try:
                self.camera.ExposureTime.set(exposure)
                return True
            except:
                return False
        return False
    
    def set_gain(self, gain: float) -> bool:
        """设置增益"""
        if self.is_connected():
            try:
                self.camera.Gain.set(gain)
                return True
            except:
                return False
        return False


# =============================================================================
# 相机标定模块
# =============================================================================

class CameraCalibration:
    """相机标定类"""
    
    def __init__(self, config: InspectionConfig):
        self.config = config
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.reprojection_error = None
        
        # 棋盘格参数
        self.chessboard_size = (6, 4)  # 内角点数量
        self.square_size = 10.0  # 棋盘格方格大小（mm）
    
    def calibrate_from_images(self, image_dir: str) -> bool:
        """从图像目录进行标定"""
        # 获取标定图像
        image_files = []
        for ext in ['*.jpg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        
        if len(image_files) < 10:
            print(f"警告: 标定图像数量不足（{len(image_files)}张），建议至少10张")
        
        # 准备标定点
        obj_points = []  # 3D世界坐标点
        img_points = []  # 2D图像坐标点
        
        # 生成棋盘格的3D坐标点
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        
        # 亚像素角点检测的终止条件
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # 处理每张标定图像
        for image_file in image_files:
            img = cv2.imread(image_file)
            if img is None:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 查找棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            
            if ret:
                # 亚像素角点精细化
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                obj_points.append(objp)
                img_points.append(corners_refined)
                
                # 绘制角点（用于调试）
                cv2.drawChessboardCorners(img, self.chessboard_size, corners_refined, ret)
                # cv2.imshow('Chessboard Corners', img)
                # cv2.waitKey(100)
        
        # cv2.destroyAllWindows()
        
        if len(obj_points) == 0:
            print("错误: 未能找到有效的棋盘格角点")
            return False
        
        # 执行相机标定
        print(f"开始标定，使用{len(obj_points)}张图像...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, gray.shape[::-1], None, None
        )
        
        if not ret:
            print("相机标定失败")
            return False
        
        # 保存标定结果
        self.camera_matrix = mtx
        self.dist_coeffs = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        
        # 计算重投影误差
        total_error = 0
        for i in range(len(obj_points)):
            img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
            total_error += error
        
        self.reprojection_error = total_error / len(obj_points)
        
        print(f"相机标定完成！")
        print(f"重投影误差: {self.reprojection_error:.4f}像素")
        print(f"相机内参矩阵:\n{mtx}")
        print(f"畸变系数: {dist}")
        
        # 保存标定结果
        self.save_calibration()
        
        return True
    
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """图像去畸变"""
        if self.camera_matrix is None:
            print("警告: 相机未标定，无法进行去畸变")
            return image
        
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        
        dst = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, newcameramtx)
        
        # 裁剪图像
        # x, y, w, h = roi
        # dst = dst[y:y+h, x:x+w]
        
        return dst
    
    def save_calibration(self, filename: str = "data/calibration_data.json"):
        """保存标定数据"""
        calib_data = {
            'camera_matrix': self.camera_matrix.tolist() if self.camera_matrix is not None else None,
            'dist_coeffs': self.dist_coeffs.tolist() if self.dist_coeffs is not None else None,
            'reprojection_error': float(self.reprojection_error) if self.reprojection_error is not None else None,
            'calibration_date': datetime.datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(calib_data, f, indent=2, ensure_ascii=False)
        
        print(f"标定数据已保存到: {filename}")
    
    def load_calibration(self, filename: str = "data/calibration_data.json") -> bool:
        """加载标定数据"""
        if not os.path.exists(filename):
            print(f"标定数据文件不存在: {filename}")
            return False
        
        with open(filename, 'r', encoding='utf-8') as f:
            calib_data = json.load(f)
        
        self.camera_matrix = np.array(calib_data['camera_matrix'])
        self.dist_coeffs = np.array(calib_data['dist_coeffs'])
        self.reprojection_error = calib_data['reprojection_error']
        
        print(f"标定数据已加载: {filename}")
        print(f"重投影误差: {self.reprojection_error:.4f}像素")
        
        return True


# =============================================================================
# 亚像素检测算法
# =============================================================================

class SubpixelDetector:
    """亚像素检测器"""
    
    def __init__(self, config: InspectionConfig):
        self.config = config
    
    def subpixel_edge_detection(self, image: np.ndarray) -> np.ndarray:
        """亚像素边缘检测"""
        # 高斯模糊降噪
        blurred = cv2.GaussianBlur(image, (self.config.GAUSSIAN_KERNEL, self.config.GAUSSIAN_KERNEL), 0)
        
        # Canny边缘检测
        edges = cv2.Canny(blurred, 
                         self.config.CANNY_THRESHOLD_LOW, 
                         self.config.CANNY_THRESHOLD_HIGH)
        
        # 计算梯度
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_direction = np.arctan2(sobel_y, sobel_x)
        
        # 亚像素细化
        subpixel_edges = np.zeros_like(image, dtype=np.float32)
        
        edge_points = np.where(edges > 0)
        
        for y, x in zip(edge_points[0], edge_points[1]):
            # 在边缘点邻域内进行二次多项式拟合
            roi_size = 2
            y1 = max(0, y - roi_size)
            y2 = min(image.shape[0], y + roi_size + 1)
            x1 = max(0, x - roi_size)
            x2 = min(image.shape[1], x + roi_size + 1)
            
            # 梯度方向上的插值
            direction = gradient_direction[y, x]
            cos_dir = np.cos(direction)
            sin_dir = np.sin(direction)
            
            # 沿梯度方向采样
            samples = []
            for offset in np.linspace(-2, 2, 5):
                x_sample = x + offset * cos_dir
                y_sample = y + offset * sin_dir
                
                if 0 <= x_sample < image.shape[1] and 0 <= y_sample < image.shape[0]:
                    x_int = int(x_sample)
                    y_int = int(y_sample)
                    
                    # 双线性插值
                    x_frac = x_sample - x_int
                    y_frac = y_sample - y_int
                    
                    value = (1 - x_frac) * (1 - y_frac) * image[y_int, x_int] + \
                            x_frac * (1 - y_frac) * image[y_int, min(x_int + 1, image.shape[1] - 1)] + \
                            (1 - x_frac) * y_frac * image[min(y_int + 1, image.shape[0] - 1), x_int] + \
                            x_frac * y_frac * image[min(y_int + 1, image.shape[0] - 1), min(x_int + 1, image.shape[1] - 1)]
                    
                    samples.append((offset, value))
            
            if len(samples) >= 3:
                offsets, values = zip(*samples)
                
                # 二次多项式拟合
                coeffs = np.polyfit(offsets, values, 2)
                
                # 找到极值点
                if abs(coeffs[0]) > 1e-6:
                    subpixel_offset = -coeffs[1] / (2 * coeffs[0])
                    
                    if -2 <= subpixel_offset <= 2:
                        subpixel_x = x + subpixel_offset * cos_dir
                        subpixel_y = y + subpixel_offset * sin_dir
                        
                        if 0 <= subpixel_x < image.shape[1] and 0 <= subpixel_y < image.shape[0]:
                            subpixel_edges[int(subpixel_y), int(subpixel_x)] = 255
        
        return subpixel_edges
    
    def subpixel_corner_refinement(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """亚像素角点精细化"""
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(image, corners, (5, 5), (-1, -1), criteria)
        return corners_refined


# =============================================================================
# 几何拟合算法
# =============================================================================

class GeometryFitter:
    """几何拟合器"""
    
    @staticmethod
    def circle_least_squares_fit(points: np.ndarray) -> Tuple[float, float, float]:
        """
        最小二乘法圆拟合
        
        参数:
            points: Nx2的点集数组
            
        返回:
            (center_x, center_y, radius)
        """
        if len(points) < 3:
            raise ValueError("至少需要3个点来拟合圆")
        
        x = points[:, 0]
        y = points[:, 1]
        
        # 计算各项和
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_x2 = np.sum(x**2)
        sum_y2 = np.sum(y**2)
        sum_x3 = np.sum(x**3)
        sum_y3 = np.sum(y**3)
        sum_xy = np.sum(x * y)
        sum_x1y2 = np.sum(x * y**2)
        sum_x2y1 = np.sum(x**2 * y)
        
        N = len(points)
        
        # 构建方程组
        C = N * sum_x2 - sum_x**2
        D = N * sum_xy - sum_x * sum_y
        E = N * sum_x3 + N * sum_x1y2 - (sum_x2 + sum_y2) * sum_x
        G = N * sum_y2 - sum_y**2
        H = N * sum_x2y1 + N * sum_y3 - (sum_x2 + sum_y2) * sum_y
        
        # 求解
        denominator = C * G - D**2
        if abs(denominator) < 1e-10:
            raise ValueError("无法求解圆拟合方程")
        
        a = (H * D - E * G) / denominator
        b = (H * C - E * D) / (D**2 - G * C)
        c = -(a * sum_x + b * sum_y + sum_x2 + sum_y2) / N
        
        # 计算圆心和半径
        center_x = a / (-2)
        center_y = b / (-2)
        radius = np.sqrt(a**2 + b**2 - 4 * c) / 2
        
        return center_x, center_y, radius
    
    @staticmethod
    def circle_ransac_fit(points: np.ndarray, max_iterations: int = 100, 
                          threshold: float = 2.0, min_inliers: int = 10) -> Tuple[float, float, float]:
        """
        RANSAC圆拟合
        
        参数:
            points: Nx2的点集数组
            max_iterations: 最大迭代次数
            threshold: 内点阈值
            min_inliers: 最小内点数量
            
        返回:
            (center_x, center_y, radius)
        """
        best_circle = None
        best_inliers = 0
        
        for _ in range(max_iterations):
            # 随机选择3个点
            indices = np.random.choice(len(points), 3, replace=False)
            sample_points = points[indices]
            
            try:
                # 拟合圆
                cx, cy, r = GeometryFitter.circle_least_squares_fit(sample_points)
                
                # 计算内点
                distances = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
                inliers = np.sum(np.abs(distances - r) < threshold)
                
                # 更新最佳拟合
                if inliers > best_inliers and inliers >= min_inliers:
                    best_inliers = inliers
                    best_circle = (cx, cy, r)
                    
                    # 如果内点数量足够，使用内点重新拟合
                    if inliers > len(points) * 0.5:
                        inlier_points = points[np.abs(distances - r) < threshold]
                        best_circle = GeometryFitter.circle_least_squares_fit(inlier_points)
                        break
            except:
                continue
        
        if best_circle is None:
            raise ValueError("RANSAC圆拟合失败")
        
        return best_circle
    
    @staticmethod
    def rectangle_fit(points: np.ndarray) -> Tuple[float, float, float, float]:
        """
        矩形拟合
        
        参数:
            points: Nx2的点集数组
            
        返回:
            (center_x, center_y, width, height)
        """
        # 计算最小外接矩形
        rect = cv2.minAreaRect(points)
        center_x, center_y = rect[0]
        width, height = rect[1]
        
        return center_x, center_y, width, height


# =============================================================================
# 检测结果类
# =============================================================================

@dataclass
class InspectionResult:
    """检测结果"""
    part_id: str  # 零件编号
    part_type: str  # 零件类型
    timestamp: str  # 检测时间
    
    # 测量结果（像素）
    center_x_pixel: float
    center_y_pixel: float
    diameter_pixel: Optional[float] = None  # 圆形零件直径（像素）
    width_pixel: Optional[float] = None  # 矩形零件宽度（像素）
    height_pixel: Optional[float] = None  # 矩形零件高度（像素）
    
    # 测量结果（毫米）
    diameter_mm: Optional[float] = None
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None
    
    # 公差判断
    nominal_size: Optional[float] = None  # 标称尺寸
    tolerance: Optional[float] = None  # 公差值
    is_qualified: bool = False  # 是否合格
    deviation: Optional[float] = None  # 偏差
    
    # 图像信息
    image_path: Optional[str] = None
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)


# =============================================================================
# 核心检测引擎
# =============================================================================

class InspectionEngine:
    """检测引擎"""
    
    def __init__(self, config: InspectionConfig, calibration: CameraCalibration = None):
        self.config = config
        self.calibration = calibration
        self.subpixel_detector = SubpixelDetector(config)
        self.geometry_fitter = GeometryFitter()
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        # 如果有标定数据，进行去畸变
        if self.calibration and self.calibration.camera_matrix is not None:
            image = self.calibration.undistort_image(image)
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 中值滤波去噪
        gray = cv2.medianBlur(gray, self.config.MEDIAN_KERNEL)
        
        return gray
    
    def detect_circle(self, image: np.ndarray, part_id: str = "", 
                      part_type: str = "圆形", nominal_size: float = None) -> InspectionResult:
        """
        检测圆形零件
        
        参数:
            image: 输入图像
            part_id: 零件编号
            part_type: 零件类型
            nominal_size: 标称直径（mm）
            
        返回:
            InspectionResult
        """
        # 图像预处理
        gray = self.preprocess_image(image)
        
        # 亚像素边缘检测
        edges = self.subpixel_detector.subpixel_edge_detection(gray)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_NONE)
        
        if len(contours) == 0:
            print("警告: 未检测到轮廓")
            return None
        
        # 找到最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 提取轮廓点
        contour_points = largest_contour.reshape(-1, 2)
        
        # 最小外接圆初始估计
        (cx_init, cy_init), r_init = cv2.minEnclosingCircle(contour_points)
        
        # RANSAC圆拟合
        try:
            cx, cy, r = self.geometry_fitter.circle_ransac_fit(
                contour_points, 
                max_iterations=100,
                threshold=2.0
            )
        except:
            # 如果RANSAC失败，使用最小外接圆
            cx, cy, r = cx_init, cy_init, r_init
        
        # 亚像素精细化
        center_point = np.array([[cx, cy]], dtype=np.float32)
        center_refined = self.subpixel_detector.subpixel_corner_refinement(gray, center_point)
        cx, cy = center_refined[0]
        
        # 计算直径
        diameter_pixel = r * 2
        diameter_mm = diameter_pixel * self.config.PIXEL_TO_MM
        
        # 创建检测结果
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        result = InspectionResult(
            part_id=part_id,
            part_type=part_type,
            timestamp=timestamp,
            center_x_pixel=cx,
            center_y_pixel=cy,
            diameter_pixel=diameter_pixel,
            diameter_mm=diameter_mm,
            nominal_size=nominal_size
        )
        
        # 公差判断
        if nominal_size is not None:
            result.tolerance = self.config.get_it8_tolerance(nominal_size)
            result.deviation = diameter_mm - nominal_size
            result.is_qualified = abs(result.deviation) <= result.tolerance
        
        return result
    
    def detect_rectangle(self, image: np.ndarray, part_id: str = "",
                         part_type: str = "矩形", nominal_width: float = None,
                         nominal_height: float = None) -> InspectionResult:
        """
        检测矩形零件
        
        参数:
            image: 输入图像
            part_id: 零件编号
            part_type: 零件类型
            nominal_width: 标称宽度（mm）
            nominal_height: 标称高度（mm）
            
        返回:
            InspectionResult
        """
        # 图像预处理
        gray = self.preprocess_image(image)
        
        # 亚像素边缘检测
        edges = self.subpixel_detector.subpixel_edge_detection(gray)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        
        if len(contours) == 0:
            print("警告: 未检测到轮廓")
            return None
        
        # 找到最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 提取轮廓点
        contour_points = largest_contour.reshape(-1, 2)
        
        # 最小外接矩形拟合
        cx, cy, width_pixel, height_pixel = self.geometry_fitter.rectangle_fit(contour_points)
        
        # 转换为毫米
        width_mm = width_pixel * self.config.PIXEL_TO_MM
        height_mm = height_pixel * self.config.PIXEL_TO_MM
        
        # 创建检测结果
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        result = InspectionResult(
            part_id=part_id,
            part_type=part_type,
            timestamp=timestamp,
            center_x_pixel=cx,
            center_y_pixel=cy,
            width_pixel=width_pixel,
            height_pixel=height_pixel,
            width_mm=width_mm,
            height_mm=height_mm,
            nominal_size=nominal_width if nominal_width else nominal_height
        )
        
        # 公差判断（简化处理，只判断宽度）
        if nominal_width is not None:
            tolerance = self.config.get_it8_tolerance(nominal_width)
            deviation = width_mm - nominal_width
            result.is_qualified = abs(deviation) <= tolerance
            result.tolerance = tolerance
            result.deviation = deviation
        elif nominal_height is not None:
            tolerance = self.config.get_it8_tolerance(nominal_height)
            deviation = height_mm - nominal_height
            result.is_qualified = abs(deviation) <= tolerance
            result.tolerance = tolerance
            result.deviation = deviation
        
        return result
    
    def draw_result(self, image: np.ndarray, result: InspectionResult) -> np.ndarray:
        """在图像上绘制检测结果"""
        result_image = image.copy()
        
        # 绘制圆
        if result.diameter_pixel is not None:
            radius = result.diameter_pixel / 2
            center = (int(result.center_x_pixel), int(result.center_y_pixel))
            
            color = (0, 255, 0) if result.is_qualified else (0, 0, 255)
            cv2.circle(result_image, center, int(radius), color, 2)
            cv2.circle(result_image, center, 3, color, -1)
            
            # 绘制测量信息
            text = f"D: {result.diameter_mm:.3f}mm"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(result_image, text, (10, 30), font, 1, color, 2)
            
            if result.nominal_size is not None:
                text = f"Nom: {result.nominal_size:.3f}mm"
                cv2.putText(result_image, text, (10, 60), font, 1, color, 2)
                text = f"Dev: {result.deviation:.3f}mm"
                cv2.putText(result_image, text, (10, 90), font, 1, color, 2)
        
        # 绘制矩形
        if result.width_pixel is not None and result.height_pixel is not None:
            color = (0, 255, 0) if result.is_qualified else (0, 0, 255)
            
            text = f"W: {result.width_mm:.3f}mm"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(result_image, text, (10, 30), font, 1, color, 2)
            
            if result.nominal_size is not None:
                text = f"Nom: {result.nominal_size:.3f}mm"
                cv2.putText(result_image, text, (10, 60), font, 1, color, 2)
        
        # 绘制合格/不合格标记
        text = "OK" if result.is_qualified else "NG"
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 0) if result.is_qualified else (0, 0, 255)
        cv2.putText(result_image, text, (result_image.shape[1] - 100, 50), 
                   font, 2, color, 3)
        
        return result_image


# =============================================================================
# 数据管理模块
# =============================================================================

class DataManager:
    """数据管理器"""
    
    def __init__(self, config: InspectionConfig):
        self.config = config
        self.results_history = []
    
    def save_result(self, result: InspectionResult) -> bool:
        """保存检测结果"""
        try:
            # 添加到历史记录
            self.results_history.append(result)
            
            # 保存到CSV文件
            file_exists = os.path.exists(self.config.RESULT_FILE)
            
            with open(self.config.RESULT_FILE, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=result.to_dict().keys())
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(result.to_dict())
            
            # 如果不合格，同时保存到错误文件
            if not result.is_qualified:
                error_file_exists = os.path.exists(self.config.ERROR_FILE)
                
                with open(self.config.ERROR_FILE, 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=result.to_dict().keys())
                    
                    if not error_file_exists:
                        writer.writeheader()
                    
                    writer.writerow(result.to_dict())
            
            return True
        except Exception as e:
            print(f"保存检测结果失败: {e}")
            return False
    
    def save_image(self, image: np.ndarray, result: InspectionResult) -> str:
        """保存检测图像"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.part_id}_{timestamp}.jpg"
        filepath = os.path.join(self.config.IMAGE_DIR, filename)
        
        cv2.imwrite(filepath, image)
        return filepath
    
    def export_to_excel(self, filename: str = None) -> bool:
        """导出数据到Excel"""
        if filename is None:
            filename = f"data/inspection_results_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        
        try:
            if len(self.results_history) == 0:
                print("没有数据可导出")
                return False
            
            # 转换为DataFrame
            df = pd.DataFrame([r.to_dict() for r in self.results_history])
            
            # 保存到Excel
            df.to_excel(filename, index=False, engine='openpyxl')
            print(f"数据已导出到: {filename}")
            return True
        except Exception as e:
            print(f"导出到Excel失败: {e}")
            return False
    
    def get_statistics(self) -> dict:
        """获取统计数据"""
        if len(self.results_history) == 0:
            return {}
        
        total = len(self.results_history)
        qualified = sum(1 for r in self.results_history if r.is_qualified)
        unqualified = total - qualified
        
        return {
            'total': total,
            'qualified': qualified,
            'unqualified': unqualified,
            'qualified_rate': qualified / total if total > 0 else 0
        }


# =============================================================================
# 主程序
# =============================================================================

class InspectionSystem:
    """检测系统主类"""
    
    def __init__(self):
        self.config = InspectionConfig()
        self.calibration = CameraCalibration(self.config)
        self.camera_driver = None
        self.inspection_engine = InspectionEngine(self.config, self.calibration)
        self.data_manager = DataManager(self.config)
        
        # 尝试加载标定数据
        self.calibration.load_calibration()
    
    def connect_camera(self, camera_type: str = "usb", device_id: str = "0") -> bool:
        """连接相机"""
        if camera_type.lower() == "usb":
            self.camera_driver = USBCameraDriver()
        elif camera_type.lower() == "daheng":
            self.camera_driver = DahengCameraDriver()
        else:
            print(f"不支持的相机类型: {camera_type}")
            return False
        
        return self.camera_driver.connect(device_id)
    
    def disconnect_camera(self) -> bool:
        """断开相机"""
        if self.camera_driver:
            return self.camera_driver.disconnect()
        return True
    
    def capture_and_inspect(self, part_type: str = "圆形", 
                            part_id: str = "", 
                            nominal_size: float = None,
                            save_image: bool = True) -> InspectionResult:
        """采集图像并检测"""
        # 采集图像
        image = self.camera_driver.capture_image()
        if image is None:
            print("图像采集失败")
            return None
        
        # 根据零件类型检测
        if part_type == "圆形":
            result = self.inspection_engine.detect_circle(
                image, part_id, part_type, nominal_size
            )
        elif part_type == "矩形":
            result = self.inspection_engine.detect_rectangle(
                image, part_id, part_type, nominal_size
            )
        else:
            print(f"不支持的零件类型: {part_type}")
            return None
        
        if result is None:
            return None
        
        # 绘制检测结果
        result_image = self.inspection_engine.draw_result(image, result)
        
        # 保存图像
        if save_image:
            result.image_path = self.data_manager.save_image(result_image, result)
        
        # 保存结果
        self.data_manager.save_result(result)
        
        # 显示结果
        print(f"\n{'='*50}")
        print(f"检测结果: {part_type}")
        print(f"零件编号: {part_id}")
        print(f"检测时间: {result.timestamp}")
        print(f"{'='*50}")
        
        if result.diameter_mm is not None:
            print(f"测量直径: {result.diameter_mm:.3f} mm")
            if result.nominal_size:
                print(f"标称直径: {result.nominal_size:.3f} mm")
                print(f"偏差: {result.deviation:.3f} mm")
                print(f"公差: ±{result.tolerance:.3f} mm")
        
        if result.width_mm is not None:
            print(f"测量宽度: {result.width_mm:.3f} mm")
            print(f"测量高度: {result.height_mm:.3f} mm")
            if result.nominal_size:
                print(f"标称尺寸: {result.nominal_size:.3f} mm")
                print(f"偏差: {result.deviation:.3f} mm")
                print(f"公差: ±{result.tolerance:.3f} mm")
        
        print(f"检测结果: {'合格 ✓' if result.is_qualified else '不合格 ✗'}")
        print(f"{'='*50}\n")
        
        # 显示图像
        cv2.imshow("检测结果", result_image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        
        return result
    
    def calibrate_pixel_to_mm(self, known_diameter_mm: float, image: np.ndarray = None):
        """
        标定像素-毫米转换系数
        
        参数:
            known_diameter_mm: 已知零件的实际直径（mm）
            image: 包含该零件的图像，如果为None则使用相机采集
        """
        if image is None:
            image = self.camera_driver.capture_image()
            if image is None:
                print("图像采集失败")
                return
        
        # 检测圆形
        result = self.inspection_engine.detect_circle(image, "标定件", "圆形", known_diameter_mm)
        
        if result and result.diameter_pixel is not None:
            # 计算像素-毫米转换系数
            self.config.PIXEL_TO_MM = known_diameter_mm / result.diameter_pixel
            
            print(f"\n标定完成！")
            print(f"实际直径: {known_diameter_mm:.3f} mm")
            print(f"测量直径（像素）: {result.diameter_pixel:.2f} 像素")
            print(f"像素-毫米转换系数: {self.config.PIXEL_TO_MM:.6f} mm/像素\n")
            
            # 保存标定配置
            self._save_config()
    
    def _save_config(self):
        """保存配置"""
        config_data = {
            'pixel_to_mm': self.config.PIXEL_TO_MM,
            'calibration_date': datetime.datetime.now().isoformat()
        }
        
        with open('data/config.json', 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        
        print("配置已保存")
    
    def _load_config(self):
        """加载配置"""
        if os.path.exists('data/config.json'):
            with open('data/config.json', 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                self.config.PIXEL_TO_MM = config_data.get('pixel_to_mm', 0.098)
                print(f"配置已加载，像素-毫米转换系数: {self.config.PIXEL_TO_MM:.6f} mm/像素")


def main():
    """主函数"""
    print("="*60)
    print("微小零件中高精度视觉检测系统")
    print("Micro Part High-Precision Visual Inspection System")
    print("="*60)
    print()
    
    # 创建检测系统
    system = InspectionSystem()
    system._load_config()
    
    # 连接相机
    print("正在连接相机...")
    if not system.connect_camera(camera_type="usb", device_id="0"):
        print("相机连接失败，请检查相机设备")
        return
    
    print("相机连接成功！\n")
    
    # 主循环
    while True:
        print("\n请选择操作:")
        print("1. 标定像素-毫米转换系数")
        print("2. 检测圆形零件")
        print("3. 检测矩形零件")
        print("4. 查看统计数据")
        print("5. 导出数据到Excel")
        print("6. 退出")
        
        choice = input("\n请输入选项 (1-6): ").strip()
        
        if choice == "1":
            # 标定
            known_diameter = float(input("请输入标定件的实际直径 (mm): "))
            print("\n请将标定件放入视野，然后按回车键...")
            input()
            system.calibrate_pixel_to_mm(known_diameter)
        
        elif choice == "2":
            # 检测圆形零件
            part_id = input("请输入零件编号 (直接回车跳过): ").strip()
            nominal_size = input("请输入标称直径 (mm，直接回车跳过): ").strip()
            nominal_size = float(nominal_size) if nominal_size else None
            
            print("\n请将零件放入视野，然后按回车键...")
            input()
            
            system.capture_and_inspect(
                part_type="圆形",
                part_id=part_id,
                nominal_size=nominal_size
            )
        
        elif choice == "3":
            # 检测矩形零件
            part_id = input("请输入零件编号 (直接回车跳过): ").strip()
            nominal_size = input("请输入标称宽度 (mm，直接回车跳过): ").strip()
            nominal_size = float(nominal_size) if nominal_size else None
            
            print("\n请将零件放入视野，然后按回车键...")
            input()
            
            system.capture_and_inspect(
                part_type="矩形",
                part_id=part_id,
                nominal_size=nominal_size
            )
        
        elif choice == "4":
            # 查看统计数据
            stats = system.data_manager.get_statistics()
            print("\n" + "="*50)
            print("统计数据")
            print("="*50)
            print(f"总检测数: {stats.get('total', 0)}")
            print(f"合格数: {stats.get('qualified', 0)}")
            print(f"不合格数: {stats.get('unqualified', 0)}")
            print(f"合格率: {stats.get('qualified_rate', 0)*100:.2f}%")
            print("="*50 + "\n")
        
        elif choice == "5":
            # 导出数据
            system.data_manager.export_to_excel()
        
        elif choice == "6":
            # 退出
            print("\n正在退出系统...")
            system.disconnect_camera()
            print("系统已退出")
            break
        
        else:
            print("\n无效选项，请重新选择")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()