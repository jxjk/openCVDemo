# -*- coding: utf-8 -*-
"""
标定管理模块
Calibration Management Module
"""

import cv2
import numpy as np
import json
import os
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from datetime import datetime

from exceptions import CalibrationException, CalibrationFailedError, CalibrationBoardNotFoundError, PixelToMmCalibrationError
from logger_config import get_logger
from config_manager import get_config


class ChessboardCalibration:
    """棋盘格标定"""
    
    def __init__(self, pattern_size: Tuple[int, int] = (9, 6), 
                 square_size: float = 1.0):
        """
        初始化棋盘格标定
        
        Args:
            pattern_size: 棋盘格角点数量 (width, height)
            square_size: 棋盘格方块大小（mm）
        """
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.logger = get_logger(self.__class__.__name__)
        
        # 准备标定板的角点坐标
        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # 标定结果
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.reprojection_error = None
    
    def calibrate(self, images: List[np.ndarray]) -> Dict[str, any]:
        """
        执行相机标定
        
        Args:
            images: 标定图像列表
        
        Returns:
            标定结果字典
        """
        try:
            if not images:
                raise CalibrationException("没有提供标定图像")
            
            objpoints = []  # 3D世界坐标
            imgpoints = []  # 2D图像坐标
            
            successful_images = 0
            
            for img in images:
                # 转换为灰度图
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img
                
                # 查找棋盘格角点
                ret, corners = cv2.findChessboardCorners(
                    gray, 
                    self.pattern_size, 
                    None
                )
                
                if ret:
                    # 亚像素细化
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners_refined = cv2.cornerSubPix(
                        gray, 
                        corners, 
                        (11, 11), 
                        (-1, -1), 
                        criteria
                    )
                    
                    objpoints.append(self.objp)
                    imgpoints.append(corners_refined)
                    successful_images += 1
            
            if successful_images == 0:
                raise CalibrationBoardNotFoundError()
            
            self.logger.info(f"成功识别 {successful_images}/{len(images)} 张标定图像")
            
            # 执行标定
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, 
                imgpoints, 
                gray.shape[::-1], 
                None, 
                None
            )
            
            if not ret:
                raise CalibrationFailedError()
            
            # 计算重投影误差
            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints_reproj, _ = cv2.projectPoints(
                    objpoints[i], 
                    rvecs[i], 
                    tvecs[i], 
                    camera_matrix, 
                    dist_coeffs
                )
                error = cv2.norm(imgpoints[i], imgpoints_reproj, cv2.NORM_L2) / len(imgpoints_reproj)
                mean_error += error
            
            self.reprojection_error = mean_error / len(objpoints)
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            self.rvecs = rvecs
            self.tvecs = tvecs
            
            result = {
                'camera_matrix': camera_matrix.tolist(),
                'dist_coeffs': dist_coeffs.tolist(),
                'reprojection_error': float(self.reprojection_error),
                'successful_images': successful_images,
                'total_images': len(images),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"标定完成，重投影误差: {self.reprojection_error:.4f}")
            
            return result
        
        except CalibrationException:
            raise
        except Exception as e:
            self.logger.error(f"标定失败: {e}")
            raise CalibrationFailedError(details={'error': str(e)})
    
    def undistort(self, image: np.ndarray) -> np.ndarray:
        """
        畸变校正
        
        Args:
            image: 输入图像
        
        Returns:
            校正后的图像
        """
        try:
            if self.camera_matrix is None:
                raise CalibrationException("相机尚未标定")
            
            undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
            return undistorted
        
        except CalibrationException:
            raise
        except Exception as e:
            self.logger.error(f"畸变校正失败: {e}")
            raise CalibrationException("畸变校正失败")
    
    def save_calibration(self, filepath: str) -> bool:
        """
        保存标定结果
        
        Args:
            filepath: 文件路径
        
        Returns:
            是否保存成功
        """
        try:
            if self.camera_matrix is None:
                raise CalibrationException("没有可保存的标定结果")
            
            calibration_data = {
                'pattern_size': self.pattern_size,
                'square_size': self.square_size,
                'camera_matrix': self.camera_matrix.tolist(),
                'dist_coeffs': self.dist_coeffs.tolist(),
                'reprojection_error': float(self.reprojection_error),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            self.logger.info(f"标定结果已保存到: {filepath}")
            
            return True
        
        except CalibrationException:
            raise
        except Exception as e:
            self.logger.error(f"保存标定结果失败: {e}")
            return False
    
    def load_calibration(self, filepath: str) -> bool:
        """
        加载标定结果
        
        Args:
            filepath: 文件路径
        
        Returns:
            是否加载成功
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.pattern_size = tuple(data['pattern_size'])
            self.square_size = data['square_size']
            self.camera_matrix = np.array(data['camera_matrix'])
            self.dist_coeffs = np.array(data['dist_coeffs'])
            self.reprojection_error = data.get('reprojection_error', 0.0)
            
            self.logger.info(f"标定结果已从 {filepath} 加载")
            
            return True
        
        except Exception as e:
            self.logger.error(f"加载标定结果失败: {e}")
            return False


class PixelToMmCalibration:
    """像素到毫米标定"""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.pixel_to_mm = 0.0  # 像素/毫米
        self.calibrated = False
    
    def calibrate_by_reference(self, image: np.ndarray, 
                              reference_length_mm: float,
                              reference_points: List[Tuple[int, int]]) -> float:
        """
        通过参考长度标定
        
        Args:
            image: 标定图像
            reference_length_mm: 参考长度（毫米）
            reference_points: 参考线段的两个端点 [(x1, y1), (x2, y2)]
        
        Returns:
            像素到毫米的转换比例
        """
        try:
            if len(reference_points) != 2:
                raise PixelToMmCalibrationError("需要两个参考点")
            
            p1, p2 = reference_points
            
            # 计算像素距离
            pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            if pixel_distance == 0:
                raise PixelToMmCalibrationError("参考点距离为零")
            
            # 计算转换比例
            self.pixel_to_mm = reference_length_mm / pixel_distance
            self.calibrated = True
            
            self.logger.info(f"标定完成: {pixel_distance:.2f}像素 = {reference_length_mm:.2f}mm, "
                           f"转换比例 = {self.pixel_to_mm:.6f} mm/像素")
            
            return self.pixel_to_mm
        
        except PixelToMmCalibrationError:
            raise
        except Exception as e:
            self.logger.error(f"标定失败: {e}")
            raise PixelToMmCalibrationError(details={'error': str(e)})
    
    def calibrate_by_circle(self, image: np.ndarray, 
                           circle_center: Tuple[int, int],
                           circle_radius_pixels: float,
                           circle_diameter_mm: float) -> float:
        """
        通过圆形标定
        
        Args:
            image: 标定图像
            circle_center: 圆心坐标
            circle_radius_pixels: 圆半径（像素）
            circle_diameter_mm: 圆直径（毫米）
        
        Returns:
            像素到毫米的转换比例
        """
        try:
            if circle_radius_pixels == 0:
                raise PixelToMmCalibrationError("圆半径为零")
            
            # 计算转换比例
            pixel_diameter = circle_radius_pixels * 2
            self.pixel_to_mm = circle_diameter_mm / pixel_diameter
            self.calibrated = True
            
            self.logger.info(f"圆形标定完成: 直径 {pixel_diameter:.2f}像素 = {circle_diameter_mm:.2f}mm, "
                           f"转换比例 = {self.pixel_to_mm:.6f} mm/像素")
            
            return self.pixel_to_mm
        
        except PixelToMmCalibrationError:
            raise
        except Exception as e:
            self.logger.error(f"圆形标定失败: {e}")
            raise PixelToMmCalibrationError(details={'error': str(e)})
    
    def pixels_to_mm(self, pixels: float) -> float:
        """
        像素转换为毫米
        
        Args:
            pixels: 像素值
        
        Returns:
            毫米值
        """
        if not self.calibrated:
            raise PixelToMmCalibrationError("尚未标定")
        
        return pixels * self.pixel_to_mm
    
    def mm_to_pixels(self, mm: float) -> float:
        """
        毫米转换为像素
        
        Args:
            mm: 毫米值
        
        Returns:
            像素值
        """
        if not self.calibrated:
            raise PixelToMmCalibrationError("尚未标定")
        
        if self.pixel_to_mm == 0:
            raise PixelToMmCalibrationError("转换比例为零")
        
        return mm / self.pixel_to_mm
    
    def save_calibration(self, filepath: str) -> bool:
        """
        保存标定结果
        
        Args:
            filepath: 文件路径
        
        Returns:
            是否保存成功
        """
        try:
            calibration_data = {
                'pixel_to_mm': self.pixel_to_mm,
                'calibrated': self.calibrated,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            self.logger.info(f"像素标定结果已保存到: {filepath}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"保存标定结果失败: {e}")
            return False
    
    def load_calibration(self, filepath: str) -> bool:
        """
        加载标定结果
        
        Args:
            filepath: 文件路径
        
        Returns:
            是否加载成功
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.pixel_to_mm = data['pixel_to_mm']
            self.calibrated = data.get('calibrated', True)
            
            self.logger.info(f"像素标定结果已从 {filepath} 加载")
            
            return True
        
        except Exception as e:
            self.logger.error(f"加载标定结果失败: {e}")
            return False


class CalibrationManager:
    """标定管理器（单例）"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.config = get_config()
        self.chessboard_calib = None
        self.pixel_to_mm_calib = PixelToMmCalibration()
    
    def get_pixel_to_mm(self) -> float:
        """
        获取像素到毫米的转换比例
        
        Returns:
            转换比例
        """
        # 如果有本地标定结果，使用本地结果
        if self.pixel_to_mm_calib.calibrated:
            return self.pixel_to_mm_calib.pixel_to_mm
        
        # 否则使用配置中的默认值
        return self.config.pixel_to_mm
    
    def calibrate_by_reference(self, image: np.ndarray,
                              reference_length_mm: float,
                              reference_points: List[Tuple[int, int]]) -> float:
        """通过参考长度标定"""
        return self.pixel_to_mm_calib.calibrate_by_reference(
            image, reference_length_mm, reference_points
        )
    
    def calibrate_by_circle(self, image: np.ndarray,
                           circle_center: Tuple[int, int],
                           circle_radius_pixels: float,
                           circle_diameter_mm: float) -> float:
        """通过圆形标定"""
        return self.pixel_to_mm_calib.calibrate_by_circle(
            image, circle_center, circle_radius_pixels, circle_diameter_mm
        )
    
    def save_calibration(self, filepath: str) -> bool:
        """保存标定结果"""
        return self.pixel_to_mm_calib.save_calibration(filepath)
    
    def load_calibration(self, filepath: str) -> bool:
        """加载标定结果"""
        return self.pixel_to_mm_calib.load_calibration(filepath)


# 便捷函数
def get_calibration_manager() -> CalibrationManager:
    """获取标定管理器实例"""
    return CalibrationManager()


def pixels_to_mm(pixels: float) -> float:
    """像素转换为毫米（便捷函数）"""
    manager = get_calibration_manager()
    return manager.pixels_to_mm_calib.pixels_to_mm(pixels)


def mm_to_pixels(mm: float) -> float:
    """毫米转换为像素（便捷函数）"""
    manager = get_calibration_manager()
    return manager.pixels_to_mm_calib.mm_to_pixels(mm)