# -*- coding: utf-8 -*-
"""
相机管理模块
Camera Management Module
"""

import cv2
import numpy as np
from typing import Optional, Dict, List
from threading import Thread, Lock
from queue import Queue
import time

from exceptions import CameraException, CameraConnectionError, CameraCaptureError, CameraNotFoundError
from logger_config import get_logger
from config_manager import get_config


class CameraDriver:
    """相机驱动基类"""
    
    def __init__(self, device_id: str):
        self.device_id = device_id
        self.is_connected = False
        self.logger = get_logger(self.__class__.__name__)
    
    def connect(self) -> bool:
        """连接相机"""
        raise NotImplementedError
    
    def disconnect(self) -> bool:
        """断开连接"""
        raise NotImplementedError
    
    def capture(self) -> Optional[np.ndarray]:
        """采集图像"""
        raise NotImplementedError
    
    def set_exposure(self, exposure: float) -> bool:
        """设置曝光"""
        return True
    
    def set_gain(self, gain: float) -> bool:
        """设置增益"""
        return True
    
    def get_resolution(self) -> tuple:
        """获取分辨率"""
        return (0, 0)


class USBCamera(CameraDriver):
    """USB相机驱动"""
    
    def __init__(self, device_id: str = "0"):
        super().__init__(device_id)
        self.camera: Optional[cv2.VideoCapture] = None
        self.capture_thread: Optional[Thread] = None
        self.frame_queue: Queue = Queue(maxsize=10)
        self.is_capturing = False
        self.lock = Lock()
    
    def connect(self) -> bool:
        """连接USB相机"""
        try:
            with self.lock:
                if self.is_connected:
                    self.logger.warning(f"相机已连接: {self.device_id}")
                    return True
                
                camera_idx = int(self.device_id)
                self.camera = cv2.VideoCapture(camera_idx)
                
                if not self.camera.isOpened():
                    raise CameraConnectionError(self.device_id)
                
                # 设置分辨率
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
                
                # 设置帧率
                self.camera.set(cv2.CAP_PROP_FPS, 30)
                
                self.is_connected = True
                self.logger.info(f"USB相机连接成功: {self.device_id}")
                
                return True
        
        except ValueError:
            raise CameraConnectionError(self.device_id, details={'error': '设备ID必须是整数'})
        except Exception as e:
            self.is_connected = False
            raise CameraConnectionError(self.device_id, details={'error': str(e)})
    
    def disconnect(self) -> bool:
        """断开连接"""
        try:
            with self.lock:
                self.stop_capture()
                
                if self.camera is not None:
                    self.camera.release()
                    self.camera = None
                
                self.is_connected = False
                self.logger.info(f"USB相机已断开: {self.device_id}")
                
                return True
        
        except Exception as e:
            self.logger.error(f"断开相机失败: {e}")
            return False
    
    def capture(self) -> Optional[np.ndarray]:
        """采集单帧图像"""
        try:
            if not self.is_connected or self.camera is None:
                raise CameraConnectionError(self.device_id)
            
            ret, frame = self.camera.read()
            
            if not ret or frame is None:
                raise CameraCaptureError()
            
            return frame
        
        except CameraException:
            raise
        except Exception as e:
            self.logger.error(f"采集图像失败: {e}")
            raise CameraCaptureError(details={'error': str(e)})
    
    def start_capture(self) -> bool:
        """开始连续采集（用于预览）"""
        try:
            if self.is_capturing:
                return True
            
            if not self.is_connected:
                raise CameraConnectionError(self.device_id)
            
            self.is_capturing = True
            
            # 清空队列
            while not self.frame_queue.empty():
                self.frame_queue.get()
            
            # 启动采集线程
            self.capture_thread = Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            self.logger.info("开始连续采集")
            
            return True
        
        except Exception as e:
            self.logger.error(f"开始采集失败: {e}")
            return False
    
    def stop_capture(self) -> bool:
        """停止连续采集"""
        try:
            self.is_capturing = False
            
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=1.0)
            
            self.logger.info("停止连续采集")
            
            return True
        
        except Exception as e:
            self.logger.error(f"停止采集失败: {e}")
            return False
    
    def _capture_loop(self):
        """采集循环"""
        while self.is_capturing:
            try:
                frame = self.capture()
                
                if frame is not None:
                    # 非阻塞放入队列
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)
                
                time.sleep(0.01)  # 限制帧率
            
            except CameraCaptureError:
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"采集循环异常: {e}")
                time.sleep(0.1)
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """获取最新帧"""
        try:
            if self.frame_queue.empty():
                return None
            
            # 获取最新帧，丢弃旧帧
            frame = None
            while not self.frame_queue.empty():
                frame = self.frame_queue.get()
            
            return frame
        
        except Exception as e:
            self.logger.error(f"获取最新帧失败: {e}")
            return None
    
    def set_exposure(self, exposure: float) -> bool:
        """设置曝光时间（微秒）"""
        try:
            if self.camera:
                # USB相机可能不支持精确曝光控制
                # 这里使用亮度模拟
                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, exposure / 10000.0)
                return True
            return False
        except Exception as e:
            self.logger.error(f"设置曝光失败: {e}")
            return False
    
    def get_resolution(self) -> tuple:
        """获取分辨率"""
        if self.camera:
            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (0, 0)


class CameraManager:
    """相机管理器（单例）"""
    
    _instance = None
    _cameras: Dict[str, CameraDriver] = {}
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    @classmethod
    def list_cameras(cls) -> List[Dict[str, str]]:
        """
        列出可用相机
        
        Returns:
            相机列表
        """
        cameras = []
        
        # 扫描USB相机
        for i in range(10):  # 最多检查10个设备
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cameras.append({
                    'camera_type': 'usb',
                    'device_id': str(i),
                    'name': f'USB Camera {i}',
                    'resolution': f'{width}x{height}'
                })
                cap.release()
        
        return cameras
    
    @classmethod
    def connect(cls, camera_type: str, device_id: str) -> bool:
        """
        连接相机
        
        Args:
            camera_type: 相机类型 (usb, daheng)
            device_id: 设备ID
        
        Returns:
            是否连接成功
        """
        with cls._lock:
            camera_key = f"{camera_type}:{device_id}"
            
            if camera_key in cls._cameras:
                return True  # 已连接
            
            try:
                if camera_type == 'usb':
                    camera = USBCamera(device_id)
                elif camera_type == 'daheng':
                    # 大恒相机需要额外SDK，这里暂时返回False
                    raise CameraException("大恒相机SDK未集成")
                else:
                    raise CameraException(f"不支持的相机类型: {camera_type}")
                
                camera.connect()
                cls._cameras[camera_key] = camera
                
                return True
            
            except CameraException as e:
                raise e
            except Exception as e:
                raise CameraException(f"连接相机失败: {e}")
    
    @classmethod
    def disconnect(cls, camera_type: str, device_id: str) -> bool:
        """
        断开相机
        
        Args:
            camera_type: 相机类型
            device_id: 设备ID
        
        Returns:
            是否断开成功
        """
        with cls._lock:
            camera_key = f"{camera_type}:{device_id}"
            
            if camera_key not in cls._cameras:
                return True  # 未连接
            
            try:
                camera = cls._cameras[camera_key]
                camera.disconnect()
                del cls._cameras[camera_key]
                
                return True
            
            except Exception as e:
                cls._cameras[camera_key].logger.error(f"断开相机失败: {e}")
                return False
    
    @classmethod
    def get_camera(cls, camera_type: str, device_id: str) -> Optional[CameraDriver]:
        """
        获取相机实例
        
        Args:
            camera_type: 相机类型
            device_id: 设备ID
        
        Returns:
            相机实例
        """
        camera_key = f"{camera_type}:{device_id}"
        return cls._cameras.get(camera_key)
    
    @classmethod
    def capture_image(cls, camera_type: str, device_id: str) -> Optional[np.ndarray]:
        """
        采集图像
        
        Args:
            camera_type: 相机类型
            device_id: 设备ID
        
        Returns:
            图像数据
        """
        camera = cls.get_camera(camera_type, device_id)
        
        if camera is None:
            raise CameraNotFoundError(device_id)
        
        return camera.capture()
    
    @classmethod
    def start_preview(cls, camera_type: str, device_id: str) -> bool:
        """
        开始预览
        
        Args:
            camera_type: 相机类型
            device_id: 设备ID
        
        Returns:
            是否成功
        """
        camera = cls.get_camera(camera_type, device_id)
        
        if camera is None:
            raise CameraNotFoundError(device_id)
        
        if isinstance(camera, USBCamera):
            return camera.start_capture()
        
        return False
    
    @classmethod
    def stop_preview(cls, camera_type: str, device_id: str) -> bool:
        """
        停止预览
        
        Args:
            camera_type: 相机类型
            device_id: 设备ID
        
        Returns:
            是否成功
        """
        camera = cls.get_camera(camera_type, device_id)
        
        if camera is None:
            return True
        
        if isinstance(camera, USBCamera):
            return camera.stop_capture()
        
        return False
    
    @classmethod
    def get_preview_frame(cls, camera_type: str, device_id: str) -> Optional[np.ndarray]:
        """
        获取预览帧
        
        Args:
            camera_type: 相机类型
            device_id: 设备ID
        
        Returns:
            图像帧
        """
        camera = cls.get_camera(camera_type, device_id)
        
        if camera is None:
            return None
        
        if isinstance(camera, USBCamera):
            return camera.get_latest_frame()
        
        return None
    
    @classmethod
    def disconnect_all(cls) -> bool:
        """断开所有相机"""
        with cls._lock:
            for camera_key in list(cls._cameras.keys()):
                try:
                    camera = cls._cameras[camera_key]
                    camera.disconnect()
                except Exception as e:
                    pass
            
            cls._cameras.clear()
            return True


# 便捷函数
def get_camera_manager() -> CameraManager:
    """获取相机管理器实例"""
    return CameraManager()


def list_available_cameras() -> List[Dict[str, str]]:
    """列出可用相机"""
    return CameraManager.list_cameras()


def connect_camera(camera_type: str, device_id: str) -> bool:
    """连接相机"""
    return CameraManager.connect(camera_type, device_id)


def disconnect_camera(camera_type: str, device_id: str) -> bool:
    """断开相机"""
    return CameraManager.disconnect(camera_type, device_id)


def capture_from_camera(camera_type: str, device_id: str) -> Optional[np.ndarray]:
    """从相机采集图像"""
    return CameraManager.capture_image(camera_type, device_id)