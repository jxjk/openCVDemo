# -*- coding: utf-8 -*-
"""
批量检测模块
Batch Inspection Module

实现连续自动检测功能，支持多线程处理
目标：≥60件/分钟检测速度

创建日期: 2026-02-10
"""

import os
import time
import threading
import queue
from datetime import datetime
from typing import Optional, List, Dict, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from exceptions import (
    CameraException,
    ImageProcessingException,
    DetectionException
)
from logger_config import get_logger
from config_manager import get_config


class InspectionStatus(Enum):
    """检测状态"""
    IDLE = "idle"  # 空闲
    RUNNING = "running"  # 运行中
    PAUSED = "paused"  # 暂停
    STOPPING = "stopping"  # 停止中
    STOPPED = "stopped"  # 已停止
    ERROR = "error"  # 错误


@dataclass
class BatchInspectionConfig:
    """批量检测配置"""
    # 性能参数
    max_workers: int = 4  # 最大工作线程数
    target_speed: int = 60  # 目标检测速度（件/分钟）
    image_queue_size: int = 10  # 图像队列大小
    result_queue_size: int = 20  # 结果队列大小
    
    # 图像采集参数
    capture_interval: float = 0.0  # 采集间隔（秒），0为最快速度
    warmup_frames: int = 3  # 预热帧数
    
    # 检测参数
    enable_subpixel: bool = True  # 启用亚像素检测
    enable_quality_check: bool = True  # 启用质量检查
    
    # 数据保存
    auto_save: bool = True  # 自动保存结果
    save_images: bool = True  # 保存检测图像
    save_failures_only: bool = False  # 仅保存不合格图像


@dataclass
class InspectionTask:
    """检测任务"""
    task_id: str
    image: np.ndarray
    part_id: str = ""
    part_type: str = "圆形"
    nominal_size: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class BatchInspectionResult:
    """批量检测结果"""
    task_id: str
    part_id: str
    result: Optional[Any]  # 检测结果对象
    is_passed: bool
    error: Optional[str] = None
    processing_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))


class BatchInspectionEngine:
    """
    批量检测引擎
    
    使用生产者-消费者模式实现多线程批量检测：
    - 生产者线程：图像采集
    - 工作线程池：图像处理和检测
    - 结果收集线程：结果汇总和保存
    """
    
    def __init__(self, 
                 detection_engine,
                 config: Optional[BatchInspectionConfig] = None):
        """
        初始化批量检测引擎
        
        Args:
            detection_engine: 检测引擎实例（InspectionEngine）
            config: 批量检测配置
        """
        self.detection_engine = detection_engine
        self.config = config or BatchInspectionConfig()
        self.logger = get_logger(self.__class__.__name__)
        
        # 状态管理
        self._status = InspectionStatus.IDLE
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        
        # 队列
        self._task_queue: queue.Queue = queue.Queue(maxsize=self.config.image_queue_size)
        self._result_queue: queue.Queue = queue.Queue(maxsize=self.config.result_queue_size)
        
        # 统计信息
        self._stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'passed_tasks': 0,
            'failed_tasks': 0,
            'error_tasks': 0,
            'total_time': 0.0,
            'start_time': None,
            'end_time': None
        }
        
        # 线程池
        self._executor: Optional[ThreadPoolExecutor] = None
        
        # 回调函数
        self._on_result_callback: Optional[Callable] = None
        self._on_error_callback: Optional[Callable] = None
        self._on_progress_callback: Optional[Callable] = None
        
        # 数据存储
        self._results: List[BatchInspectionResult] = []
        
        self.logger.info(f"批量检测引擎初始化完成，目标速度: {self.config.target_speed}件/分钟")
    
    # =========================================================================
    # 状态管理
    # =========================================================================
    
    @property
    def status(self) -> InspectionStatus:
        """获取当前状态"""
        return self._status
    
    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._status == InspectionStatus.RUNNING
    
    @property
    def is_paused(self) -> bool:
        """是否已暂停"""
        return self._status == InspectionStatus.PAUSED
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        stats = self._stats.copy()
        
        # 计算速度
        if stats['total_time'] > 0:
            stats['current_speed'] = stats['completed_tasks'] / stats['total_time'] * 60  # 件/分钟
            stats['avg_time_per_part'] = stats['total_time'] / stats['completed_tasks'] if stats['completed_tasks'] > 0 else 0
        else:
            stats['current_speed'] = 0.0
            stats['avg_time_per_part'] = 0.0
        
        # 计算合格率
        if stats['completed_tasks'] > 0:
            stats['pass_rate'] = stats['passed_tasks'] / stats['completed_tasks'] * 100
        else:
            stats['pass_rate'] = 0.0
        
        return stats
    
    # =========================================================================
    # 回调函数设置
    # =========================================================================
    
    def set_result_callback(self, callback: Callable):
        """设置结果回调函数"""
        self._on_result_callback = callback
        self.logger.debug("结果回调函数已设置")
    
    def set_error_callback(self, callback: Callable):
        """设置错误回调函数"""
        self._on_error_callback = callback
        self.logger.debug("错误回调函数已设置")
    
    def set_progress_callback(self, callback: Callable):
        """设置进度回调函数"""
        self._on_progress_callback = callback
        self.logger.debug("进度回调函数已设置")
    
    # =========================================================================
    # 启动和控制
    # =========================================================================
    
    def start(self):
        """启动批量检测"""
        if self._status in [InspectionStatus.RUNNING, InspectionStatus.PAUSED]:
            self.logger.warning("批量检测已经在运行中")
            return False
        
        try:
            self._status = InspectionStatus.RUNNING
            self._stop_event.clear()
            self._pause_event.clear()
            self._stats['start_time'] = time.time()
            self._stats['end_time'] = None
            
            # 重置统计信息
            self._stats.update({
                'total_tasks': 0,
                'completed_tasks': 0,
                'passed_tasks': 0,
                'failed_tasks': 0,
                'error_tasks': 0,
                'total_time': 0.0
            })
            
            # 创建线程池
            self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
            
            # 启动结果收集线程
            self._result_collector_thread = threading.Thread(
                target=self._result_collector_worker,
                name="ResultCollector",
                daemon=True
            )
            self._result_collector_thread.start()
            
            self.logger.info("批量检测已启动")
            return True
        
        except Exception as e:
            self.logger.error(f"启动批量检测失败: {e}")
            self._status = InspectionStatus.ERROR
            return False
    
    def pause(self):
        """暂停批量检测"""
        if self._status != InspectionStatus.RUNNING:
            self.logger.warning("批量检测未在运行，无法暂停")
            return False
        
        self._status = InspectionStatus.PAUSED
        self._pause_event.set()
        self.logger.info("批量检测已暂停")
        return True
    
    def resume(self):
        """恢复批量检测"""
        if self._status != InspectionStatus.PAUSED:
            self.logger.warning("批量检测未暂停，无法恢复")
            return False
        
        self._status = InspectionStatus.RUNNING
        self._pause_event.clear()
        self.logger.info("批量检测已恢复")
        return True
    
    def stop(self, wait: bool = True):
        """
        停止批量检测
        
        Args:
            wait: 是否等待所有任务完成
        """
        if self._status == InspectionStatus.IDLE:
            return
        
        self._status = InspectionStatus.STOPPING
        self._stop_event.set()
        self._pause_event.clear()  # 清除暂停事件，允许线程继续
        
        self.logger.info("批量检测正在停止...")
        
        # 关闭线程池
        if self._executor:
            if wait:
                self._executor.shutdown(wait=True)
            else:
                self._executor.shutdown(wait=False)
            self._executor = None
        
        # 清空队列
        while not self._task_queue.empty():
            try:
                self._task_queue.get_nowait()
            except queue.Empty:
                break
        
        self._stats['end_time'] = time.time()
        if self._stats['start_time']:
            self._stats['total_time'] = self._stats['end_time'] - self._stats['start_time']
        
        self._status = InspectionStatus.STOPPED
        self.logger.info(f"批量检测已停止，统计信息: {self.get_statistics()}")
    
    # =========================================================================
    # 任务处理
    # =========================================================================
    
    def add_task(self, task: InspectionTask) -> bool:
        """
        添加检测任务
        
        Args:
            task: 检测任务
        
        Returns:
            是否成功添加
        """
        if self._status != InspectionStatus.RUNNING:
            self.logger.warning("批量检测未运行，无法添加任务")
            return False
        
        try:
            self._task_queue.put(task, timeout=1.0)
            self._stats['total_tasks'] += 1
            
            # 提交检测任务到线程池
            future = self._executor.submit(self._process_task, task)
            future.add_done_callback(self._on_task_completed)
            
            self.logger.debug(f"任务已添加: {task.task_id}")
            return True
        
        except queue.Full:
            self.logger.warning("任务队列已满")
            return False
        except Exception as e:
            self.logger.error(f"添加任务失败: {e}")
            return False
    
    def add_image(self, image: np.ndarray, 
                  part_id: str = "",
                  part_type: str = "圆形",
                  nominal_size: Optional[float] = None) -> bool:
        """
        添加图像进行检测（便捷方法）
        
        Args:
            image: 图像
            part_id: 零件编号
            part_type: 零件类型
            nominal_size: 标称尺寸
        
        Returns:
            是否成功添加
        """
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        task = InspectionTask(
            task_id=task_id,
            image=image,
            part_id=part_id,
            part_type=part_type,
            nominal_size=nominal_size
        )
        
        return self.add_task(task)
    
    def _process_task(self, task: InspectionTask) -> BatchInspectionResult:
        """
        处理单个检测任务
        
        Args:
            task: 检测任务
        
        Returns:
            检测结果
        """
        start_time = time.time()
        
        try:
            # 检查暂停事件
            if self._pause_event.is_set():
                self._pause_event.wait()  # 等待恢复
            
            # 检查停止事件
            if self._stop_event.is_set():
                return BatchInspectionResult(
                    task_id=task.task_id,
                    part_id=task.part_id,
                    result=None,
                    is_passed=False,
                    error="检测已停止",
                    processing_time=time.time() - start_time
                )
            
            # 执行检测
            if task.part_type == "圆形":
                result = self.detection_engine.detect_circle(
                    task.image,
                    part_id=task.part_id,
                    part_type=task.part_type,
                    nominal_size=task.nominal_size
                )
            elif task.part_type == "矩形":
                result = self.detection_engine.detect_rectangle(
                    task.image,
                    part_id=task.part_id,
                    part_type=task.part_type,
                    nominal_width=task.nominal_size,
                    nominal_height=task.nominal_size
                )
            else:
                raise DetectionException(f"不支持的零件类型: {task.part_type}")
            
            # 检查结果
            if result is None:
                return BatchInspectionResult(
                    task_id=task.task_id,
                    part_id=task.part_id,
                    result=None,
                    is_passed=False,
                    error="检测失败，未返回结果",
                    processing_time=time.time() - start_time
                )
            
            # 返回结果
            return BatchInspectionResult(
                task_id=task.task_id,
                part_id=task.part_id,
                result=result,
                is_passed=result.is_qualified,
                processing_time=time.time() - start_time
            )
        
        except CameraException as e:
            error_msg = f"相机错误: {str(e)}"
            self.logger.error(error_msg)
            return BatchInspectionResult(
                task_id=task.task_id,
                part_id=task.part_id,
                result=None,
                is_passed=False,
                error=error_msg,
                processing_time=time.time() - start_time
            )
        
        except ImageProcessingException as e:
            error_msg = f"图像处理错误: {str(e)}"
            self.logger.error(error_msg)
            return BatchInspectionResult(
                task_id=task.task_id,
                part_id=task.part_id,
                result=None,
                is_passed=False,
                error=error_msg,
                processing_time=time.time() - start_time
            )
        
        except DetectionException as e:
            error_msg = f"检测错误: {str(e)}"
            self.logger.error(error_msg)
            return BatchInspectionResult(
                task_id=task.task_id,
                part_id=task.part_id,
                result=None,
                is_passed=False,
                error=error_msg,
                processing_time=time.time() - start_time
            )
        
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            self.logger.exception(error_msg)
            return BatchInspectionResult(
                task_id=task.task_id,
                part_id=task.part_id,
                result=None,
                is_passed=False,
                error=error_msg,
                processing_time=time.time() - start_time
            )
    
    def _on_task_completed(self, future):
        """
        任务完成回调
        
        Args:
            future: Future对象
        """
        try:
            result = future.result()
            
            # 将结果放入结果队列
            try:
                self._result_queue.put(result, timeout=1.0)
            except queue.Full:
                self.logger.warning("结果队列已满，丢弃结果")
            
        except Exception as e:
            self.logger.error(f"任务完成回调失败: {e}")
    
    def _result_collector_worker(self):
        """结果收集工作线程"""
        while not self._stop_event.is_set():
            try:
                # 从队列获取结果（带超时）
                result = self._result_queue.get(timeout=0.5)
                
                # 更新统计信息
                self._stats['completed_tasks'] += 1
                
                if result.error:
                    self._stats['error_tasks'] += 1
                elif result.is_passed:
                    self._stats['passed_tasks'] += 1
                else:
                    self._stats['failed_tasks'] += 1
                
                # 保存结果
                self._results.append(result)
                
                # 调用结果回调
                if self._on_result_callback:
                    try:
                        self._on_result_callback(result)
                    except Exception as e:
                        self.logger.error(f"结果回调失败: {e}")
                
                # 调用错误回调
                if result.error and self._on_error_callback:
                    try:
                        self._on_error_callback(result)
                    except Exception as e:
                        self.logger.error(f"错误回调失败: {e}")
                
                # 调用进度回调
                if self._on_progress_callback:
                    try:
                        stats = self.get_statistics()
                        self._on_progress_callback(stats)
                    except Exception as e:
                        self.logger.error(f"进度回调失败: {e}")
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"结果收集失败: {e}")
    
    # =========================================================================
    # 结果获取
    # =========================================================================
    
    def get_results(self) -> List[BatchInspectionResult]:
        """获取所有检测结果"""
        return self._results.copy()
    
    def get_recent_results(self, count: int = 10) -> List[BatchInspectionResult]:
        """
       获取最近的检测结果
        
        Args:
            count: 数量
        
        Returns:
            结果列表
        """
        return self._results[-count:] if self._results else []
    
    def clear_results(self):
        """清空结果"""
        self._results.clear()
        self.logger.debug("检测结果已清空")


# ============================================================================
# 相机批量采集器
# ============================================================================

class CameraBatchAcquisition:
    """相机批量采集器"""
    
    def __init__(self, 
                 camera_driver,
                 batch_engine: BatchInspectionEngine,
                 config: Optional[BatchInspectionConfig] = None):
        """
        初始化相机批量采集器
        
        Args:
            camera_driver: 相机驱动实例
            batch_engine: 批量检测引擎
            config: 配置
        """
        self.camera = camera_driver
        self.batch_engine = batch_engine
        self.config = config or BatchInspectionConfig()
        self.logger = get_logger(self.__class__.__name__)
        
        self._acquisition_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._part_counter = 0
    
    def start_acquisition(self, 
                         part_type: str = "圆形",
                         nominal_size: Optional[float] = None):
        """
        启动连续采集
        
        Args:
            part_type: 零件类型
            nominal_size: 标称尺寸
        """
        if self._acquisition_thread and self._acquisition_thread.is_alive():
            self.logger.warning("采集已在运行中")
            return
        
        self._stop_event.clear()
        self._part_counter = 0
        
        self._acquisition_thread = threading.Thread(
            target=self._acquisition_worker,
            args=(part_type, nominal_size),
            name="CameraAcquisition",
            daemon=True
        )
        self._acquisition_thread.start()
        
        self.logger.info("相机批量采集已启动")
    
    def stop_acquisition(self):
        """停止采集"""
        if self._stop_event.is_set():
            return
        
        self._stop_event.set()
        
        if self._acquisition_thread:
            self._acquisition_thread.join(timeout=5.0)
        
        self.logger.info("相机批量采集已停止")
    
    def _acquisition_worker(self, part_type: str, nominal_size: Optional[float]):
        """采集工作线程"""
        warmup_count = 0
        
        while not self._stop_event.is_set():
            try:
                # 检查批量引擎状态
                if not self.batch_engine.is_running:
                    time.sleep(0.1)
                    continue
                
                # 检查暂停
                if self.batch_engine.is_paused:
                    time.sleep(0.1)
                    continue
                
                # 采集图像
                image = self.camera.capture_image()
                
                if image is None:
                    self.logger.warning("图像采集失败，跳过")
                    time.sleep(0.1)
                    continue
                
                # 预热
                if warmup_count < self.config.warmup_frames:
                    warmup_count += 1
                    self.logger.debug(f"预热帧: {warmup_count}/{self.config.warmup_frames}")
                    continue
                
                # 生成零件ID
                self._part_counter += 1
                part_id = f"PART_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._part_counter:04d}"
                
                # 添加到批量检测引擎
                success = self.batch_engine.add_image(
                    image=image,
                    part_id=part_id,
                    part_type=part_type,
                    nominal_size=nominal_size
                )
                
                if not success:
                    self.logger.warning("添加检测任务失败")
                
                # 控制采集速度
                if self.config.capture_interval > 0:
                    time.sleep(self.config.capture_interval)
            
            except Exception as e:
                self.logger.error(f"采集失败: {e}")
                time.sleep(0.5)


# ============================================================================
# 便捷函数
# ============================================================================

def create_batch_engine(detection_engine, 
                       max_workers: int = 4,
                       target_speed: int = 60) -> BatchInspectionEngine:
    """
    创建批量检测引擎（便捷函数）
    
    Args:
        detection_engine: 检测引擎
        max_workers: 最大工作线程数
        target_speed: 目标速度（件/分钟）
    
    Returns:
        批量检测引擎实例
    """
    config = BatchInspectionConfig(
        max_workers=max_workers,
        target_speed=target_speed
    )
    
    return BatchInspectionEngine(detection_engine, config)