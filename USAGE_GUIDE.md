# 增强功能使用指南

## 概述

本文档介绍微小零件中高精度视觉检测系统新增功能的详细使用方法。

## 新增功能模块

### 1. 批量检测模块 (batch_inspection.py)

#### 功能特性
- **多线程处理**：使用线程池实现并行检测
- **生产者-消费者模式**：图像采集和检测分离
- **性能监控**：实时统计检测速度和合格率
- **任务队列管理**：支持暂停/恢复/停止

#### 基本使用

```python
from inspection_system import InspectionEngine, InspectionConfig
from batch_inspection import BatchInspectionEngine, BatchInspectionConfig

# 初始化
config = InspectionConfig()
inspection_engine = InspectionEngine(config)

# 创建批量检测引擎
batch_config = BatchInspectionConfig(
    max_workers=4,        # 工作线程数
    target_speed=60,      # 目标速度（件/分钟）
    auto_save=True        # 自动保存结果
)
batch_engine = BatchInspectionEngine(inspection_engine, batch_config)

# 设置回调函数
def on_result(result):
    print(f"检测结果: {result.part_id} - {'合格' if result.is_passed else '不合格'}")

def on_progress(stats):
    print(f"进度: {stats['completed_tasks']}/{stats['total_tasks']} - "
          f"速度: {stats['current_speed']:.1f}件/分")

batch_engine.set_result_callback(on_result)
batch_engine.set_progress_callback(on_progress)

# 启动批量检测
batch_engine.start()

# 添加检测任务
import cv2
image = cv2.imread("part.jpg")
batch_engine.add_image(image, part_id="PART001", part_type="圆形", nominal_size=5.0)

# 停止批量检测
batch_engine.stop()

# 获取统计信息
stats = batch_engine.get_statistics()
print(f"总计: {stats['completed_tasks']} | "
      f"合格: {stats['passed_tasks']} | "
      f"合格率: {stats['pass_rate']:.2f}%")
```

#### 性能优化建议
1. **调整工作线程数**：根据CPU核心数设置，通常为CPU核心数的1-2倍
2. **图像队列大小**：增加队列大小可以减少等待时间，但会增加内存占用
3. **启用亚像素检测**：牺牲速度提高精度，可根据需求开关
4. **批量保存**：设置`auto_save=True`自动保存，或定期批量保存

---

### 2. 缺陷检测模块 (defect_detection.py)

#### 功能特性
- **表面缺陷检测**：划痕、污渍、孔洞、斑点
- **边缘缺陷检测**：缺口、崩边、锯齿
- **毛刺检测**：边缘毛刺和突起
- **综合缺陷检测**：整合多种检测方法

#### 基本使用

```python
from defect_detection import (
    SurfaceDefectDetector,
    EdgeDefectDetector,
    BurrDetector,
    ComprehensiveDefectDetector
)
import cv2

# 加载图像
image = cv2.imread("part.jpg", cv2.IMREAD_GRAYSCALE)

# 方法1: 单独检测表面缺陷
surface_detector = SurfaceDefectDetector()
surface_result = surface_detector.detect(image)
print(f"表面缺陷数量: {len(surface_result.defects)}")
print(f"质量评分: {surface_result.quality_score:.2f}")

# 方法2: 单独检测边缘缺陷
edge_detector = EdgeDefectDetector()
edge_result = edge_detector.detect(image)
print(f"边缘缺陷数量: {len(edge_result.defects)}")

# 方法3: 综合检测（推荐）
comprehensive_detector = ComprehensiveDefectDetector()
result = comprehensive_detector.detect_all(image)

# 查看结果
if result.has_defect:
    print(f"发现 {len(result.defects)} 个缺陷:")
    for i, defect in enumerate(result.defects):
        print(f"  {i+1}. {defect.defect_type.value} - "
              f"位置: {defect.location} - "
              f"严重程度: {defect.severity:.2f}")
else:
    print("未发现缺陷，质量良好")

# 绘制缺陷标记
result_image = comprehensive_detector.draw_defects(image, result)
cv2.imwrite("result_with_defects.jpg", result_image)
```

#### 检测参数调整

```python
# 表面缺陷检测器参数
surface_detector = SurfaceDefectDetector()
surface_detector.min_defect_area = 20      # 最小缺陷面积
surface_detector.max_defect_area = 3000   # 最大缺陷面积
surface_detector.defect_threshold = 40    # 检测阈值

# 边缘缺陷检测器参数
edge_detector = EdgeDefectDetector()
edge_detector.min_defect_length = 15     # 最小缺陷长度
edge_detector.max_defect_length = 150    # 最大缺陷长度
edge_detector.defect_depth_threshold = 8 # 缺陷深度阈值

# 毛刺检测器参数
burr_detector = BurrDetector()
burr_detector.burr_threshold = 3.0       # 毛刺阈值
burr_detector.burr_length_min = 3        # 最小毛刺长度
burr_detector.burr_length_max = 25       # 最大毛刺长度
```

---

### 3. 增强版GUI (inspection_gui_enhanced.py)

#### 启动程序

```bash
python inspection_gui_enhanced.py
```

#### 功能菜单

**文件菜单**
- 打开图像：加载本地图像文件
- 历史数据查询：查询历史检测记录
- 统计报表：生成统计图表和报告
- 退出：关闭程序

**相机菜单**
- USB相机：选择USB相机
- 大恒相机：选择大恒工业相机
- 连接相机：连接选定的相机
- 断开相机：断开相机连接

**检测菜单**
- 单次检测：对当前图像执行一次检测
- 缺陷检测：对当前图像执行缺陷检测

**工具菜单**
- 图纸标注：打开图纸标注工具
- 相机标定：打开相机标定工具
- 启用报警：启用/禁用声光报警

#### 批量检测控制面板

**控制按钮**
- 启动：开始批量检测
- 暂停：暂停检测（再次点击恢复）
- 停止：停止检测

**状态显示**
- 状态：空闲/运行中/已暂停/已停止
- 统计：已检测数量、合格数、不合格数、检测速度

**使用流程**
1. 连接相机或加载图像
2. 设置检测参数（零件类型、标称尺寸）
3. 点击"启动"开始批量检测
4. 实时查看检测进度和结果
5. 点击"暂停"或"停止"控制检测流程

#### 历史数据查询

**查询条件**
- 时间范围：设置起始和结束时间
- 零件类型：选择圆形、矩形或全部
- 检测结果：选择合格、不合格或全部

**操作按钮**
- 查询：执行查询
- 导出：导出查询结果（Excel/CSV）
- 清空：清空查询条件

**统计信息**
底部显示查询结果的统计信息：总计、合格数、不合格数、合格率

#### 统计报表

**图表类型**
1. 合格率饼图：显示合格/不合格比例
2. 零件类型分布：不同零件类型的数量
3. 检测数量时间趋势：按小时统计检测数量
4. 测量值分布：测量值的直方图

**操作**
- 刷新报表：重新生成报表
- 导出报表：导出为Excel文件

---

### 4. 声光报警功能

#### 报警类型
- **NG**：检测到不合格品
- **ERROR**：系统错误
- **WARNING**：警告信息

#### 使用方式

```python
from inspection_gui_enhanced import AlarmController

# 创建报警控制器
alarm = AlarmController()

# 启用/禁用报警
alarm.enable_alarm(True)
alarm.enable_alarm(False)

# 触发报警
alarm.trigger_alarm("NG", duration=2.0)  # 报警2秒
alarm.trigger_alarm("ERROR", duration=3.0)
alarm.trigger_alarm("WARNING", duration=1.0)

# 停止报警
alarm.stop_alarm()
```

#### 硬件集成（可选）

在实际系统中，可以集成硬件报警设备：

```python
class HardwareAlarmController(AlarmController):
    def __init__(self, serial_port='COM3'):
        super().__init__()
        self.serial_port = serial_port
        self.serial = None
        self._connect_hardware()
    
    def _connect_hardware(self):
        """连接硬件报警设备"""
        import serial
        try:
            self.serial = serial.Serial(self.serial_port, 9600)
            self.logger.info(f"硬件报警设备已连接: {self.serial_port}")
        except Exception as e:
            self.logger.error(f"连接硬件失败: {e}")
    
    def _play_sound(self, alarm_type: str, duration: float):
        """控制硬件设备"""
        if self.serial and self.serial.is_open:
            if alarm_type == "NG":
                self.serial.write(b"NG_ON\n")
            elif alarm_type == "ERROR":
                self.serial.write(b"ERROR_ON\n")
            elif alarm_type == "WARNING":
                self.serial.write(b"WARNING_ON\n")
            
            time.sleep(duration)
            self.serial.write(b"ALARM_OFF\n")
```

---

## 完整工作流程示例

### 场景1：手动单次检测

```python
import cv2
from inspection_system import InspectionEngine, InspectionConfig
from defect_detection import ComprehensiveDefectDetector

# 初始化
config = InspectionConfig()
config.PIXEL_TO_MM = 0.098  # 设置像素-毫米转换系数

inspection_engine = InspectionEngine(config)
defect_detector = ComprehensiveDefectDetector()

# 加载图像
image = cv2.imread("part.jpg")

# 执行尺寸检测
result = inspection_engine.detect_circle(
    image,
    part_id="TEST001",
    part_type="圆形",
    nominal_size=5.0
)

# 执行缺陷检测
defect_result = defect_detector.detect_all(image)

# 输出结果
print(f"尺寸检测结果: {'合格' if result.is_qualified else '不合格'}")
print(f"缺陷检测结果: {'有缺陷' if defect_result.has_defect else '无缺陷'}")
print(f"质量评分: {defect_result.quality_score:.2f}")

# 综合判断
is_final_pass = result.is_qualified and not defect_result.has_defect
print(f"最终结果: {'合格' if is_final_pass else '不合格'}")
```

### 场景2：批量自动检测

```python
from inspection_system import InspectionEngine, InspectionConfig
from batch_inspection import BatchInspectionEngine, CameraBatchAcquisition
from inspection_system import USBCameraDriver

# 初始化
config = InspectionConfig()
inspection_engine = InspectionEngine(config)

# 创建批量检测引擎
batch_engine = BatchInspectionEngine(inspection_engine)
batch_engine.start()

# 设置结果回调
def on_result(result):
    if result.is_passed:
        print(f"✓ {result.part_id}: 合格")
    else:
        print(f"✗ {result.part_id}: 不合格")

batch_engine.set_result_callback(on_result)

# 启动相机批量采集
camera = USBCameraDriver()
if camera.connect():
    acquisition = CameraBatchAcquisition(camera, batch_engine)
    acquisition.start_acquisition(part_type="圆形", nominal_size=5.0)
    
    # 运行一段时间后停止
    import time
    time.sleep(60)
    
    acquisition.stop_acquisition()
    batch_engine.stop()
    
    # 获取统计信息
    stats = batch_engine.get_statistics()
    print(f"总计检测: {stats['completed_tasks']} 件")
    print(f"合格率: {stats['pass_rate']:.2f}%")
    print(f"平均速度: {stats['current_speed']:.1f} 件/分钟")
```

### 场景3：缺陷检测和质量评分

```python
import cv2
from defect_detection import ComprehensiveDefectDetector

# 加载图像
image = cv2.imread("part_with_defects.jpg")

# 执行综合缺陷检测
detector = ComprehensiveDefectDetector()
result = detector.detect_all(image)

# 分析缺陷
if result.has_defect:
    print(f"发现 {len(result.defects)} 个缺陷")
    
    # 按类型统计
    defect_types = {}
    for defect in result.defects:
        dtype = defect.defect_type.value
        if dtype not in defect_types:
            defect_types[dtype] = 0
        defect_types[dtype] += 1
    
    print("\n缺陷类型分布:")
    for dtype, count in defect_types.items():
        print(f"  {dtype}: {count} 个")
    
    # 评估严重程度
    severe_defects = [d for d in result.defects if d.severity > 0.7]
    if severe_defects:
        print(f"\n警告: 发现 {len(severe_defects)} 个严重缺陷！")
    
    # 绘制缺陷标记
    result_image = detector.draw_defects(image, result)
    cv2.imwrite("defect_marked.jpg", result_image)
else:
    print("未发现缺陷，零件质量良好")

# 质量评分
print(f"\n质量评分: {result.quality_score:.2f} / 1.00")
if result.quality_score > 0.9:
    print("评级: 优秀")
elif result.quality_score > 0.7:
    print("评级: 良好")
elif result.quality_score > 0.5:
    print("评级: 合格")
else:
    print("评级: 不合格")
```

---

## 性能优化建议

### 1. 批量检测性能

```python
# 调整工作线程数
import multiprocessing

# 获取CPU核心数
cpu_count = multiprocessing.cpu_count()

# 设置工作线程数为CPU核心数的1.5倍
max_workers = int(cpu_count * 1.5)

batch_config = BatchInspectionConfig(
    max_workers=max_workers,
    target_speed=60
)
```

### 2. 图像预处理优化

```python
# 禁用亚像素检测以提高速度
batch_config.enable_subpixel = False

# 调整图像队列大小
batch_config.image_queue_size = 20  # 增大队列
```

### 3. 内存管理

```python
# 定期清理结果
batch_engine.clear_results()

# 仅保存不合格图像
batch_config.save_failures_only = True
```

---

## 故障排查

### 问题1：批量检测速度低于预期

**可能原因**：
- 工作线程数设置过少
- 图像处理算法耗时过长
- 磁盘I/O瓶颈

**解决方案**：
```python
# 增加工作线程数
batch_config.max_workers = 8

# 禁用亚像素检测
batch_config.enable_subpixel = False

# 批量保存而非实时保存
batch_config.auto_save = False
```

### 问题2：缺陷检测误报率高

**可能原因**：
- 检测阈值设置过低
- 图像质量不佳
- 参数不适用当前场景

**解决方案**：
```python
# 提高检测阈值
surface_detector.defect_threshold = 50

# 增加最小缺陷面积
surface_detector.min_defect_area = 50

# 提高毛刺检测阈值
burr_detector.burr_threshold = 5.0
```

### 问题3：内存占用过高

**可能原因**：
- 图像队列过大
- 结果未及时清理
- 图像分辨率过高

**解决方案**：
```python
# 减小图像队列
batch_config.image_queue_size = 5

# 定期清理结果
batch_engine.clear_results()

# 降低图像分辨率
image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
```

---

## 最佳实践

1. **标定优先**：使用前务必进行相机标定，确保测量精度
2. **参数调优**：根据实际零件类型调整检测参数
3. **定期验证**：使用标准件定期验证系统精度
4. **数据备份**：定期备份检测数据和结果
5. **性能监控**：持续监控检测速度和合格率，及时优化

---

## 技术支持

如有问题，请查看：
- 项目README.md
- 代码文档字符串
- 日志文件（logs/目录）

---

**文档版本**: V1.0  
**最后更新**: 2026-02-10