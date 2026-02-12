# 微小零件中高精度视觉检测系统 V3.0

## 项目概述

本项目是基于OpenCV计算机视觉技术和亚像素级检测算法的自动化检测系统，用于检测直径20mm以下、长度40mm以下、精度等级为IT8级的微小零件。

**版本**: V3.0 (2026-02-10)  
**精度目标**: IT8级（亚像素检测精度≥1/20像素）  
**检测速度**: ≥60件/分钟（批量检测模式）

---

## V3.0 新增功能

### 核心功能增强

| 功能 | 模块 | 说明 |
|------|------|------|
| 批量自动检测 | `batch_inspection.py` | 多线程并行处理，支持≥60件/分钟检测速度 |
| 缺陷检测 | `defect_detection.py` | 表面缺陷、边缘缺损、毛刺检测 |
| 声光报警 | `inspection_gui_enhanced.py` | 不合格品时自动触发报警 |
| 历史数据查询 | `inspection_gui_enhanced.py` | 按时间、类型、结果查询历史记录 |
| 统计报表 | `inspection_gui_enhanced.py` | 合格率、趋势图表可视化 |
| 增强GUI | `inspection_gui_enhanced.py` | 统一界面，集成所有功能 |

### 性能提升

- **检测速度**: 批量检测模式下可达60件/分钟以上
- **多线程优化**: 支持并行图像处理
- **内存优化**: 任务队列管理，避免内存溢出
- **实时反馈**: 实时显示检测进度和统计信息

---

## 快速开始

### 环境要求

- Python 3.8+
- Windows 10/11 64位
- 4GB+ 内存（推荐8GB+）
- 工业相机或USB相机

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动程序

```bash
# 启动增强版GUI
python inspection_gui_enhanced.py

# 或启动Web版本
python inspection_web_v2.py
```

### 运行测试

```bash
# 测试新功能
python test_new_features.py

# 运行单元测试
python -m pytest tests/ -v
```

---

## 功能模块说明

### 1. 批量检测模块 (`batch_inspection.py`)

**功能特性**:
- 生产者-消费者架构
- 多线程并行处理
- 实时性能监控
- 任务队列管理
- 暂停/恢复/停止控制

**使用示例**:

```python
from inspection_system import InspectionEngine, InspectionConfig
from batch_inspection import BatchInspectionEngine

# 初始化
config = InspectionConfig()
inspection_engine = InspectionEngine(config)
batch_engine = BatchInspectionEngine(inspection_engine)

# 设置回调
def on_result(result):
    print(f"{result.part_id}: {'合格' if result.is_passed else '不合格'}")

batch_engine.set_result_callback(on_result)

# 启动批量检测
batch_engine.start()

# 添加任务
import cv2
image = cv2.imread("part.jpg")
batch_engine.add_image(image, part_id="PART001", part_type="圆形")

# 停止
batch_engine.stop()
```

**性能参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| max_workers | 4 | 工作线程数 |
| target_speed | 60 | 目标速度（件/分钟） |
| image_queue_size | 10 | 图像队列大小 |
| enable_subpixel | True | 启用亚像素检测 |

---

### 2. 缺陷检测模块 (`defect_detection.py`)

**缺陷类型**:
- **表面缺陷**: 划痕、污渍、孔洞、斑点
- **边缘缺陷**: 缺口、崩边、锯齿
- **毛刺**: 边缘毛刺和突起

**检测算法**:
- 局部对比度分析
- 形态学处理
- 纹理分析
- 凸包缺陷检测
- 轮廓距离分析

**使用示例**:

```python
from defect_detection import ComprehensiveDefectDetector
import cv2

# 初始化
detector = ComprehensiveDefectDetector()

# 加载图像
image = cv2.imread("part.jpg", cv2.IMREAD_GRAYSCALE)

# 执行检测
result = detector.detect_all(image)

# 查看结果
if result.has_defect:
    print(f"发现 {len(result.defects)} 个缺陷")
    for defect in result.defects:
        print(f"  {defect.defect_type.value}: 严重程度={defect.severity:.2f}")
else:
    print("未发现缺陷")

# 绘制缺陷标记
result_image = detector.draw_defects(image, result)
cv2.imwrite("result.jpg", result_image)
```

**质量评分**:
- 0.9-1.0: 优秀
- 0.7-0.9: 良好
- 0.5-0.7: 合格
- <0.5: 不合格

---

### 3. 增强版GUI (`inspection_gui_enhanced.py`)

**界面组成**:

```
┌─────────────────────────────────────────────────────────┐
│ 菜单栏: 文件 | 相机 | 检测 | 工具 | 帮助                  │
├─────────────────────┬───────────────────────────────────┤
│                     │  批量检测控制                      │
│   图像显示区域       │  ┌─────┬─────┬─────┐              │
│                     │  │启动 │暂停 │停止 │              │
│   [图像预览]         │  └─────┴─────┴─────┘              │
│                     │  状态: 运行中                      │
│   [缩放控制]         │  已检测: 100 | 合格: 95          │
│                     │  [进度条]                         │
│   [相机控制]         │                                   │
│   [连接] [采集]      │  检测结果                         │
│                     │  ┌─────────────────────────────┐ │
└─────────────────────┴─│ 零件编号: PART001            │ │
                         │ 实测值: 5.023 mm            │ │
                         │ 结果: 合格                   │ │
                         └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**主要功能**:
- 图像采集和预览
- 单次检测和批量检测
- 缺陷检测和质量评分
- 声光报警控制
- 历史数据查询
- 统计报表生成
- 图纸标注工具

---

### 4. 声光报警功能

**报警类型**:
- **NG**: 检测到不合格品
- **ERROR**: 系统错误
- **WARNING**: 警告信息

**使用方式**:

```python
from inspection_gui_enhanced import AlarmController

# 创建报警控制器
alarm = AlarmController()

# 启用报警
alarm.enable_alarm(True)

# 触发报警
alarm.trigger_alarm("NG", duration=2.0)

# 停止报警
alarm.stop_alarm()
```

**硬件集成**:
支持通过串口控制硬件报警设备（LED灯、蜂鸣器等）

---

### 5. 历史数据查询

**查询条件**:
- 时间范围
- 零件类型
- 检测结果

**导出格式**:
- Excel (.xlsx)
- CSV (.csv)

**使用方式**:
1. 在GUI中选择"文件" → "历史数据查询"
2. 设置查询条件
3. 点击"查询"查看结果
4. 点击"导出"保存数据

---

### 6. 统计报表

**图表类型**:
- 合格率饼图
- 零件类型分布
- 检测数量时间趋势
- 测量值分布直方图

**统计指标**:
- 总检测数
- 合格数/不合格数
- 合格率
- 平均值/标准差
- 过程能力指数 (Cpk)

**使用方式**:
1. 在GUI中选择"文件" → "统计报表"
2. 查看自动生成的图表
3. 点击"导出报表"保存为Excel

---

## 项目结构

```
微小零件低精度视觉检测/
├── batch_inspection.py          # 批量检测模块
├── defect_detection.py          # 缺陷检测模块
├── inspection_gui_enhanced.py   # 增强版GUI
├── inspection_system.py         # 核心检测系统
├── inspection_system_gui_v2.py  # 原GUI（标注工具）
├── inspection_web_v2.py         # Web版本
├── core_detection.py            # 核心检测算法
├── camera_manager.py            # 相机管理
├── calibration_manager.py       # 标定管理
├── config_manager.py            # 配置管理
├── data_export.py               # 数据导出
├── drawing_annotation.py        # 图纸标注
├── dxf_parser.py                # DXF解析
├── exceptions.py                # 异常定义
├── logger_config.py             # 日志配置
├── USAGE_GUIDE.md               # 使用指南
├── requirements.txt             # 依赖列表
├── test_new_features.py         # 新功能测试
├── tests/                       # 单元测试
│   ├── test_core_detection.py
│   ├── test_config_manager.py
│   └── ...
├── data/                        # 数据目录
│   ├── images/                  # 检测图像
│   ├── inspection_results.csv   # 检测结果
│   └── inspection_errors.csv    # 错误记录
├── templates/                   # 模板目录
│   ├── index.html
│   └── drawing_annotation.html
└── logs/                        # 日志目录
    └── inspection_system_*.log
```

---

## 技术架构

### 分层架构

```
┌─────────────────────────────────────┐
│      用户界面层 (GUI/Web)           │
│  - inspection_gui_enhanced.py       │
│  - inspection_web_v2.py             │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│         业务逻辑层                   │
│  - batch_inspection.py              │
│  - defect_detection.py              │
│  - inspection_system.py             │
│  - data_export.py                   │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│         核心算法层                   │
│  - core_detection.py                │
│  - 亚像素检测                       │
│  - 几何拟合                         │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│         硬件接口层                   │
│  - camera_manager.py                │
│  - calibration_manager.py           │
└─────────────────────────────────────┘
```

### 数据流程

```
相机采集 → 预处理 → 边缘检测 → 轮廓提取 → 尺寸计算 → 公差判断 → 结果输出
    ↓         ↓          ↓          ↓          ↓          ↓          ↓
  实时预览   图像增强    亚像素细化  几何拟合   IT8判断   合格/不合格   显示/保存
    ↓                                                    ↓
  批量队列                                            声光报警
```

---

## 配置说明

### IT8级公差标准

| 公称尺寸段（mm） | IT8公差值（mm） | IT7公差值（mm） | IT11公差值（mm） |
|------------------|------------------|------------------|------------------|
| >1 - 3 | 0.014 | 0.010 | 0.060 |
| >3 - 6 | 0.018 | 0.012 | 0.075 |
| >6 - 10 | 0.022 | 0.015 | 0.090 |
| >10 - 18 | 0.027 | 0.018 | 0.110 |
| >18 - 30 | 0.033 | 0.021 | 0.130 |
| >30 - 40 | 0.039 | 0.025 | 0.160 |

### 检测参数

```python
class InspectionConfig:
    # 像素-毫米转换系数
    PIXEL_TO_MM: float = 0.098  # 需要标定
    
    # 亚像素精度
    SUBPIXEL_PRECISION: float = 1.0 / 20.0  # 1/20像素
    
    # 图像处理参数
    GAUSSIAN_KERNEL: int = 5
    MEDIAN_KERNEL: int = 5
    CANNY_THRESHOLD_LOW: int = 50
    CANNY_THRESHOLD_HIGH: int = 150
```

---

## 性能指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 检测精度 | IT8级 | ±0.014~0.039mm |
| 亚像素精度 | ≥1/20像素 | 0.05像素 |
| 检测速度 | ≥60件/分钟 | 批量检测模式 |
| 图像处理延迟 | <300ms/张 | 含亚像素处理 |
| 系统响应时间 | <1s | UI响应 |
| 系统可用性 | ≥99.5% | 7x24运行 |

---

## 开发指南

### 代码规范

- 遵循PEP 8规范
- 使用类型注解
- 编写文档字符串
- 异常处理完善
- 日志记录详细

详见 `CODE_STYLE.md`

### 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行覆盖率测试
python -m pytest tests/ --cov=. --cov-report=html

# 测试新功能
python test_new_features.py
```

### 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'feat: Add AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

## 常见问题

### Q1: 批量检测速度低于预期？

**A**: 检查以下配置：
- 工作线程数是否合适（建议为CPU核心数的1-2倍）
- 图像队列大小是否足够
- 是否启用了亚像素检测（可暂时禁用以提高速度）

### Q2: 缺陷检测误报率高？

**A**: 调整检测参数：
- 提高检测阈值 (`defect_threshold`)
- 增加最小缺陷面积 (`min_defect_area`)
- 提高毛刺检测阈值 (`burr_threshold`)

### Q3: 如何提高检测精度？

**A**:
- 进行相机标定
- 使用高质量的图像
- 启用亚像素检测
- 确保光照均匀
- 使用远心镜头

### Q4: 如何集成硬件报警设备？

**A**: 参考 `USAGE_GUIDE.md` 中的"硬件集成"章节，实现 `HardwareAlarmController` 类。

---

## 更新日志

### V3.0 (2026-02-10)

**新增**:
- 批量检测模块，支持≥60件/分钟检测速度
- 缺陷检测模块（表面、边缘、毛刺）
- 声光报警功能
- 历史数据查询
- 统计报表和图表
- 增强版GUI，集成所有功能

**改进**:
- 优化多线程处理性能
- 改进异常处理机制
- 增强日志系统
- 完善代码文档

**修复**:
- 修复批量检测内存泄漏问题
- 修复缺陷检测误报问题

### V2.0 (2026-02-09)

**新增**:
- 图纸标注工具
- DXF文件导入
- 基于标注的检测
- Web版本

### V1.0 (2026-02-09)

**初始版本**:
- 基础检测功能
- GUI界面
- 相机管理
- 数据导出

---

## 致谢

感谢以下开源项目的支持：
- OpenCV
- NumPy
- Pandas
- Flask
- PyQt5
- Matplotlib

---

## 许可证

本项目采用 MIT 许可证。

---

## 联系方式

- 项目主页: [GitHub仓库链接]
- 问题反馈: [Issues链接]
- 邮箱: [项目邮箱]

---

**文档版本**: V3.0  
**最后更新**: 2026-02-10  
**维护者**: 项目团队