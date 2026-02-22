# AGENTS.md - 项目上下文文档

## 项目概述

本项目是**微小零件中高精度视觉检测系统**的完整实现，基于OpenCV计算机视觉技术和亚像素级检测算法的自动化检测系统，用于检测直径20mm以下、长度40mm以下、精度等级为IT8级的微小零件，并包含IT5级精度检测的技术分析。

**项目类型：** 完整产品实现 + 产品规划与设计文档

**创建日期：** 2026年2月9日

**最后更新：** 2026年2月22日

**项目状态：** V3.0 生产可用版本

**Git仓库：** https://github.com/jxjk/openCVDemo.git

**精度目标：** IT8级（中等高精度，比IT11级提升约4倍）

**核心技术：** 亚像素级检测算法（1/20像素精度），批量检测（≥60件/分钟），缺陷检测

---

## 目录内容

当前目录包含以下核心文档、实现代码和参考程序：

### 核心实现代码

| 文件名 | 类型 | 内容描述 |
|--------|------|----------|
| `inspection_gui_enhanced.py` | 主程序 | 增强版GUI，集成所有功能的主界面 |
| `batch_inspection.py` | 核心模块 | 批量检测引擎，支持≥60件/分钟检测速度 |
| `defect_detection.py` | 核心模块 | 缺陷检测模块（表面、边缘、毛刺） |
| `inspection_system.py` | 核心模块 | 核心检测系统，包含检测引擎和数据管理 |
| `core_detection.py` | 核心模块 | 核心检测算法（边缘检测、轮廓提取、几何拟合） |
| `camera_manager.py` | 核心模块 | 相机管理（USB相机、大恒相机） |
| `calibration_manager.py` | 核心模块 | 标定管理（相机内参标定、像素-毫米标定） |
| `config_manager.py` | 核心模块 | 配置管理（IT5/IT7/IT8/IT9/IT11公差表） |
| `data_export.py` | 核心模块 | 数据导出（Excel、CSV）和统计计算 |
| `drawing_annotation.py` | 核心模块 | 图纸标注和基于标注的检测 |
| `dxf_parser.py` | 核心模块 | DXF文件解析和模板转换 |
| `dwg_converter.py` | 核心模块 | DWG文件转换模块，支持DWG转DXF |
| `image_registration.py` | 核心模块 | 图像配准模块，支持多种特征检测和变换方法 |
| `auto_measurement.py` | 核心模块 | 自动测量引擎，基于DWG/DXF图纸自动测量 |
| `exceptions.py` | 核心模块 | 异常定义和处理 |
| `logger_config.py` | 核心模块 | 日志配置 |

### GUI工具模块

| 文件名 | 类型 | 内容描述 |
|--------|------|----------|
| `calibration_gui.py` | GUI工具 | 相机内参标定GUI（PyQt5） |
| `pixel_mm_calibration_gui.py` | GUI工具 | 像素-毫米标定GUI（PyQt5） |
| `camera_preview_gui.py` | GUI工具 | 实时图像预览GUI（PyQt5） |
| `result_visualization_gui.py` | GUI工具 | 检测结果可视化GUI（PyQt5） |
| `config_editor_gui.py` | GUI工具 | 配置管理GUI（PyQt5） |
| `inspection_system_gui_v2.py` | GUI工具 | 图纸标注工具（Tkinter） |
| `dwg_auto_measurement_gui.py` | GUI工具 | DWG自动测量GUI，集成图纸解析、图像配准和自动测量功能 |

### Web版本

| 文件名 | 类型 | 内容描述 |
|--------|------|----------|
| `inspection_web_v2.py` | Web应用 | Flask + SocketIO实现的Web版本 |

### 文档目录

| 文件名 | 类型 | 内容描述 |
|--------|------|----------|
| `产品需求规格说明书.md` | 产品文档 | 完整的产品功能需求、性能需求、技术约束和验收标准 |
| `图纸检测部位标定方案.md` | 技术方案 | 2D/3D图纸检测部位预先标定的实现方案和代码示例 |
| `产品进化接口设计方案.md` | 架构设计 | 系统架构设计和可扩展接口方案，包含完整的Python接口代码 |
| `亚像素IT5级精度检测分析.md` | 技术分析 | IT5级精度检测的可行性分析，包含Zernike矩等高级算法实现代码 |
| `DWG_AUTO_MEASUREMENT_GUIDE.md` | 使用文档 | DWG自动测量功能使用指南，包含完整的操作流程和API文档 |
| `README_V3.md` | 使用文档 | V3.0版本说明文档 |
| `README.md` | 使用文档 | 项目总体说明文档 |
| `USAGE_GUIDE.md` | 使用文档 | 详细使用指南和API文档 |
| `GUI_MODULES_USAGE.md` | 使用文档 | GUI模块使用指南 |
| `CODE_STYLE.md` | 开发文档 | 代码规范和最佳实践 |
| `REFACTORING_REPORT.md` | 开发文档 | 代码重构报告 |
| `INSTALL.md` | 安装文档 | 安装和配置指南 |
| `DOCKER_DEPLOY.md` | 部署文档 | Docker部署指南 |

### 归档目录

| 目录名 | 内容描述 |
|--------|----------|
| `archive/` | 历史版本归档，包含旧版本的GUI和测试文件 |
| `archive/old_versions/` | 旧版本的GUI实现文件 |
| `archive/test_files/` | 测试辅助文件 |

### 参考程序目录

| 目录名 | 内容描述 |
|--------|----------|
| `参考程序/11.detectOD` | 外径检测程序，使用大恒相机SDK，实现同步带轮齿数和OD尺寸检测 |
| `参考程序/7.chiCunJianCe.demo` | 尺寸检测演示程序，包含相机标定、圆拟合、图像变换等核心算法实现 |

### 测试和配置

| 目录/文件 | 内容描述 |
|-----------|----------|
| `tests/` | 单元测试目录 |
| `tests/test_auto_measurement.py` | DWG自动测量功能单元测试 |
| `test_new_features.py` | 新功能测试脚本 |
| `run_tests.py` | 单元测试运行脚本（支持运行所有测试或特定测试文件） |
| `requirements.txt` | Python依赖库列表 |
| `config.json` | 系统配置文件 |
| `install_dependencies.bat` | Windows依赖安装脚本 |
| `data/` | 数据目录（检测结果、图像） |
| `templates/` | Web模板目录 |
| `logs/` | 日志目录 |

---

## 项目核心信息

### 应用领域
- 电子制造业微小元件检测
- 精密机械零部件质量检测
- 汽车零部件尺寸检测
- 医疗器械微小零件检测

### 技术栈

**开发语言：**
- Python 3.8+

**图像处理库：**
- OpenCV (cv2) >= 4.5.0：图像采集、预处理、边缘检测、轮廓提取、角点检测
- OpenCV contrib-python：额外功能模块
- NumPy >= 1.19.0：数组运算、数学计算
- SciPy >= 1.5.0：科学计算、高级算法
- PIL (Pillow) >= 8.0.0：图像格式转换

**GUI框架：**
- Tkinter：图形用户界面（inspection_system_gui_v2.py）
- PyQt5：高级GUI框架（calibration_gui.py、pixel_mm_calibration_gui.py等）

**Web框架：**
- Flask >= 2.0.0：Web应用框架
- Flask-SocketIO >= 5.0.0：实时通信
- Flask-CORS >= 3.0.0：跨域支持
- Eventlet >= 0.30.0：异步处理

**图像采集：**
- 大恒相机SDK (gxipy)：大水星系列工业相机驱动（可选）
- OpenCV VideoCapture：通用相机接口

**数据处理：**
- Pandas >= 1.2.0：数据分析、CSV文件处理
- CSV模块：数据记录
- OpenPyXL >= 3.0.0：Excel文件处理

**可视化：**
- Matplotlib >= 3.3.0：图表绘制、统计可视化

**CAD解析：**
- ezdxf >= 1.0.0：DXF文件解析（图纸标注功能）

**相机标定：**
- OpenCV calibrateCamera：相机内参标定
- OpenCV cornerSubPix：亚像素角点定位

**几何算法：**
- 最小二乘法圆拟合
- 欧氏距离计算
- 形态学操作
- 凸包分析

**亚像素检测：**
- OpenCV cornerSubPix：亚像素角点定位（1/20像素精度）
- 自定义亚像素算法：多项式插值、迭代拟合

**测试框架：**
- Pytest >= 7.0.0：单元测试框架
- Pytest-cov >= 4.0.0：测试覆盖率
- Coverage >= 7.0.0：代码覆盖率统计

**其他工具：**
- tqdm >= 4.50.0：进度条显示
- YAML：配置文件解析

### 检测范围
- **零件直径：** 1mm - 20mm
- **零件长度：** 2mm - 40mm
- **检测精度：** IT8级公差标准（中等高精度）
- **亚像素精度：** ≥1/20像素
- **批量检测速度：** ≥60件/分钟

### 性能指标
- **检测速度：** ≥60件/分钟（批量检测模式）
- **检测精度：** 符合IT8级标准
- **亚像素精度：** ≥1/20像素
- **图像处理延迟：** <300ms/张（含亚像素处理）
- **系统响应时间：** <1s
- **系统可用性：** ≥99.5%
- **启动时间：** <10s

---

## V3.0 核心功能

### 1. 批量自动检测
- **模块：** `batch_inspection.py`
- **架构：** 生产者-消费者模式，多线程并行处理
- **性能：** ≥60件/分钟
- **特性：**
  - 实时性能监控
  - 任务队列管理
  - 暂停/恢复/停止控制
  - 进度回调
  - 结果回调

### 2. 缺陷检测
- **模块：** `defect_detection.py`
- **缺陷类型：**
  - 表面缺陷（划痕、污渍、孔洞、斑点）
  - 边缘缺陷（缺口、崩边、锯齿）
  - 毛刺（边缘毛刺和突起）
- **检测算法：**
  - 局部对比度分析
  - 形态学处理
  - 纹理分析
  - 凸包缺陷检测
  - 轮廓距离分析
- **质量评分：** 0-1范围，0.9-1.0优秀，<0.5不合格

### 3. 声光报警
- **模块：** `inspection_gui_enhanced.py`
- **功能：**
  - 不合格品自动报警
  - NG/ERROR/WARNING三种报警类型
  - 可配置报警时长
  - 支持硬件设备集成

### 4. 历史数据查询
- **模块：** `inspection_gui_enhanced.py`
- **查询条件：**
  - 时间范围
  - 零件类型
  - 检测结果
- **导出格式：** Excel、CSV

### 5. 统计报表
- **模块：** `inspection_gui_enhanced.py`
- **图表类型：**
  - 合格率饼图
  - 零件类型分布
  - 检测数量时间趋势
  - 测量值分布直方图
- **统计指标：**
  - 总检测数、合格数、不合格数
  - 合格率
  - 平均值、标准差
  - 过程能力指数（Cp/Cpk）

### 6. 增强版GUI
- **模块：** `inspection_gui_enhanced.py`
- **界面组成：**
  - 图像显示区域
  - 批量检测控制面板
  - 检测结果显示
  - 相机控制
  - 菜单栏（文件、相机、检测、工具、帮助）

### 7. 相机标定工具集
- **相机内参标定：** `calibration_gui.py`
  - 棋盘格标定
  - 实时预览和图像采集
  - 重投影误差可视化
  - 标定结果保存/加载
- **像素-毫米标定：** `pixel_mm_calibration_gui.py`
  - 参考长度标定（交互式选点）
  - 圆形标定（自动检测）
  - 标定历史记录
- **实时预览：** `camera_preview_gui.py`
  - 30fps实时预览
  - 曝光/增益参数调整
  - 图像质量指标显示

### 8. 图纸标注功能
- **模块：** `drawing_annotation.py`, `inspection_system_gui_v2.py`
- **功能：**
  - DXF文件导入
  - 图纸标注（圆形、线段、矩形、角度）
  - 标注模板管理
  - 基于标注的检测

### 9. 配置管理
- **模块：** `config_manager.py`, `config_editor_gui.py`
- **支持配置：**
  - IT5/IT7/IT8/IT9/IT11公差表
  - 检测参数
  - 路径配置
  - 相机参数

### 10. DWG自动测量（V3.1新增）
- **模块：** `dwg_converter.py`, `image_registration.py`, `auto_measurement.py`, `dwg_auto_measurement_gui.py`
- **功能：**
  - DWG文件自动转换为DXF（可选，需要ODA File Converter）
  - DXF图纸自动解析和标注提取
  - 图纸渲染为图像
  - 图像配准（ORB/SIFT/SURF/AKAZE特征检测）
  - 基于图纸的自动测量
  - 测量报告生成（JSON + 文本格式）
- **GUI工具：** `dwg_auto_measurement_gui.py`
  - 4步操作流程：加载图纸 → 加载图像 → 图像配准 → 自动测量
  - 实时进度显示
  - 可视化配准结果
  - 测量结果表格展示
  - 报告导出功能
- **使用流程：**
  1. 加载DWG/DXF文件
  2. 系统自动解析图纸标注
  3. 加载待测零件图像
  4. 图纸与图像自动配准
  5. 基于标注自动测量
  6. 生成详细测量报告
- **技术特性：**
  - 支持多种特征检测算法（ORB、SIFT、SURF、AKAZE）
  - 支持多种变换方法（单应性、仿射、相似、平移）
  - 自动提取尺寸标注信息
  - 支持圆形和线段测量
  - 自动计算偏差和合格性判断
  - 详细的测量结果报告

---

## 系统架构

### 分层架构

```
┌─────────────────────────────────────────────────────┐
│         用户界面层 (GUI/Web)                         │
│  - inspection_gui_enhanced.py (主GUI)              │
│  - calibration_gui.py (标定工具)                   │
│  - pixel_mm_calibration_gui.py (像素标定)          │
│  - camera_preview_gui.py (实时预览)                │
│  - result_visualization_gui.py (结果可视化)        │
│  - config_editor_gui.py (配置管理)                 │
│  - dwg_auto_measurement_gui.py (DWG自动测量)       │
│  - inspection_web_v2.py (Web版本)                  │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│         业务逻辑层                                  │
│  - batch_inspection.py (批量检测)                  │
│  - defect_detection.py (缺陷检测)                  │
│  - inspection_system.py (检测引擎)                 │
│  - data_export.py (数据导出)                       │
│  - drawing_annotation.py (图纸标注)                │
│  - auto_measurement.py (自动测量)                  │
│  - dwg_converter.py (DWG转换)                      │
│  - image_registration.py (图像配准)                │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│         核心算法层                                  │
│  - core_detection.py (检测算法)                    │
│  - dxf_parser.py (DXF解析)                         │
│  - 亚像素检测 (cornerSubPix, 1/20像素)             │
│  - 几何拟合 (最小二乘法、RANSAC)                   │
│  - 特征检测 (ORB/SIFT/SURF/AKAZE)                 │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│         硬件接口层                                  │
│  - camera_manager.py (相机管理)                    │
│  - calibration_manager.py (标定管理)               │
└─────────────────────────────────────────────────────┘
```

### 数据流程

**标准检测流程：**
```
相机采集 → 预处理 → 边缘检测 → 轮廓提取 → 尺寸计算 → 公差判断 → 结果输出
    ↓         ↓          ↓          ↓          ↓          ↓          ↓
  实时预览   图像增强    亚像素细化  几何拟合   IT8判断   合格/不合格   显示/保存
    ↓                                                    ↓
  批量队列                                            声光报警
    ↓
  缺陷检测
    ↓
  质量评分
```

**DWG自动测量流程：**
```
DWG文件 → 转换DXF → 解析标注 → 渲染图纸
                                       ↓
图像文件 ─────────────────────→ 图像配准 → 坐标转换
                                       ↓
                             自动测量 → 生成报告
```

---

## IT8级公差标准

| 公称尺寸段（mm） | IT5公差值（mm） | IT7公差值（mm） | IT8公差值（mm） | IT11公差值（mm） | IT8提升（vs IT11） |
|------------------|------------------|------------------|------------------|------------------|-------------------|
| >1 - 3 | 0.004 | 0.010 | 0.014 | 0.060 | 4.3倍 |
| >3 - 6 | 0.005 | 0.012 | 0.018 | 0.075 | 4.2倍 |
| >6 - 10 | 0.006 | 0.015 | 0.022 | 0.090 | 4.1倍 |
| >10 - 18 | 0.008 | 0.018 | 0.027 | 0.110 | 4.1倍 |
| >18 - 30 | 0.009 | 0.021 | 0.033 | 0.130 | 3.9倍 |
| >30 - 40 | 0.011 | 0.025 | 0.039 | 0.160 | 4.1倍 |

**精度等级说明：**
- IT3-IT5级：超高精度（精密测量仪器、航空航天）← 未来目标（分析阶段）
- IT7级：高精度（精密轴承、齿轮）
- IT8级：中等高精度（精密机械零件）← 本项目当前目标
- IT9级：中等精度（一般机械零件）
- IT11级：低精度（粗加工零件）

---

## 硬件要求

| 硬件类型 | 最低配置 | 推荐配置 |
|----------|----------|----------|
| CPU | Intel Core i5 或同等性能 | Intel Core i7 或更高 |
| 内存 | 8GB | 16GB |
| 存储 | 100GB可用空间 | 200GB可用空间（SSD） |
| 相机 | 1000万像素以上工业相机 | 2000万像素工业相机 |
| 镜头 | 定焦镜头，放大倍数0.5X - 5X | 远心镜头，放大倍数1X - 5X |
| 光源 | 环形LED光源 | 高亮度环形LED光源 |
| 标定板 | IT7级标定板 | IT5级标定板 |
| 操作系统 | Windows 10/11 64位 | Windows 10/11 64位 |

**说明：** IT8级精度对硬件要求比IT11级提高约2-3倍，但远低于IT5级的要求。

---

## 快速开始

### 环境要求

- Python 3.8+
- Windows 10/11 64位
- 4GB+ 内存（推荐8GB+）
- 工业相机或USB相机

### 安装依赖

**Windows用户（推荐）：**
```bash
# 双击运行安装脚本
install_dependencies.bat
```

**手动安装：**
```bash
pip install -r requirements.txt
```

**依赖库包括：**
- Web框架：Flask, Flask-SocketIO, Flask-CORS, eventlet
- 图像处理：OpenCV, NumPy, SciPy, Pillow
- 数据处理：Pandas, openpyxl
- CAD解析：ezdxf
- 可视化：Matplotlib
- 测试框架：Pytest, pytest-cov, coverage
- 其他工具：tqdm

### 启动程序

```bash
# 启动增强版GUI（推荐）
python inspection_gui_enhanced.py

# 启动DWG自动测量GUI（V3.1新增）
python dwg_auto_measurement_gui.py

# 启动Web版本
python inspection_web_v2.py

# 启动图纸标注工具
python inspection_system_gui_v2.py
```

### 运行测试

```bash
# 运行所有单元测试（推荐）
python run_tests.py

# 运行特定测试文件
python run_tests.py test_config_manager
python run_tests.py test_core_detection
python run_tests.py test_data_export
python run_tests.py test_exceptions
python run_tests.py test_auto_measurement

# 测试新功能
python test_new_features.py

# 或使用pytest（如果已安装）
python -m pytest tests/ -v
python -m pytest tests/ --cov=. --cov-report=html
```

---

## 使用指南

### 主界面菜单

**文件菜单：**
- 打开图像
- 历史数据查询
- 统计报表
- 退出

**相机菜单：**
- USB相机/大恒相机选择
- 连接相机
- 断开相机
- 实时预览

**检测菜单：**
- 单次检测
- 缺陷检测

**工具菜单：**
- 图纸标注
- 相机内参标定
- 像素-毫米标定
- 启用/禁用报警

**帮助菜单：**
- 关于

### 批量检测流程

1. 连接相机
2. 加载或配置检测模板
3. 点击"启动批量检测"
4. 系统自动连续检测
5. 实时查看进度和结果
6. 查看统计数据和报表

### 标定流程

**相机内参标定：**
1. 准备棋盘格标定板
2. 点击"工具" → "相机内参标定"
3. 采集多角度标定图像（10-20张）
4. 执行标定
5. 查看重投影误差
6. 保存标定结果

**像素-毫米标定：**
1. 准备标准件（已知尺寸）
2. 点击"工具" → "像素-毫米标定"
3. 选择标定方法（参考长度/圆形）
4. 采集图像
5. 标定
6. 应用到系统

### DWG自动测量流程（V3.1新增）

**使用GUI：**
```bash
python dwg_auto_measurement_gui.py
```

**操作步骤：**
1. **加载图纸**：点击"加载图纸"按钮，选择DWG或DXF文件
   - DWG文件会自动转换为DXF（需要ODA File Converter）
   - 系统自动解析图纸标注
   - 显示提取的标注数量

2. **加载图像**：点击"加载图像"按钮，选择待测量的零件图像
   - 支持常见图像格式（JPG、PNG、BMP等）

3. **图像配准**：点击"图像配准"按钮
   - 系统自动检测特征点
   - 计算变换矩阵
   - 显示配准置信度
   - 可视化配准结果

4. **自动测量**：点击"自动测量"按钮
   - 基于图纸标注自动测量
   - 计算偏差和合格性
   - 显示测量结果表格

5. **导出报告**：点击"导出报告"按钮
   - 生成JSON格式报告
   - 生成文本格式报告
   - 保存到指定目录

**命令行使用：**
```bash
# DWG转换
python dwg_converter.py input.dwg output.dxf

# 自动测量
python auto_measurement.py part.dwg part.jpg ./reports

# 或使用DXF文件
python auto_measurement.py part.dxf part.jpg ./reports
```

**代码集成：**
```python
from auto_measurement import AutoMeasurementEngine

# 创建测量引擎
engine = AutoMeasurementEngine()

# 执行测量
report = engine.measure_from_dwg("part.dwg", "part.jpg", "./reports")

# 查看结果
print(f"配准成功: {report.registration_success}")
print(f"配准置信度: {report.registration_confidence}")
print(f"测量特征数: {report.measured_features}")
for result in report.measurement_results:
    print(f"{result.feature_name}: {result.measured_value}mm, 偏差: {result.deviation}mm")
```

---

## 关键特性

### 1. 中高精度检测
- IT8级精度标准（比IT11级提升4倍）
- 亚像素级检测算法（1/20像素精度）
- 适用于精密机械、电子制造等领域

### 2. 高速批量检测
- 批量检测模式下≥60件/分钟
- 多线程并行处理
- 生产者-消费者架构
- 实时性能监控

### 3. 缺陷检测
- 表面缺陷、边缘缺陷、毛刺检测
- 多种检测算法组合
- 质量评分系统
- 缺陷标记和可视化

### 4. 模块化设计
- 所有核心功能都通过抽象接口定义
- 使用工厂模式管理组件创建
- 支持运行时动态替换实现

### 5. 可扩展性
- 插件系统支持动态加载功能模块
- 事件总线实现组件间解耦
- RESTful API支持外部集成
- 可向IT7级精度升级

### 6. 完善的工具集
- 相机内参标定工具
- 像素-毫米标定工具
- 实时预览工具
- 结果可视化工具
- 配置管理工具
- 图纸标注工具

### 7. 数据管理
- 多种存储方式（文件系统、CSV、Excel）
- 检测结果持久化
- 模板管理
- 历史数据查询
- 统计报表生成

### 8. 声光报警
- 不合格品自动报警
- 多种报警类型
- 可配置报警参数
- 支持硬件设备集成

### 9. DWG自动测量（V3.1新增）
- 支持DWG/DXF图纸导入
- 自动解析尺寸标注
- 智能图像配准
- 基于图纸的自动测量
- 生成详细测量报告
- 图形化操作界面
- 命令行和代码集成支持

---

## 开发指南

### 代码规范

- 遵循PEP 8规范
- 使用类型注解
- 编写文档字符串
- 异常处理完善
- 日志记录详细

详见 `CODE_STYLE.md`

### 项目结构

```
微小零件低精度视觉检测/
├── 核心模块/
│   ├── inspection_gui_enhanced.py   # 主GUI
│   ├── batch_inspection.py          # 批量检测
│   ├── defect_detection.py          # 缺陷检测
│   ├── inspection_system.py         # 检测系统
│   ├── core_detection.py            # 检测算法
│   ├── camera_manager.py            # 相机管理
│   ├── calibration_manager.py       # 标定管理
│   ├── config_manager.py            # 配置管理
│   ├── data_export.py               # 数据导出
│   ├── drawing_annotation.py        # 图纸标注
│   ├── dxf_parser.py                # DXF解析
│   ├── dwg_converter.py            # DWG转换（V3.1新增）
│   ├── image_registration.py       # 图像配准（V3.1新增）
│   ├── auto_measurement.py          # 自动测量（V3.1新增）
│   ├── exceptions.py                # 异常定义
│   └── logger_config.py             # 日志配置
├── GUI工具/
│   ├── calibration_gui.py           # 相机标定
│   ├── pixel_mm_calibration_gui.py  # 像素标定
│   ├── camera_preview_gui.py        # 实时预览
│   ├── result_visualization_gui.py  # 结果可视化
│   ├── config_editor_gui.py         # 配置管理
│   ├── inspection_system_gui_v2.py  # 图纸标注
│   └── dwg_auto_measurement_gui.py  # DWG自动测量（V3.1新增）
├── Web版本/
│   └── inspection_web_v2.py         # Web应用
├── 文档/
│   ├── 产品需求规格说明书.md
│   ├── 图纸检测部位标定方案.md
│   ├── 产品进化接口设计方案.md
│   ├── 亚像素IT5级精度检测分析.md
│   ├── DWG_AUTO_MEASUREMENT_GUIDE.md  # DWG自动测量指南（V3.1新增）
│   ├── README_V3.md
│   ├── USAGE_GUIDE.md
│   ├── GUI_MODULES_USAGE.md
│   ├── CODE_STYLE.md
│   ├── INSTALL.md
│   └── DOCKER_DEPLOY.md
├── 测试/
│   ├── tests/
│   │   ├── test_config_manager.py
│   │   ├── test_core_detection.py
│   │   ├── test_data_export.py
│   │   ├── test_exceptions.py
│   │   ├── test_auto_measurement.py  # DWG自动测量测试（V3.1新增）
│   │   └── __init__.py
│   ├── test_new_features.py
│   └── run_tests.py
├── 配置/
│   ├── requirements.txt
│   ├── config.json
│   ├── install_dependencies.bat
│   ├── docker-compose.yml
│   └── Dockerfile
├── 数据/
│   ├── data/
│   ├── templates/
│   └── logs/
├── 部署/
│   ├── nginx.conf
│   └── DOCKER_DEPLOY.md
└── 参考程序/
    ├── 11.detectOD/
    └── 7.chiCunJianCe.demo/
```

### 测试

```bash
# 运行所有单元测试（推荐）
python run_tests.py

# 运行特定测试文件
python run_tests.py test_config_manager
python run_tests.py test_core_detection
python run_tests.py test_data_export
python run_tests.py test_exceptions

# 测试新功能
python test_new_features.py

# 或使用pytest（如果已安装）
python -m pytest tests/ -v
python -m pytest tests/ --cov=. --cov-report=html
```

**测试文件列表：**
- `test_config_manager.py` - 配置管理器测试
- `test_core_detection.py` - 核心检测算法测试
- `test_data_export.py` - 数据导出测试
- `test_exceptions.py` - 异常处理测试
- `test_auto_measurement.py` - DWG自动测量功能测试（V3.1新增）

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

### Q5: PyQT5窗口无法打开？

**A**: 确保已安装PyQt5：
```bash
pip install PyQt5
```

### Q6: Windows环境下运行测试报错？

**A**: 确保已安装pytest和相关依赖：
```bash
pip install pytest pytest-cov coverage
```

或者使用项目提供的测试运行脚本：
```bash
python run_tests.py
```

### Q7: DWG文件无法转换？

**A**: DWG转换需要ODA File Converter：
- 下载ODA File Converter（免费工具）
- 安装后系统会自动检测
- 或者手动将DWG转换为DXF后使用

### Q8: 图像配准失败？

**A**: 检查以下内容：
- 确保图纸和图像包含足够的特征点
- 检查光照条件是否一致
- 尝试不同的特征检测算法（ORB/SIFT/SURF/AKAZE）
- 确保图纸和图像的拍摄角度相似

### Q9: DWG自动测量测量结果不准确？

**A**: 优化测量精度：
- 进行像素-毫米标定
- 使用高质量的图纸和图像
- 确保图纸标注准确
- 检查配准置信度是否足够高（建议>0.7）
- 使用亚像素检测提高精度

---

## 扩展点

系统设计了以下扩展点，用于未来的功能扩展：

| 扩展点 | 接口 | 用途 |
|--------|------|------|
| 图像采集 | `IImageAcquisition` | 支持不同类型的相机 |
| 图像预处理 | `IImagePreprocessor` | 添加新的图像处理算法 |
| 检测算法 | `IDetectionAlgorithm` | 添加新的检测算法 |
| 数据存储 | `IDataStorage` | 支持不同的存储方式 |
| 插件系统 | `IPlugin` | 添加自定义功能模块 |
| 特征类型 | `FeatureType`枚举 | 支持新的特征类型 |
| 配置提供者 | `IConfigurationProvider` | 支持不同的配置来源 |
| 亚像素算法 | `ISubpixelDetector` | 支持不同的亚像素检测算法 |

**精度扩展路径：**
- 当前：IT8级（1/20像素，使用OpenCV cornerSubPix）
- 未来：IT7级（1/30像素，使用高级亚像素算法）
- 远期：IT5级（1/50像素，使用Zernike矩算法，需要更高硬件配置）

---

## 未来扩展方向

1. **精度提升：**
   - 从IT8级升级到IT7级（1/30像素，成本增加2-3倍）
   - 从IT7级升级到IT5级（Zernike矩算法，1/50像素，成本增加10-50倍）

2. **深度学习集成：** 添加基于深度学习的检测算法

3. **3D检测增强：** 完善3D视觉检测功能

4. **云端集成：** 支持云端数据存储和分析

5. **多语言支持：** 扩展多语言界面支持

6. **移动端支持：** 开发移动端监控应用

7. **AI优化：** 使用AI进行检测算法自动优化

8. **工业物联网：** 集成IIoT平台

9. **亚像素算法优化：** 开发更高级的亚像素算法（如Zernike矩）

10. **混合检测方案：** 针对不同精度要求采用不同检测方法

---

## 参考标准

- GB/T 1800.1-2009 产品几何技术规范（GPS）极限与配合
- ISO 2768-1:1989 一般公差 未注公差的线性和角度尺寸的公差
- OpenCV官方文档 https://docs.opencv.org/

---

## 文档维护

**文档创建：** 2026年2月9日

**最后更新：** 2026年2月22日

**维护周期：** 根据项目进展定期更新

**版本控制：** 使用Git进行文档版本管理（当前仓库: https://github.com/jxjk/openCVDemo.git）

**更新记录：**
- V1.0 (2026-02-09): 初始版本，IT11级精度目标
- V2.0 (2026-02-09): 升级至IT8级精度目标，添加亚像素检测支持
- V3.0 (2026-02-10): 添加IT5级精度检测分析文档，更新参考程序说明，完善技术栈描述
- V4.0 (2026-02-10): 更新为V3.0生产可用版本，添加批量检测、缺陷检测、声光报警等功能，完善GUI工具集
- V4.1 (2026-02-22): 更新文档信息，确认当前项目状态和测试运行方式
- V5.0 (2026-02-22): 添加DWG自动测量功能，支持基于图纸的自动检测和测量

---

## 更新日志

### V4.0 (2026-02-10)

**重大更新：**
- 项目升级为V3.0生产可用版本
- 添加批量检测模块，支持≥60件/分钟检测速度
- 添加缺陷检测模块（表面、边缘、毛刺）
- 添加声光报警功能
- 添加历史数据查询功能
- 添加统计报表和图表可视化
- 完善GUI工具集（6个PyQt5工具）
- 集成所有功能到增强版GUI

**新增模块：**
- `batch_inspection.py` - 批量检测引擎
- `defect_detection.py` - 缺陷检测模块
- `calibration_gui.py` - 相机标定GUI
- `pixel_mm_calibration_gui.py` - 像素-毫米标定GUI
- `camera_preview_gui.py` - 实时预览GUI
- `result_visualization_gui.py` - 结果可视化GUI
- `config_editor_gui.py` - 配置管理GUI

**改进：**
- 优化多线程处理性能
- 改进异常处理机制
- 增强日志系统
- 完善代码文档
- 添加单元测试
- 添加使用指南

**修复：**
- 修复批量检测内存泄漏问题
- 修复缺陷检测误报问题
- 修复配置文件JSON序列化问题
- 修复凸包索引非单调错误
- 修复统计数据计算失败问题

### V4.1 (2026-02-22)

**文档更新：**
- 更新项目文档，反映当前状态
- 添加Git仓库信息
- 完善测试运行说明，添加run_tests.py脚本使用说明
- 更新依赖安装方式，添加Windows安装脚本
- 补充常见问题解答
- 添加归档目录说明
- 更新最后更新日期

### V5.0 (2026-02-22)

**重大更新：DWG自动测量功能**

**新增核心模块：**
- `dwg_converter.py` - DWG文件转换模块，支持DWG转DXF
- `image_registration.py` - 图像配准模块，支持多种特征检测和变换方法
- `auto_measurement.py` - 自动测量引擎，基于DWG/DXF图纸自动测量

**新增GUI工具：**
- `dwg_auto_measurement_gui.py` - DWG自动测量GUI，集成图纸解析、图像配准和自动测量功能

**新增功能：**
- DWG文件自动转换为DXF（可选，需要ODA File Converter）
- DXF图纸自动解析和标注提取
- 图纸渲染为图像
- 图像配准（ORB/SIFT/SURF/AKAZE特征检测）
- 基于图纸的自动测量
- 测量报告生成（JSON + 文本格式）

**新增文档：**
- `DWG_AUTO_MEASUREMENT_GUIDE.md` - DWG自动测量功能使用指南

**新增测试：**
- `tests/test_auto_measurement.py` - DWG自动测量功能单元测试

**技术特性：**
- 支持多种特征检测算法（ORB、SIFT、SURF、AKAZE）
- 支持多种变换方法（单应性、仿射、相似、平移）
- 自动提取尺寸标注信息
- 支持圆形和线段测量
- 自动计算偏差和合格性判断
- 详细的测量结果报告

**使用方式：**
- GUI界面：4步操作流程（加载图纸 → 加载图像 → 图像配准 → 自动测量）
- 命令行：支持直接命令调用
- 代码集成：提供完整的Python API

**改进：**
- 修复matplotlib API兼容性问题
- 修复ezdxf API兼容性问题
- 完善异常处理机制
- 增强日志系统

**修复：**
- 修复DWG边界获取错误
- 修复DXF渲染颜色范围错误
- 修复Line对象属性访问错误
- 修复FigureCanvasAgg API兼容性问题

---

## 联系信息

如有疑问或需要补充文档内容，请联系项目负责人。

---

**文档结束**