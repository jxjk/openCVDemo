# 代码规范文档
Code Style Guide

## 1. Python代码风格

### 1.1 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 类名 | 大驼峰 (PascalCase) | `CameraManager`, `InspectionEngine` |
| 函数名 | 小写+下划线 (snake_case) | `get_camera()`, `calculate_statistics()` |
| 变量名 | 小写+下划线 (snake_case) | `pixel_to_mm`, `image_data` |
| 常量名 | 大写+下划线 (UPPER_SNAKE_CASE) | `IT8_TOLERANCE`, `MAX_RADIUS` |
| 私有成员 | 单下划线前缀 | `_instance`, `_config` |
| 模块名 | 小写+下划线 | `camera_manager.py`, `data_export.py` |

### 1.2 导入顺序

```python
# 1. 标准库导入
import os
import sys
import json

# 2. 第三方库导入
import cv2
import numpy as np
from flask import Flask, jsonify

# 3. 本地模块导入
from exceptions import CameraException
from logger_config import get_logger
from config_manager import get_config
```

### 1.3 文档字符串

使用Google风格的文档字符串：

```python
def detect_circles(image: np.ndarray, 
                   min_radius: int = 5,
                   max_radius: int = 100) -> List[DetectionResult]:
    """
    检测圆形
    
    使用Hough变换检测图像中的圆形
    
    Args:
        image: 输入图像（灰度图）
        min_radius: 最小半径（像素）
        max_radius: 最大半径（像素）
    
    Returns:
        检测结果列表
    
    Raises:
        ImageProcessingException: 图像处理失败时
    
    Example:
        >>> results = detector.detect_circles(gray_image)
        >>> for result in results:
        ...     print(f"圆形: 中心={result.center}, 半径={result.radius}")
    """
    # 实现代码
```

### 1.4 类型注解

为函数参数和返回值添加类型注解：

```python
from typing import List, Optional, Dict, Tuple

def process_image(image: np.ndarray,
                  threshold: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """处理图像并返回结果"""
    # 实现代码
```

## 2. 异常处理

### 2.1 异常层次结构

使用项目定义的异常层次结构：

```python
from exceptions import CameraException, CameraConnectionError, ImageProcessingException

try:
    camera.connect()
except CameraConnectionError as e:
    logger.error(f"相机连接失败: {e}")
    # 处理连接错误
except CameraException as e:
    logger.error(f"相机错误: {e}")
    # 处理其他相机错误
```

### 2.2 不要忽略异常

```python
# ❌ 错误：忽略异常
try:
    result = process_image(image)
except:
    pass

# ✅ 正确：记录异常
try:
    result = process_image(image)
except CameraException as e:
    logger.error(f"处理失败: {e}")
    raise
```

## 3. 日志规范

### 3.1 日志级别使用

| 级别 | 使用场景 | 示例 |
|------|----------|------|
| DEBUG | 调试信息，详细的执行流程 | `logger.debug(f"亚像素细化: {point} -> {refined}")` |
| INFO | 重要操作信息 | `logger.info(f"相机连接成功: {device_id}")` |
| WARNING | 警告信息，不影响运行 | `logger.warning("未检测到圆形")` |
| ERROR | 错误信息，需要处理 | `logger.error(f"采集图像失败: {e}")` |
| CRITICAL | 严重错误，系统无法继续 | `logger.critical("系统初始化失败")` |

### 3.2 日志格式

```python
from logger_config import get_logger

logger = get_logger(__name__)

logger.info("操作开始")
logger.debug(f"详细参数: {params}")
logger.error("操作失败", exc_info=True)  # 包含堆栈信息
```

## 4. 配置管理

### 4.1 使用配置管理器

```python
from config_manager import get_config, get_tolerance

config = get_config()

# 获取配置值
pixel_to_mm = config.pixel_to_mm

# 获取公差
tolerance = get_tolerance(5.0, "IT8")
```

### 4.2 不要硬编码配置

```python
# ❌ 错误：硬编码
if measured_value > 0.018:
    return False

# ✅ 正确：使用配置
tolerance = get_tolerance(nominal_value)
if abs(measured_value - nominal_value) > tolerance:
    return False
```

## 5. 代码组织

### 5.1 模块结构

```
project/
├── exceptions.py          # 异常定义
├── logger_config.py       # 日志配置
├── config_manager.py      # 配置管理
├── camera_manager.py      # 相机管理
├── core_detection.py      # 核心检测
├── data_export.py         # 数据导出
├── calibration_manager.py # 标定管理
├── drawing_annotation.py  # 图纸标注
├── dxf_parser.py          # DXF解析
├── inspection_system.py   # 核心系统
├── inspection_gui_v2.py   # GUI版本
├── inspection_web_v2.py   # Web版本
└── tests/                 # 测试目录
    ├── __init__.py
    ├── test_exceptions.py
    ├── test_config_manager.py
    ├── test_core_detection.py
    └── test_data_export.py
```

### 5.2 类设计原则

- **单一职责原则**: 每个类只负责一个功能
- **开闭原则**: 对扩展开放，对修改关闭
- **依赖倒置原则**: 依赖抽象接口，不依赖具体实现

## 6. 性能优化

### 6.1 使用NumPy向量化操作

```python
# ❌ 低效：循环
result = []
for i in range(len(array)):
    result.append(array[i] * 2)

# ✅ 高效：向量化
result = array * 2
```

### 6.2 避免不必要的拷贝

```python
# ❌ 创建新数组
new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
processed = new_image.copy()

# ✅ 直接使用
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

## 7. 测试规范

### 7.1 测试命名

```python
class TestCircleDetector(unittest.TestCase):
    
    def test_detect_circles_with_single_circle(self):
        """测试单个圆形检测"""
        pass
    
    def test_detect_circles_with_multiple_circles(self):
        """测试多个圆形检测"""
        pass
    
    def test_detect_circles_with_no_circles(self):
        """测试无圆形情况"""
        pass
```

### 7.2 测试覆盖率

- 核心模块覆盖率 ≥ 80%
- 工具类覆盖率 ≥ 90%
- GUI/Web模块覆盖率 ≥ 50%

## 8. Git提交规范

### 8.1 提交信息格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### 8.2 类型说明

| 类型 | 说明 | 示例 |
|------|------|------|
| feat | 新功能 | `feat(detection): 添加圆形检测算法` |
| fix | 修复bug | `fix(camera): 修复相机连接超时问题` |
| docs | 文档更新 | `docs(readme): 更新安装说明` |
| style | 代码格式 | `style(config): 统一代码格式` |
| refactor | 重构 | `refactor(core): 重构检测引擎` |
| test | 测试 | `test(detection): 添加圆形检测测试` |
| chore | 构建/工具 | `chore(deps): 更新依赖库` |

## 9. 安全规范

### 9.1 输入验证

```python
def process_image(image: np.ndarray) -> np.ndarray:
    """处理图像"""
    if image is None:
        raise ValueError("图像不能为空")
    
    if len(image.shape) not in [2, 3]:
        raise ValueError("图像格式不正确")
    
    # 处理逻辑
```

### 9.2 路径安全

```python
import os

def load_config(filepath: str) -> dict:
    """加载配置文件"""
    # 验证路径
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    # 限制在项目目录内
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.abspath(filepath).startswith(project_dir):
        raise ValueError("不允许访问项目目录外的文件")
    
    # 加载配置
```

## 10. 代码审查清单

提交代码前检查：

- [ ] 代码符合PEP 8规范
- [ ] 所有函数都有文档字符串
- [ ] 异常都被正确处理和记录
- [ ] 没有硬编码的配置值
- [ ] 日志级别使用正确
- [ ] 没有明显的性能问题
- [ ] 没有安全问题（路径遍历、SQL注入等）
- [ ] 添加了必要的测试
- [ ] 测试通过
- [ ] 更新了相关文档

## 11. IDE配置建议

### 11.1 VSCode

```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "editor.formatOnSave": true
}
```

### 11.2 PyCharm

- 启用PEP 8检查
- 配置代码格式化（Black）
- 配置pytest测试框架
- 启用类型检查（mypy）

## 12. 持续集成

建议使用GitHub Actions或GitLab CI进行：

- 代码风格检查
- 单元测试
- 测试覆盖率报告
- 安全扫描

---

**文档版本**: V1.0  
**最后更新**: 2026-02-10  
**维护者**: 项目团队