# 依赖安装指南

## 快速安装

### Windows用户

双击运行 `install_dependencies.bat` 脚本，它会自动安装所有必需的Python库。

### 手动安装

如果自动脚本失败，可以手动执行以下命令：

```bash
# 升级pip
python -m pip install --upgrade pip

# 安装核心依赖
pip install flask>=2.0.0
pip install flask-socketio>=5.0.0
pip install flask-cors>=3.0.0
pip install eventlet>=0.30.0

# 安装图像处理库
pip install opencv-python>=4.5.0
pip install numpy>=1.19.0
pip install Pillow>=8.0.0

# 安装数据处理库
pip install pandas>=1.2.0
pip install openpyxl>=3.0.0

# 安装DXF解析库（必需）
pip install ezdxf>=1.0.0

# 安装可选依赖
pip install scipy>=1.5.0
pip install matplotlib>=3.3.0
pip install tqdm>=4.50.0
```

### 使用国内镜像源（推荐）

如果下载速度慢，使用清华镜像源：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

## 依赖库说明

### 必需库

| 库名 | 版本 | 用途 | DXF功能必需 |
|------|------|------|-------------|
| flask | >=2.0.0 | Web框架 | 否 |
| flask-socketio | >=5.0.0 | WebSocket支持 | 否 |
| flask-cors | >=3.0.0 | 跨域支持 | 否 |
| eventlet | >=0.30.0 | 异步支持 | 否 |
| opencv-python | >=4.5.0 | 图像处理 | 否 |
| numpy | >=1.19.0 | 数值计算 | 否 |
| pandas | >=1.2.0 | 数据处理 | 否 |
| Pillow | >=8.0.0 | 图像IO | 否 |
| openpyxl | >=3.0.0 | Excel支持 | 否 |
| **ezdxf** | **>=1.0.0** | **DXF文件解析** | **是** |

### 可选库

| 库名 | 版本 | 用途 |
|------|------|------|
| scipy | >=1.5.0 | 科学计算 |
| matplotlib | >=3.3.0 | 绘图可视化 |
| tqdm | >=4.50.0 | 进度条 |
| svgpathtools | >=1.5.0 | SVG文件解析 |
| gxipy | >=1.0.0 | 大恒相机SDK |

## DXF文件导入问题

如果您遇到DXF文件无法导入的问题，请检查：

1. **ezdxf库是否安装**
   ```bash
   python -c "import ezdxf; print(ezdxf.__version__)"
   ```
   如果报错，说明未安装，请执行：
   ```bash
   pip install ezdxf>=1.0.0
   ```

2. **DXF文件格式**
   - 支持DXF R12及以上版本（AC1009-AC1032）
   - 文件必须包含ENTITIES部分
   - 文件编码建议使用UTF-8或ANSI

3. **DXF文件内容**
   - 确保文件包含几何实体（CIRCLE、LINE、ARC等）
   - 尺寸标注（DIMENSION）可选，但有助于自动提取

## 验证安装

运行测试脚本验证所有库是否正确安装：

```bash
python test_dxf_parser.py
```

如果所有测试通过，说明依赖安装成功。

## 常见问题

### Q1: pip安装速度慢或失败

**A:** 使用国内镜像源：
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ezdxf
```

### Q2: ezdxf安装后仍然无法导入

**A:** 检查Python版本，ezdxf需要Python 3.6+
```bash
python --version
```

### Q3: OpenCV安装失败

**A:** 尝试安装无GUI版本：
```bash
pip install opencv-python-headless
```

### Q4: DXF文件加载报错

**A:** 检查DXF文件是否损坏：
```bash
python test_dxf_parser.py
```

## 联系支持

如果遇到其他问题，请检查：
1. Python版本是否 >= 3.6
2. pip版本是否为最新
3. 网络连接是否正常
4. 是否有防火墙或代理限制