@echo off
REM 微小零件视觉检测系统 - Python依赖安装脚本
REM Windows版本

echo ============================================================
echo 微小零件视觉检测系统 - Python依赖安装
echo ============================================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到Python，请先安装Python 3.8或更高版本
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [信息] 检测到Python环境
python --version
echo.

REM 检查pip是否可用
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] pip不可用，请检查Python安装
    pause
    exit /b 1
)

echo [信息] 升级pip到最新版本...
python -m pip install --upgrade pip
echo.

echo ============================================================
echo [步骤 1] 安装核心依赖库
echo ============================================================
echo.

echo [1/10] 安装 Flask (Web框架)...
python -m pip install flask>=2.0.0
if %errorlevel% neq 0 (
    echo [警告] Flask安装失败
)

echo [2/10] 安装 Flask-SocketIO (WebSocket支持)...
python -m pip install flask-socketio>=5.0.0
if %errorlevel% neq 0 (
    echo [警告] Flask-SocketIO安装失败
)

echo [3/10] 安装 Flask-CORS (跨域支持)...
python -m pip install flask-cors>=3.0.0
if %errorlevel% neq 0 (
    echo [警告] Flask-CORS安装失败
)

echo [4/10] 安装 eventlet (异步支持)...
python -m pip install eventlet>=0.30.0
if %errorlevel% neq 0 (
    echo [警告] eventlet安装失败
)

echo.
echo ============================================================
echo [步骤 2] 安装图像处理库
echo ============================================================
echo.

echo [5/10] 安装 OpenCV (图像处理)...
python -m pip install opencv-python>=4.5.0
if %errorlevel% neq 0 (
    echo [警告] OpenCV安装失败
)

echo [6/10] 安装 NumPy (数值计算)...
python -m pip install numpy>=1.19.0
if %errorlevel% neq 0 (
    echo [警告] NumPy安装失败
)

echo [7/10] 安装 Pillow (图像IO)...
python -m pip install Pillow>=8.0.0
if %errorlevel% neq 0 (
    echo [警告] Pillow安装失败
)

echo.
echo ============================================================
echo [步骤 3] 安装数据处理库
echo ============================================================
echo.

echo [8/10] 安装 Pandas (数据处理)...
python -m pip install pandas>=1.2.0
if %errorlevel% neq 0 (
    echo [警告] Pandas安装失败
)

echo [9/10] 安装 openpyxl (Excel支持)...
python -m pip install openpyxl>=3.0.0
if %errorlevel% neq 0 (
    echo [警告] openpyxl安装失败
)

echo.
echo ============================================================
echo [步骤 4] 安装DXF解析库（图纸标注功能必需）
echo ============================================================
echo.

echo [10/10] 安装 ezdxf (DXF文件解析)...
python -m pip install ezdxf>=1.0.0
if %errorlevel% neq 0 (
    echo [警告] ezdxf安装失败，DXF文件导入功能将不可用
)

echo.
echo ============================================================
echo [步骤 5] 安装可选依赖库
echo ============================================================
echo.

echo [可选] 安装 SciPy (科学计算)...
python -m pip install scipy>=1.5.0

echo [可选] 安装 Matplotlib (绘图)...
python -m pip install matplotlib>=3.3.0

echo [可选] 安装 tqdm (进度条)...
python -m pip install tqdm>=4.50.0

echo.
echo ============================================================
echo 安装完成！
echo ============================================================
echo.

REM 验证安装
echo [验证] 检查已安装的库...
python -c "import flask; print('  ✓ Flask:', flask.__version__)"
python -c "import cv2; print('  ✓ OpenCV:', cv2.__version__)"
python -c "import numpy; print('  ✓ NumPy:', numpy.__version__)"
python -c "import pandas; print('  ✓ Pandas:', pandas.__version__)"
python -c "import PIL; print('  ✓ Pillow:', PIL.__version__)"
python -c "import openpyxl; print('  ✓ openpyxl:', openpyxl.__version__)"
python -c "import ezdxf; print('  ✓ ezdxf:', ezdxf.__version__)" 2>nul || echo "  ✗ ezdxf 未安装"

echo.
echo ============================================================
echo 提示:
echo 1. 如果某些库安装失败，请检查网络连接或使用国内镜像源
echo 2. 使用国内镜像源: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package_name
echo 3. 大恒相机SDK(gxipy)需要先安装相机驱动程序
echo ============================================================
echo.

pause