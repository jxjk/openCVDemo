# 微小零件中高精度视觉检测系统 - Docker镜像
# 基于Python 3.9构建

FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libfontconfig1 \
    libcairo2 \
    libpango-1.0-0 \
    libatk1.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用程序文件
COPY inspection_system.py .
COPY inspection_system_gui.py .
COPY inspection_web.py .
COPY README.md .

# 创建数据目录
RUN mkdir -p /app/data/images

# 设置权限
RUN chmod -R 755 /app/data

# 暴露端口
EXPOSE 5008

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5008/health || exit 1

# 启动命令
CMD ["python", "inspection_web.py", "--host", "0.0.0.0", "--port", "5008"]