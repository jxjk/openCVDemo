# Docker部署指南

## 快速启动

### 1. 构建并启动容器

```bash
# 使用docker-compose启动
docker-compose up -d

# 或者手动构建和运行
docker build -t inspection-system .
docker run -d -p 5008:5008 --name inspection-system inspection-system
```

### 2. 访问Web界面

在浏览器中打开：
```
http://localhost:5008
```

## 详细部署步骤

### 前置要求

确保已安装以下软件：

1. **Docker** (版本 20.10+)
2. **Docker Compose** (版本 1.29+)

检查安装：
```bash
docker --version
docker-compose --version
```

### 部署步骤

#### 方式一：使用Docker Compose（推荐）

1. **克隆或下载项目文件**
   ```bash
   cd 微小零件低精度视觉检测
   ```

2. **构建并启动服务**
   ```bash
   docker-compose up -d
   ```

3. **查看日志**
   ```bash
   docker-compose logs -f inspection-system
   ```

4. **停止服务**
   ```bash
   docker-compose down
   ```

#### 方式二：使用Docker命令

1. **构建镜像**
   ```bash
   docker build -t inspection-system:latest .
   ```

2. **运行容器**
   ```bash
   docker run -d \
     --name inspection-system \
     -p 5008:5008 \
     -v $(pwd)/data:/app/data \
     --privileged \
     inspection-system:latest
   ```

3. **查看日志**
   ```bash
   docker logs -f inspection-system
   ```

4. **停止容器**
   ```bash
   docker stop inspection-system
   docker rm inspection-system
   ```

## 配置说明

### 端口配置

默认端口：`5008`

如需修改端口，编辑 `docker-compose.yml`：
```yaml
ports:
  - "5008:5008"  # 修改为 "其他端口:5008"
```

### 数据持久化

数据目录挂载到宿主机：
```yaml
volumes:
  - ./data:/app/data
```

数据目录结构：
```
data/
├── calibration_data.json  # 相机标定数据
├── config.json            # 系统配置
├── inspection_results.csv # 检测结果
├── inspection_errors.csv  # 错误记录
└── images/                # 检测图像
```

### 相机设备访问

如果需要访问USB相机，需要挂载设备：
```yaml
volumes:
  - /dev/video0:/dev/video0
```

或者在运行时添加：
```bash
docker run -d \
  --device=/dev/video0 \
  ...
```

## Web界面功能

### 主要功能

1. **相机控制**
   - 连接/断开相机
   - 实时预览
   - 支持USB相机和大恒相机

2. **零件检测**
   - 圆形零件直径测量
   - 矩形零件长宽测量
   - IT8级公差判断

3. **系统标定**
   - 像素-毫米转换标定
   - 自动计算转换系数

4. **数据管理**
   - 检测结果实时显示
   - 统计信息查看
   - 数据导出到Excel

### API接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web界面主页 |
| `/api/status` | GET | 获取系统状态 |
| `/api/camera/connect` | POST | 连接相机 |
| `/api/camera/disconnect` | POST | 断开相机 |
| `/api/camera/start_preview` | POST | 开始预览 |
| `/api/camera/stop_preview` | POST | 停止预览 |
| `/api/inspect` | POST | 检测零件 |
| `/api/calibrate` | POST | 标定系统 |
| `/api/results` | GET | 获取检测结果 |
| `/api/results/export` | POST | 导出结果 |
| `/health` | GET | 健康检查 |

## 故障排查

### 问题1：容器无法启动

**检查日志：**
```bash
docker logs inspection-system
```

**常见原因：**
- 端口5008已被占用
- Docker版本过低
- 依赖库安装失败

**解决方案：**
```bash
# 检查端口占用
netstat -ano | findstr :5008

# 停止占用端口的进程或修改端口
```

### 问题2：无法访问相机

**检查相机设备：**
```bash
# 在宿主机上查看相机设备
ls -la /dev/video*

# 在容器内查看相机设备
docker exec inspection-system ls -la /dev/
```

**解决方案：**
```bash
# 确保容器有权限访问设备
docker run -d \
  --device=/dev/video0 \
  --privileged \
  ...
```

### 问题3：Web界面无法访问

**检查容器状态：**
```bash
docker ps
docker logs inspection-system
```

**检查网络连接：**
```bash
# 测试容器内部网络
docker exec inspection-system curl http://localhost:5008/health
```

**解决方案：**
- 确保容器正在运行
- 检查端口映射是否正确
- 检查防火墙设置

### 问题4：预览卡顿或延迟

**优化建议：**
1. 降低图像分辨率
2. 减少传输帧率
3. 优化网络连接

**调整参数：**
编辑 `inspection_web.py` 中的预览参数：
```python
time.sleep(0.05)  # 调整帧率
max_size = 800    # 调整图像大小
```

## 生产环境部署

### 使用Nginx反向代理

1. **创建Nginx配置文件** `nginx.conf`：
```nginx
events {
    worker_connections 1024;
}

http {
    upstream inspection_backend {
        server inspection-system:5008;
    }

    server {
        listen 80;
        server_name your-domain.com;

        location / {
            proxy_pass http://inspection_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # WebSocket支持
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

2. **更新docker-compose.yml**：
```yaml
nginx:
  image: nginx:alpine
  ports:
    - "80:80"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf:ro
  depends_on:
    - inspection-system
```

3. **启动服务**：
```bash
docker-compose up -d
```

### 使用HTTPS

1. **准备SSL证书**
   ```bash
   mkdir ssl
   # 将证书文件复制到ssl目录
   ```

2. **更新Nginx配置**：
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    location / {
        proxy_pass http://inspection_backend;
        # ... 其他配置
    }
}
```

3. **挂载证书目录**：
```yaml
volumes:
  - ./ssl:/etc/nginx/ssl:ro
```

## 性能优化

### 资源限制

在 `docker-compose.yml` 中添加资源限制：
```yaml
services:
  inspection-system:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### 日志管理

限制日志大小：
```yaml
services:
  inspection-system:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## 备份与恢复

### 备份数据

```bash
# 备份数据目录
tar -czf data_backup_$(date +%Y%m%d).tar.gz data/

# 备份Docker镜像
docker save inspection-system:latest | gzip > inspection_system_$(date +%Y%m%d).tar.gz
```

### 恢复数据

```bash
# 恢复数据目录
tar -xzf data_backup_20260210.tar.gz

# 恢复Docker镜像
docker load < inspection_system_20260210.tar.gz
```

## 监控与维护

### 查看容器状态

```bash
# 查看运行状态
docker ps

# 查看资源使用
docker stats inspection-system

# 查看日志
docker logs -f inspection-system
```

### 定期维护

```bash
# 清理未使用的镜像
docker image prune -a

# 清理未使用的容器
docker container prune

# 清理未使用的卷
docker volume prune
```

## 安全建议

1. **使用非root用户运行容器**
2. **限制容器权限**
3. **定期更新镜像**
4. **使用HTTPS加密传输**
5. **配置防火墙规则**
6. **定期备份数据**

## 技术支持

如遇到问题，请检查：
1. Docker日志
2. 容器状态
3. 网络连接
4. 相机设备权限

---

**最后更新：** 2026-02-10  
**版本：** V1.0