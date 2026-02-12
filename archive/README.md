# 归档说明

此目录包含已归档的旧版本文件和测试文件。

## 目录结构

```
archive/
├── old_versions/          # 旧版本文件
│   ├── inspection_system_gui.py    # V1.0 GUI版本
│   └── inspection_web.py          # V1.0 Web版本
├── test_files/             # 测试文件
│   ├── test_dxf_parser.py         # DXF解析测试
│   ├── diagnose_gui_dxf.py        # GUI诊断脚本
│   └── test_gui_dxf_import.py     # GUI导入测试
└── =1.0.0                   # 意外创建的文件
```

## 归档文件说明

### 旧版本文件 (old_versions/)

#### inspection_system_gui.py
- **版本**: V1.0
- **描述**: 原始GUI版本，不支持图纸标注功能
- **被替代**: inspection_system_gui_v2.py
- **归档日期**: 2026-02-10

#### inspection_web.py
- **版本**: V1.0
- **描述**: 原始Web版本，不支持图纸标注功能
- **被替代**: inspection_web_v2.py
- **归档日期**: 2026-02-10

### 测试文件 (test_files/)

#### test_dxf_parser.py
- **描述**: DXF文件解析测试脚本
- **用途**: 测试DXF文件解析功能
- **创建日期**: 2026-02-10

#### diagnose_gui_dxf.py
- **描述**: GUI DXF导入问题诊断脚本
- **用途**: 诊断GUI无法加载DXF文件的问题
- **创建日期**: 2026-02-10

#### test_gui_dxf_import.py
- **描述**: GUI DXF导入模拟测试
- **用途**: 模拟GUI的DXF导入流程
- **创建日期**: 2026-02-10

## 当前版本文件

项目根目录中的当前使用文件：

### 核心系统文件
- `inspection_system.py` - 核心检测系统（主版本）
- `drawing_annotation.py` - 图纸标注模块
- `dxf_parser.py` - DXF解析模块

### V2.0 版本文件（支持图纸标注）
- `inspection_system_gui_v2.py` - GUI V2.0版本
- `inspection_web_v2.py` - Web V2.0版本

### 文档文件
- `AGENTS.md` - 项目上下文文档
- `README.md` - 项目说明文档
- `INSTALL.md` - 安装指南
- `requirements.txt` - Python依赖列表
- `产品需求规格说明书.md` - 产品需求文档
- `产品进化接口设计方案.md` - 接口设计文档
- `图纸检测部位标定方案.md` - 标定方案文档
- `亚像素IT5级精度检测分析.md` - IT5级精度分析文档

### Docker部署文件
- `Dockerfile` - Docker镜像构建文件
- `docker-compose.yml` - Docker Compose配置
- `nginx.conf` - Nginx配置文件
- `DOCKER_DEPLOY.md` - Docker部署文档

### 安装文件
- `install_dependencies.bat` - Windows依赖安装脚本

### 目录
- `data/` - 数据存储目录
- `templates/` - Web模板目录
- `参考程序/` - 参考程序目录
- `archive/` - 归档目录（本文件所在目录）

## 版本历史

### V1.0 (已归档)
- 基础检测功能
- 简单GUI界面
- 基础Web界面
- 不支持图纸标注

### V2.0 (当前版本)
- 完整的图纸标注功能
- DXF文件导入
- 基于标注的检测
- 增强的GUI界面
- 增强的Web界面

## 注意事项

1. **不要删除归档文件**：这些文件可能用于参考或回退
2. **测试文件**：可以重新运行测试脚本进行调试
3. **旧版本**：如果需要回退，可以从归档中恢复

## 恢复旧版本

如果需要恢复旧版本文件，可以执行：

```bash
# 恢复GUI V1.0
copy archive\old_versions\inspection_system_gui.py .\inspection_system_gui_v1.py

# 恢复Web V1.0
copy archive\old_versions\inspection_web.py .\inspection_web_v1.py
```

## 维护建议

- 定期清理测试文件
- 及时归档旧版本
- 更新此说明文档

---

**归档日期**: 2026-02-10  
**维护人员**: 项目开发团队