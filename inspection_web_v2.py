# -*- coding: utf-8 -*-
"""
Web服务器端点 V2版本 - 支持图纸标注
Web Server Endpoints V2 - Drawing Annotation Support

版本: V2.0
新增功能: 图纸标注、基于标注的检测
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import json
import os
import uuid
from datetime import datetime
from io import BytesIO
import base64

# 导入核心模块
from inspection_system import InspectionConfig, InspectionEngine, DataManager

# 导入图纸标注模块
from drawing_annotation import (
    InspectionTemplate,
    AnnotationTool,
    AnnotationBasedInspector,
    Point2D,
    BoundingBox,
    CircleAnnotation,
    LineAnnotation,
    RectangleAnnotation,
    AngleAnnotation,
    FeatureType,
    ToleranceStandard,
    create_default_template,
    load_template,
    save_template
)

# 导入DXF解析模块
from dxf_parser import (
    DXFParser,
    DXFToTemplateConverter,
    parse_dxf_file,
    dxf_to_template,
    extract_dimensions_from_dxf
)

# 初始化Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# 全局配置
config = InspectionConfig()

# 当前模板
current_template = create_default_template("默认模板")
annotation_tool = AnnotationTool(current_template)
inspector = AnnotationBasedInspector(current_template, config)


# =============================================================================
# 路由定义
# =============================================================================

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/drawing_annotation')
def drawing_annotation():
    """图纸标注页面"""
    return render_template('drawing_annotation.html')


@app.route('/health')
def health():
    """健康检查"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


# =============================================================================
# 相机控制API
# =============================================================================

@app.route('/api/camera/status', methods=['GET'])
def get_camera_status():
    """获取相机状态"""
    try:
        status = {
            'connected': False,
            'camera_type': None,
            'resolution': None
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/camera/connect', methods=['POST'])
def connect_camera():
    """连接相机"""
    try:
        data = request.json
        camera_type = data.get('camera_type', 'usb')
        device_id = data.get('device_id', '0')
        
        # TODO: 实现相机连接逻辑
        return jsonify({'success': True, 'message': '相机连接成功'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/camera/disconnect', methods=['POST'])
def disconnect_camera():
    """断开相机"""
    try:
        # TODO: 实现相机断开逻辑
        return jsonify({'success': True, 'message': '相机已断开'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/camera/capture', methods=['POST'])
def capture_image():
    """采集图像"""
    try:
        # TODO: 实现图像采集逻辑
        # 返回base64编码的图像
        return jsonify({
            'success': True,
            'image': None,  # base64编码的图像
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# 图纸标注API
# =============================================================================

@app.route('/api/template/create', methods=['POST'])
def create_template():
    """创建新模板"""
    try:
        data = request.json
        name = data.get('name', '新模板')
        
        global current_template, annotation_tool, inspector
        current_template = create_default_template(name)
        annotation_tool = AnnotationTool(current_template)
        inspector = AnnotationBasedInspector(current_template, config)
        
        return jsonify({
            'success': True,
            'template': current_template.to_dict()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/template/load', methods=['POST'])
def load_template_api():
    """加载模板"""
    try:
        data = request.json
        filepath = data.get('filepath')
        
        if not filepath:
            return jsonify({'error': '请提供模板文件路径'}), 400
        
        global current_template, annotation_tool, inspector
        current_template = load_template(filepath)
        annotation_tool = AnnotationTool(current_template)
        inspector = AnnotationBasedInspector(current_template, config)
        
        return jsonify({
            'success': True,
            'template': current_template.to_dict()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/template/save', methods=['POST'])
def save_template_api():
    """保存模板"""
    try:
        data = request.json
        filepath = data.get('filepath')
        
        if not filepath:
            return jsonify({'error': '请提供保存路径'}), 400
        
        save_template(current_template, filepath)
        
        return jsonify({
            'success': True,
            'message': '模板保存成功'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/template', methods=['GET'])
def get_template():
    """获取当前模板"""
    try:
        return jsonify({
            'success': True,
            'template': current_template.to_dict()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/template/update', methods=['POST'])
def update_template():
    """更新模板"""
    try:
        data = request.json
        
        # 更新模板基本信息
        if 'name' in data:
            current_template.name = data['name']
        if 'image_scale' in data:
            current_template.image_scale = data['image_scale']
        if 'tolerance_standard' in data:
            current_template.tolerance_standard = ToleranceStandard(data['tolerance_standard'])
        
        current_template._update_timestamp()
        
        return jsonify({
            'success': True,
            'template': current_template.to_dict()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# 标注管理API
# =============================================================================

@app.route('/api/annotations', methods=['GET'])
def get_annotations():
    """获取所有标注"""
    try:
        annotations = [anno.to_dict() for anno in current_template.annotations]
        return jsonify({
            'success': True,
            'annotations': annotations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/annotations/add', methods=['POST'])
def add_annotation():
    """添加标注"""
    try:
        data = request.json
        anno_type = data.get('type')
        
        if not anno_type:
            return jsonify({'error': '请提供标注类型'}), 400
        
        annotation = None
        
        if anno_type in ['diameter', 'radius']:
            center = Point2D(*data['center'])
            radius = data['radius']
            feature_type = FeatureType(anno_type)
            
            annotation = annotation_tool.add_circle(center, radius, feature_type)
            
        elif anno_type in ['length', 'width', 'height']:
            start = Point2D(*data['start'])
            end = Point2D(*data['end'])
            feature_type = FeatureType(anno_type)
            
            annotation = annotation_tool.add_line(start, end, feature_type)
            
        elif anno_type == 'angle':
            vertex = Point2D(*data['vertex'])
            start_point = Point2D(*data['start_point'])
            end_point = Point2D(*data['end_point'])
            
            annotation = annotation_tool.add_angle(vertex, start_point, end_point)
        
        # 设置标称值和公差
        if annotation and 'nominal_value' in data:
            tolerance = data.get('tolerance')
            annotation_tool.update_annotation_value(
                annotation.id,
                data['nominal_value'],
                tolerance
            )
        
        return jsonify({
            'success': True,
            'annotation': annotation.to_dict() if annotation else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/annotations/<annotation_id>', methods=['PUT'])
def update_annotation(annotation_id):
    """更新标注"""
    try:
        data = request.json
        
        annotation = current_template.get_annotation(annotation_id)
        if not annotation:
            return jsonify({'error': '标注不存在'}), 404
        
        # 更新参数
        if 'nominal_value' in data:
            tolerance = data.get('tolerance')
            annotation_tool.update_annotation_value(annotation_id, data['nominal_value'], tolerance)
        
        if 'description' in data:
            annotation.description = data['description']
        
        return jsonify({
            'success': True,
            'annotation': annotation.to_dict()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/annotations/<annotation_id>', methods=['DELETE'])
def delete_annotation(annotation_id):
    """删除标注"""
    try:
        annotation_tool.remove_annotation(annotation_id)
        
        return jsonify({
            'success': True,
            'message': '标注已删除'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# 检测API
# =============================================================================

@app.route('/api/inspect', methods=['POST'])
def inspect():
    """基于标注进行检测"""
    try:
        data = request.json
        
        # 获取图像
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': '请提供图像数据'}), 400
        
        # 解码图像
        if isinstance(image_data, str):
            # Base64编码
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            # 直接numpy数组
            image = np.array(image_data)
        
        # 执行检测
        part_id = data.get('part_id', '')
        results = inspector.inspect_from_annotations(image, part_id)
        
        # 绘制标注和结果
        result_image = inspector.draw_annotations(image)
        
        # 编码结果图像
        _, buffer = cv2.imencode('.jpg', result_image)
        result_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'results': results,
            'result_image': result_image_base64,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/inspect/batch', methods=['POST'])
def inspect_batch():
    """批量检测"""
    try:
        data = request.json
        images = data.get('images', [])
        part_id = data.get('part_id', '')
        
        all_results = []
        
        for i, image_data in enumerate(images):
            # 解码图像
            if isinstance(image_data, str):
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                image = np.array(image_data)
            
            # 执行检测
            results = inspector.inspect_from_annotations(image, f"{part_id}_{i}")
            all_results.extend(results)
        
        return jsonify({
            'success': True,
            'results': all_results,
            'total_count': len(all_results),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# 数据管理API
# =============================================================================

@app.route('/api/results/export', methods=['POST'])
def export_results():
    """导出检测结果"""
    try:
        data = request.json
        results = data.get('results', [])
        format_type = data.get('format', 'excel')
        
        # TODO: 实现数据导出逻辑
        return jsonify({
            'success': True,
            'message': '数据导出成功',
            'file_url': None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """获取统计信息"""
    try:
        # TODO: 实现统计逻辑
        stats = {
            'total_inspected': 0,
            'qualified': 0,
            'unqualified': 0,
            'qualified_rate': 0.0
        }
        
        return jsonify({
            'success': True,
            'statistics': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# DXF导入API
# =============================================================================

@app.route('/api/dxf/import', methods=['POST'])
def import_dxf():
    """导入DXF文件"""
    try:
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({'error': '请上传DXF文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '请选择文件'}), 400
        
        if not file.filename.lower().endswith('.dxf'):
            return jsonify({'error': '请上传DXF格式文件'}), 400
        
        # 获取导入选项
        data = request.form
        template_name = data.get('template_name', os.path.splitext(file.filename)[0])
        tolerance_standard = data.get('tolerance_standard', 'IT8')
        auto_extract_dimensions = data.get('auto_extract_dimensions', 'true').lower() == 'true'
        auto_identify_features = data.get('auto_identify_features', 'true').lower() == 'true'
        
        # 保存临时文件
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        
        try:
            # 转换DXF为模板
            global current_template, annotation_tool, inspector
            template = dxf_to_template(
                temp_path,
                template_name=template_name,
                tolerance_standard=tolerance_standard,
                auto_extract_dimensions=auto_extract_dimensions,
                auto_identify_features=auto_identify_features
            )
            
            if template:
                current_template = template
                annotation_tool = AnnotationTool(current_template)
                inspector = AnnotationBasedInspector(current_template, config)
                
                return jsonify({
                    'success': True,
                    'message': 'DXF文件导入成功',
                    'template': current_template.to_dict(),
                    'annotations_count': len(current_template.annotations)
                })
            else:
                return jsonify({'error': 'DXF文件解析失败'}), 500
        
        finally:
            # 删除临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except ImportError:
        return jsonify({'error': 'ezdxf库未安装，无法导入DXF文件'}), 500
    except Exception as e:
        return jsonify({'error': f'导入DXF文件失败: {str(e)}'}), 500


@app.route('/api/dxf/dimensions', methods=['POST'])
def extract_dxf_dimensions():
    """提取DXF文件的尺寸标注"""
    try:
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({'error': '请上传DXF文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '请选择文件'}), 400
        
        # 保存临时文件
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        
        try:
            # 提取尺寸标注
            dimensions = extract_dimensions_from_dxf(temp_path)
            
            return jsonify({
                'success': True,
                'dimensions': dimensions,
                'count': len(dimensions)
            })
        
        finally:
            # 删除临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        return jsonify({'error': f'提取尺寸标注失败: {str(e)}'}), 500


@app.route('/api/dxf/parse', methods=['POST'])
def parse_dxf():
    """解析DXF文件"""
    try:
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({'error': '请上传DXF文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '请选择文件'}), 400
        
        # 保存临时文件
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        
        try:
            # 解析DXF文件
            result = parse_dxf_file(temp_path)
            
            if result:
                # 转换为JSON可序列化的格式
                entities_data = {
                    'circles': [
                        {
                            'center': {'x': c.center.x, 'y': c.center.y},
                            'radius': c.radius,
                            'diameter': c.diameter,
                            'layer': c.layer
                        } for c in result['entities']['circles']
                    ],
                    'arcs': [
                        {
                            'center': {'x': a.center.x, 'y': a.center.y},
                            'radius': a.radius,
                            'start_angle': a.start_angle,
                            'end_angle': a.end_angle,
                            'layer': a.layer
                        } for a in result['entities']['arcs']
                    ],
                    'lines': [
                        {
                            'start': {'x': l.start.x, 'y': l.start.y},
                            'end': {'x': l.end.x, 'y': l.end.y},
                            'length': l.length,
                            'layer': l.layer
                        } for l in result['entities']['lines']
                    ],
                    'polylines': [
                        {
                            'points': [{'x': p.x, 'y': p.y} for p in p.points],
                            'is_closed': p.is_closed,
                            'layer': p.layer
                        } for p in result['entities']['polylines']
                    ]
                }
                
                dimensions_data = [
                    {
                        'type': d.dim_type,
                        'text': d.text,
                        'value': d.extract_value(),
                        'unit': d.unit,
                        'layer': d.layer,
                        'points': [{'x': p.x, 'y': p.y} for p in d.points]
                    } for d in result['dimensions']
                ]
                
                return jsonify({
                    'success': True,
                    'entities': entities_data,
                    'dimensions': dimensions_data,
                    'layers': result['layers'],
                    'summary': {
                        'circles_count': len(result['entities']['circles']),
                        'arcs_count': len(result['entities']['arcs']),
                        'lines_count': len(result['entities']['lines']),
                        'polylines_count': len(result['entities']['polylines']),
                        'dimensions_count': len(result['dimensions'])
                    }
                })
            else:
                return jsonify({'error': 'DXF文件解析失败'}), 500
        
        finally:
            # 删除临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        return jsonify({'error': f'解析DXF文件失败: {str(e)}'}), 500


# =============================================================================
# 标定API
# =============================================================================

@app.route('/api/calibration/pixel_to_mm', methods=['POST'])
def calibrate_pixel_to_mm():
    """像素-毫米标定"""
    try:
        data = request.json
        actual_length = data.get('actual_length')
        pixel_length = data.get('pixel_length')
        
        if not actual_length or not pixel_length:
            return jsonify({'error': '请提供实际长度和像素长度'}), 400
        
        # 计算转换系数
        pixel_to_mm = actual_length / pixel_length
        
        # 更新配置
        config.PIXEL_TO_MM = pixel_to_mm
        current_template.image_scale = pixel_to_mm
        
        return jsonify({
            'success': True,
            'pixel_to_mm': pixel_to_mm,
            'message': f'标定完成: 1像素 = {pixel_to_mm:.6f}毫米'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/calibration/camera', methods=['POST'])
def calibrate_camera():
    """相机内参标定"""
    try:
        data = request.json
        images = data.get('images', [])
        
        if not images:
            return jsonify({'error': '请提供标定图像'}), 400
        
        # TODO: 实现相机标定逻辑
        return jsonify({
            'success': True,
            'message': '相机标定完成',
            'camera_matrix': None,
            'distortion_coefficients': None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# WebSocket事件
# =============================================================================

@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    print(f'客户端已连接: {request.sid}')
    emit('connected', {'message': '连接成功'})


@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开"""
    print(f'客户端已断开: {request.sid}')


@socketio.on('start_preview')
def handle_start_preview():
    """开始实时预览"""
    try:
        # TODO: 实现实时预览逻辑
        emit('preview_started', {'message': '预览已开始'})
    except Exception as e:
        emit('error', {'message': str(e)})


@socketio.on('stop_preview')
def handle_stop_preview():
    """停止实时预览"""
    try:
        # TODO: 实现停止预览逻辑
        emit('preview_stopped', {'message': '预览已停止'})
    except Exception as e:
        emit('error', {'message': str(e)})


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    # 创建必要的目录
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/images', exist_ok=True)
    
    # 启动服务器
    socketio.run(app, host='0.0.0.0', port=5008, debug=True)