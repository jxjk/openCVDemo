# -*- coding: utf-8 -*-
"""
微小零件中高精度视觉检测系统 - Web版本
Micro Part High-Precision Visual Inspection System - Web Version

版本: V1.0
创建日期: 2026-02-10
精度目标: IT8级（亚像素检测精度≥1/20像素）
Web框架: Flask + Socket.IO

功能:
- Web界面访问
- 实时图像流传输
- 远程检测控制
- 历史数据查询
"""

import os
import sys
import json
import time
import base64
import threading
import datetime
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, Response, send_file
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# 导入核心模块
from inspection_system import (
    InspectionConfig,
    CameraCalibration,
    USBCameraDriver,
    DahengCameraDriver,
    SubpixelDetector,
    GeometryFitter,
    InspectionResult,
    InspectionEngine,
    DataManager
)

# =============================================================================
# Flask应用配置
# =============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'inspection-system-secret-key-2026'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# 启用CORS
CORS(app)

# 配置Socket.IO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# =============================================================================
# 全局变量
# =============================================================================

# 初始化核心组件
config = InspectionConfig()
calibration = CameraCalibration(config)
camera_driver = None
inspection_engine = InspectionEngine(config, calibration)
data_manager = DataManager(config)

# 加载配置和标定数据
if os.path.exists('data/config.json'):
    with open('data/config.json', 'r', encoding='utf-8') as f:
        config_data = json.load(f)
        config.PIXEL_TO_MM = config_data.get('pixel_to_mm', 0.098)

calibration.load_calibration()

# 控制变量
is_camera_connected = False
is_previewing = False
preview_thread = None
stop_preview = False
current_frame = None

# =============================================================================
# 路由定义
# =============================================================================

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """获取系统状态"""
    stats = data_manager.get_statistics()
    
    return jsonify({
        'camera_connected': is_camera_connected,
        'previewing': is_previewing,
        'pixel_to_mm': config.PIXEL_TO_MM,
        'stats': stats,
        'timestamp': datetime.datetime.now().isoformat()
    })


@app.route('/api/camera/connect', methods=['POST'])
def connect_camera():
    """连接相机"""
    global camera_driver, is_camera_connected
    
    try:
        data = request.json
        camera_type = data.get('camera_type', 'usb')
        device_id = data.get('device_id', '0')
        
        if camera_type.lower() == 'usb':
            camera_driver = USBCameraDriver()
        elif camera_type.lower() == 'daheng':
            camera_driver = DahengCameraDriver()
        else:
            return jsonify({'success': False, 'message': f'不支持的相机类型: {camera_type}'}), 400
        
        if camera_driver.connect(device_id):
            is_camera_connected = True
            socketio.emit('camera_status', {'connected': True})
            return jsonify({'success': True, 'message': '相机连接成功'})
        else:
            return jsonify({'success': False, 'message': '相机连接失败'}), 500
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'连接异常: {str(e)}'}), 500


@app.route('/api/camera/disconnect', methods=['POST'])
def disconnect_camera():
    """断开相机"""
    global camera_driver, is_camera_connected
    
    stop_preview_func()
    
    if camera_driver:
        camera_driver.disconnect()
        camera_driver = None
    
    is_camera_connected = False
    socketio.emit('camera_status', {'connected': False})
    
    return jsonify({'success': True, 'message': '相机已断开'})


@app.route('/api/camera/start_preview', methods=['POST'])
def start_preview():
    """开始预览"""
    global is_previewing, stop_preview, preview_thread
    
    if not is_camera_connected:
        return jsonify({'success': False, 'message': '请先连接相机'}), 400
    
    if is_previewing:
        return jsonify({'success': False, 'message': '已经在预览中'}), 400
    
    try:
        is_previewing = True
        stop_preview = False
        
        preview_thread = threading.Thread(target=preview_loop, daemon=True)
        preview_thread.start()
        
        return jsonify({'success': True, 'message': '预览已启动'})
    
    except Exception as e:
        is_previewing = False
        return jsonify({'success': False, 'message': f'启动预览失败: {str(e)}'}), 500


@app.route('/api/camera/stop_preview', methods=['POST'])
def stop_preview():
    """停止预览"""
    global is_previewing, stop_preview
    
    stop_preview = True
    is_previewing = False
    
    return jsonify({'success': True, 'message': '预览已停止'})


@app.route('/api/inspect', methods=['POST'])
def inspect_part():
    """检测零件"""
    global is_previewing, stop_preview
    
    if not is_camera_connected:
        return jsonify({'success': False, 'message': '请先连接相机'}), 400
    
    try:
        # 停止预览
        was_previewing = is_previewing
        if was_previewing:
            stop_preview = True
            time.sleep(0.1)
        
        # 采集图像
        image = camera_driver.capture_image()
        if image is None:
            return jsonify({'success': False, 'message': '图像采集失败'}), 500
        
        # 获取参数
        data = request.json
        part_id = data.get('part_id', '')
        part_type = data.get('part_type', '圆形')
        nominal_size = data.get('nominal_size')
        
        try:
            nominal_size = float(nominal_size) if nominal_size else None
        except ValueError:
            nominal_size = None
        
        # 检测
        if part_type == '圆形':
            result = inspection_engine.detect_circle(
                image, part_id, part_type, nominal_size
            )
        else:
            result = inspection_engine.detect_rectangle(
                image, part_id, part_type, nominal_size
            )
        
        if result is None:
            return jsonify({'success': False, 'message': '检测失败，未检测到零件'}), 400
        
        # 绘制结果
        result_image = inspection_engine.draw_result(image, result)
        
        # 保存图像
        result.image_path = data_manager.save_image(result_image, result)
        
        # 保存结果
        data_manager.save_result(result)
        
        # 编码结果图像为base64
        _, buffer = cv2.imencode('.jpg', result_image)
        result_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 恢复预览
        if was_previewing:
            stop_preview = False
            preview_thread = threading.Thread(target=preview_loop, daemon=True)
            preview_thread.start()
        
        # 发送结果
        socketio.emit('inspection_result', {
            'result': result.to_dict(),
            'image': result_image_base64
        })
        
        return jsonify({
            'success': True,
            'result': result.to_dict(),
            'image': result_image_base64
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'检测过程出错: {str(e)}'}), 500


@app.route('/api/calibrate', methods=['POST'])
def calibrate_system():
    """标定系统"""
    global is_previewing, stop_preview
    
    if not is_camera_connected:
        return jsonify({'success': False, 'message': '请先连接相机'}), 400
    
    try:
        data = request.json
        known_diameter = float(data.get('known_diameter', 10.0))
        
        # 停止预览
        was_previewing = is_previewing
        if was_previewing:
            stop_preview = True
            time.sleep(0.1)
        
        # 采集图像
        image = camera_driver.capture_image()
        if image is None:
            return jsonify({'success': False, 'message': '图像采集失败'}), 500
        
        # 检测圆形
        result = inspection_engine.detect_circle(
            image, '标定件', '圆形', known_diameter
        )
        
        if result is None or result.diameter_pixel is None:
            return jsonify({'success': False, 'message': '标定失败，未检测到圆形零件'}), 400
        
        # 计算像素-毫米转换系数
        pixel_to_mm = known_diameter / result.diameter_pixel
        config.PIXEL_TO_MM = pixel_to_mm
        
        # 保存配置
        config_data = {
            'pixel_to_mm': pixel_to_mm,
            'calibration_date': datetime.datetime.now().isoformat()
        }
        
        with open('data/config.json', 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        
        # 恢复预览
        if was_previewing:
            stop_preview = False
            preview_thread = threading.Thread(target=preview_loop, daemon=True)
            preview_thread.start()
        
        return jsonify({
            'success': True,
            'message': '标定完成',
            'pixel_to_mm': pixel_to_mm,
            'measured_diameter_pixel': result.diameter_pixel
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'标定过程出错: {str(e)}'}), 500


@app.route('/api/results')
def get_results():
    """获取检测结果"""
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        results = data_manager.results_history[-(offset + limit):]
        if offset > 0:
            results = results[-limit:]
        
        return jsonify({
            'success': True,
            'results': [r.to_dict() for r in results],
            'total': len(data_manager.results_history)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/results/export', methods=['POST'])
def export_results():
    """导出结果"""
    try:
        if data_manager.export_to_excel():
            return jsonify({'success': True, 'message': '数据已导出'})
        else:
            return jsonify({'success': False, 'message': '导出失败'}), 500
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/images/<filename>')
def get_image(filename):
    """获取检测图像"""
    try:
        filepath = os.path.join('data/images', filename)
        if os.path.exists(filepath):
            return send_file(filepath)
        else:
            return jsonify({'success': False, 'message': '图像不存在'}), 404
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/health')
def health_check():
    """健康检查"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.datetime.now().isoformat()})


# =============================================================================
# Socket.IO事件
# =============================================================================

@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    print(f'客户端已连接: {request.sid}')
    emit('connected', {'message': '已连接到服务器'})


@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开"""
    print(f'客户端已断开: {request.sid}')


# =============================================================================
# 辅助函数
# =============================================================================

def preview_loop():
    """预览循环"""
    global is_previewing, stop_preview, current_frame
    
    while not stop_preview and is_camera_connected:
        try:
            image = camera_driver.capture_image()
            if image is not None:
                # 调整图像大小以减少传输数据量
                h, w = image.shape[:2]
                max_size = 800
                
                if max(h, w) > max_size:
                    scale = max_size / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    image = cv2.resize(image, (new_w, new_h))
                
                # 编码为JPEG
                _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # 发送到客户端
                socketio.emit('preview_frame', {'image': frame_base64})
            
            time.sleep(0.05)  # ~20 FPS
        
        except Exception as e:
            print(f'预览循环错误: {e}')
            time.sleep(0.1)


def stop_preview_func():
    """停止预览"""
    global is_previewing, stop_preview
    stop_preview = True
    is_previewing = False


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='微小零件视觉检测系统 - Web版本')
    parser.add_argument('--host', default='0.0.0.0', help='主机地址')
    parser.add_argument('--port', type=int, default=5008, help='端口号')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    # 创建模板目录
    os.makedirs('templates', exist_ok=True)
    
    # 检查模板文件是否存在
    if not os.path.exists('templates/index.html'):
        print('警告: 模板文件 templates/index.html 不存在')
        print('请创建模板文件或使用其他方式访问')
    
    print(f'启动Web服务器...')
    print(f'访问地址: http://{args.host}:{args.port}')
    
    socketio.run(app, host=args.host, port=args.port, debug=args.debug)