#!/usr/bin/env python3
"""
YOLO物体检测节点 - 提供yolo_detect(threshold)工具功能
订阅深度相机，使用YOLO模型检测物体并返回位置和距离信息
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from pgi_agent_msgs.srv import YOLODetect, YOLODetectResponse
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from threading import Lock
import os

# 尝试导入YOLO，如果不可用则使用模拟模式
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("警告: ultralytics未安装，使用模拟模式")


class DetectionNode(Node):
    """YOLO物体检测节点，提供物体检测服务"""
    
    def __init__(self):
        super().__init__('detection_node')
        
        # 声明参数
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', 'yolo11n.pt'),
                ('engine_path', 'yolo11n.engine'),
                ('img_size', 320),
                ('default_threshold', 0.8),
                ('camera_topic', '/depth_cam/rgb/image_raw'),
                ('depth_topic', '/depth_cam/depth/image_raw'),
                ('service_name', '/pgi_agent/yolo_detect'),
                ('use_tensorrt', False),
                ('use_simulation', False),
            ]
        )
        
        # 获取参数
        self.model_path = self.get_parameter('model_path').value
        self.engine_path = self.get_parameter('engine_path').value
        self.img_size = self.get_parameter('img_size').value
        self.default_threshold = self.get_parameter('default_threshold').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.service_name = self.get_parameter('service_name').value
        self.use_tensorrt = self.get_parameter('use_tensorrt').value
        self.use_simulation = self.get_parameter('use_simulation').value or not YOLO_AVAILABLE
        
        # 初始化工具
        self.bridge = CvBridge()
        self.latest_color_frame = None
        self.latest_depth_frame = None
        self.frame_lock = Lock()
        
        # 初始化模型
        self.model = None
        if not self.use_simulation and YOLO_AVAILABLE:
            self._initialize_model()
        
        # 创建订阅器
        self.color_sub = self.create_subscription(
            Image, self.camera_topic, self.color_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, 10
        )
        
        # 创建服务
        self.detect_service = self.create_service(
            YOLODetect,
            self.service_name,
            self.handle_detect_request,
            callback_group=ReentrantCallbackGroup()
        )
        
        # 状态变量
        self.last_frame_time = 0
        self.frame_rate = 0
        
        self.get_logger().info(f"YOLO检测节点已启动，服务: {self.service_name}")
        self.get_logger().info(f"模型: {self.model_path}, 图像大小: {self.img_size}")
        if self.use_simulation:
            self.get_logger().info("运行在模拟模式")
    
    def _initialize_model(self):
        """初始化YOLO模型"""
        try:
            if self.use_tensorrt and os.path.exists(self.engine_path):
                self.get_logger().info(f"加载TensorRT引擎: {self.engine_path}")
                self.model = YOLO(self.engine_path)
            else:
                self.get_logger().info(f"加载PyTorch模型: {self.model_path}")
                self.model = YOLO(self.model_path)
            
            # 预热模型
            dummy_input = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
            _ = self.model(dummy_input, verbose=False)
            self.get_logger().info("YOLO模型加载成功")
            
        except Exception as e:
            self.get_logger().error(f"模型加载失败: {e}")
            self.use_simulation = True
            self.get_logger().info("切换到模拟模式")
    
    def color_callback(self, msg):
        """颜色图像回调"""
        try:
            with self.frame_lock:
                self.latest_color_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                
                # 计算帧率
                current_time = time.time()
                if self.last_frame_time > 0:
                    self.frame_rate = 1.0 / (current_time - self.last_frame_time)
                self.last_frame_time = current_time
                
        except Exception as e:
            self.get_logger().error(f"颜色图像处理错误: {e}")
    
    def depth_callback(self, msg):
        """深度图像回调"""
        try:
            with self.frame_lock:
                self.latest_depth_frame = self.bridge.imgmsg_to_cv2(msg, '16UC1')
        except Exception as e:
            self.get_logger().error(f"深度图像处理错误: {e}")
    
    def handle_detect_request(self, request, response):
        """处理检测请求"""
        try:
            # 获取阈值
            threshold = request.threshold if request.threshold > 0 else self.default_threshold
            
            # 检查是否有最新图像
            with self.frame_lock:
                if self.latest_color_frame is None or self.latest_depth_frame is None:
                    response.success = False
                    response.message = "未收到相机图像"
                    response.objects = []
                    response.distances = []
                    response.positions = []
                    return response
                
                color_frame = self.latest_color_frame.copy()
                depth_frame = self.latest_depth_frame.copy()
            
            # 执行检测
            if self.use_simulation:
                detection_result = self._simulate_detection(color_frame, depth_frame, threshold)
            else:
                detection_result = self._perform_detection(color_frame, depth_frame, threshold)
            
            # 填充响应
            response.success = True
            response.message = f"检测到 {len(detection_result['objects'])} 个物体"
            response.objects = detection_result['objects']
            response.distances = detection_result['distances']
            response.positions = detection_result['positions']
            response.frame_rate = self.frame_rate
            
            self.get_logger().info(f"检测完成: {response.message}")
            
        except Exception as e:
            self.get_logger().error(f"检测过程中发生错误: {e}")
            response.success = False
            response.message = f"检测失败: {str(e)}"
            response.objects = []
            response.distances = []
            response.positions = []
        
        return response
    
    def _perform_detection(self, color_frame, depth_frame, threshold):
        """执行实际的YOLO检测"""
        h, w = color_frame.shape[:2]
        
        # 运行YOLO检测
        results = self.model(
            color_frame,
            imgsz=self.img_size,
            conf=threshold,
            verbose=False
        )
        
        objects = []
        distances = []
        positions = []
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            
            # 获取类别名称
            if hasattr(results[0], 'names'):
                class_names = results[0].names
            else:
                class_names = {i: f"class_{i}" for i in range(int(max(class_ids)) + 1)}
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                # 计算中心点
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                
                # 获取深度值
                distance = self._get_depth_at_point(depth_frame, cx, cy)
                
                # 确定位置描述
                position = self._describe_position(cx, cy, w, h)
                
                # 添加到结果
                class_name = class_names.get(int(cls_id), f"class_{int(cls_id)}")
                objects.append(class_name)
                distances.append(float(distance))
                positions.append(position)
                
                # 记录日志
                self.get_logger().debug(
                    f"检测到: {class_name} (置信度: {conf:.2f}), "
                    f"距离: {distance:.2f}m, 位置: {position}"
                )
        
        return {
            'objects': objects,
            'distances': distances,
            'positions': positions
        }
    
    def _simulate_detection(self, color_frame, depth_frame, threshold):
        """模拟检测（用于测试）"""
        h, w = color_frame.shape[:2]
        
        # 模拟一些检测结果
        objects = ['person', 'electric_box', 'transformer']
        distances = [1.5, 2.3, 3.7]
        
        # 生成随机位置
        positions = []
        for _ in range(len(objects)):
            cx = np.random.randint(w // 4, 3 * w // 4)
            cy = np.random.randint(h // 4, 3 * h // 4)
            positions.append(self._describe_position(cx, cy, w, h))
        
        self.get_logger().info(f"模拟检测: {len(objects)} 个物体")
        
        return {
            'objects': objects,
            'distances': distances,
            'positions': positions
        }
    
    def _get_depth_at_point(self, depth_frame, x, y):
        """获取指定点的深度值"""
        h, w = depth_frame.shape
        
        # 确保坐标在范围内
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        
        # 获取深度值（假设深度图单位是毫米）
        depth_value = depth_frame[y, x]
        
        # 转换为米
        if depth_value > 0:
            return float(depth_value) / 1000.0
        else:
            return 0.0
    
    def _describe_position(self, x, y, img_width, img_height):
        """描述物体在图像中的位置"""
        # 水平位置
        if x < img_width * 0.33:
            horizontal = "左侧"
        elif x < img_width * 0.66:
            horizontal = "中间"
        else:
            horizontal = "右侧"
        
        # 垂直位置
        if y < img_height * 0.33:
            vertical = "上方"
        elif y < img_height * 0.66:
            vertical = "中间"
        else:
            vertical = "下方"
        
        return f"{horizontal}{vertical}"
    
    def destroy_node(self):
        """清理资源"""
        if self.model is not None:
            # 清理模型资源
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = DetectionNode()
        
        # 使用多线程执行器
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            node.get_logger().info("检测节点正在关闭...")
        finally:
            executor.shutdown()
            node.destroy_node()
            
    except Exception as e:
        print(f"检测节点启动失败: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()