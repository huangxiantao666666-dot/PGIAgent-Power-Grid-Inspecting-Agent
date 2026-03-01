#!/usr/bin/env python3
"""
目标追踪节点 - 提供track(which)工具功能
追踪指定目标（如人员），保持安全距离
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from pgi_agent_msgs.srv import Track, TrackResponse
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import math
from threading import Lock, Thread, Event
from enum import Enum
import os

# 尝试导入YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("警告: ultralytics未安装，使用模拟模式")


class TrackingState(Enum):
    """追踪状态"""
    IDLE = "idle"
    SEARCHING = "searching"
    TRACKING = "tracking"
    LOST = "lost"
    COMPLETED = "completed"


class TrackNode(Node):
    """目标追踪节点，提供目标追踪服务"""
    
    def __init__(self):
        super().__init__('track_node')
        
        # 声明参数
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', 'yolo11n.pt'),
                ('default_target', 'person'),
                ('target_distance', 1.2),  # 目标距离（米）
                ('distance_tolerance', 0.2),  # 距离容差
                ('max_tracking_time', 60.0),  # 最大追踪时间（秒）
                ('search_timeout', 10.0),  # 搜索超时时间
                ('camera_topic', '/depth_cam/rgb/image_raw'),
                ('depth_topic', '/depth_cam/depth/image_raw'),
                ('cmd_vel_topic', '/cmd_vel'),
                ('service_name', '/pgi_agent/track'),
                ('kp_linear', 0.5),
                ('kd_linear', 0.1),
                ('kp_angular', 0.8),
                ('kd_angular', 0.2),
                ('max_linear_vel', 0.3),
                ('max_angular_vel', 0.5),
                ('use_simulation', False),
            ]
        )
        
        # 获取参数
        self.model_path = self.get_parameter('model_path').value
        self.default_target = self.get_parameter('default_target').value
        self.target_distance = self.get_parameter('target_distance').value
        self.distance_tolerance = self.get_parameter('distance_tolerance').value
        self.max_tracking_time = self.get_parameter('max_tracking_time').value
        self.search_timeout = self.get_parameter('search_timeout').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.service_name = self.get_parameter('service_name').value
        self.kp_linear = self.get_parameter('kp_linear').value
        self.kd_linear = self.get_parameter('kd_linear').value
        self.kp_angular = self.get_parameter('kp_angular').value
        self.kd_angular = self.get_parameter('kd_angular').value
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
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
        
        # 创建发布器
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        
        # 创建服务
        self.track_service = self.create_service(
            Track,
            self.service_name,
            self.handle_track_request,
            callback_group=ReentrantCallbackGroup()
        )
        
        # 创建停止服务
        self.stop_service = self.create_service(
            Trigger,
            f'{self.service_name}/stop',
            self.handle_stop_request,
            callback_group=ReentrantCallbackGroup()
        )
        
        # 状态变量
        self.state = TrackingState.IDLE
        self.state_lock = Lock()
        self.tracking_thread = None
        self.stop_event = Event()
        
        # 追踪变量
        self.current_target = self.default_target
        self.target_class_id = 0  # 默认person类别
        self.last_error_distance = 0.0
        self.last_error_angle = 0.0
        self.tracking_start_time = 0.0
        self.search_start_time = 0.0
        
        # 目标信息
        self.target_distance_measured = 0.0
        self.target_angle = 0.0
        self.target_confidence = 0.0
        self.target_lost_count = 0
        
        # 类别映射
        self.class_mapping = {
            'person': 0,
            'electric_box': 1,
            'transformer': 2,
            'car': 3,
            'dog': 4,
        }
        
        self.get_logger().info(f"目标追踪节点已启动，服务: {self.service_name}")
        self.get_logger().info(f"默认目标: {self.default_target}, 目标距离: {self.target_distance}m")
        if self.use_simulation:
            self.get_logger().info("运行在模拟模式")
    
    def _initialize_model(self):
        """初始化YOLO模型"""
        try:
            self.get_logger().info(f"加载追踪模型: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # 预热模型
            dummy_input = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
            _ = self.model.track(dummy_input, persist=True, verbose=False)
            self.get_logger().info("追踪模型加载成功")
            
        except Exception as e:
            self.get_logger().error(f"模型加载失败: {e}")
            self.use_simulation = True
            self.get_logger().info("切换到模拟模式")
    
    def color_callback(self, msg):
        """颜色图像回调"""
        try:
            with self.frame_lock:
                self.latest_color_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"颜色图像处理错误: {e}")
    
    def depth_callback(self, msg):
        """深度图像回调"""
        try:
            with self.frame_lock:
                self.latest_depth_frame = self.bridge.imgmsg_to_cv2(msg, '16UC1')
        except Exception as e:
            self.get_logger().error(f"深度图像处理错误: {e}")
    
    def handle_track_request(self, request, response):
        """处理追踪请求"""
        try:
            with self.state_lock:
                if self.state != TrackingState.IDLE:
                    response.success = False
                    response.message = f"当前正在{self.state.value}，无法开始新追踪"
                    return response
                
                # 设置目标
                target = request.target if request.target else self.default_target
                self.current_target = target
                self.target_class_id = self.class_mapping.get(target, 0)
                
                # 开始追踪
                self.state = TrackingState.SEARCHING
                self.stop_event.clear()
                self.tracking_start_time = time.time()
                self.search_start_time = time.time()
                self.target_lost_count = 0
                
                # 启动追踪线程
                self.tracking_thread = Thread(target=self._tracking_loop)
                self.tracking_thread.start()
                
                response.success = True
                response.message = f"开始追踪目标: {target}"
                
                self.get_logger().info(f"开始追踪: {target}")
                
        except Exception as e:
            self.get_logger().error(f"追踪请求处理错误: {e}")
            response.success = False
            response.message = f"追踪启动失败: {str(e)}"
        
        return response
    
    def handle_stop_request(self, request, response):
        """处理停止请求"""
        try:
            with self.state_lock:
                if self.state == TrackingState.IDLE:
                    response.success = True
                    response.message = "当前没有追踪任务"
                else:
                    self._stop_tracking()
                    response.success = True
                    response.message = "追踪已停止"
                    self.get_logger().info("追踪被手动停止")
                    
        except Exception as e:
            self.get_logger().error(f"停止请求处理错误: {e}")
            response.success = False
            response.message = f"停止失败: {str(e)}"
        
        return response
    
    def _tracking_loop(self):
        """追踪主循环"""
        try:
            while not self.stop_event.is_set():
                current_state = self.state
                
                if current_state == TrackingState.SEARCHING:
                    self._search_target()
                elif current_state == TrackingState.TRACKING:
                    self._track_target()
                elif current_state == TrackingState.LOST:
                    self._handle_target_lost()
                elif current_state == TrackingState.COMPLETED:
                    break
                elif current_state == TrackingState.IDLE:
                    break
                
                # 控制循环频率
                time.sleep(0.1)  # 10Hz
                
        except Exception as e:
            self.get_logger().error(f"追踪循环错误: {e}")
            with self.state_lock:
                self.state = TrackingState.IDLE
        
        # 清理
        self._stop_robot()
    
    def _search_target(self):
        """搜索目标"""
        current_time = time.time()
        
        # 检查超时
        if current_time - self.search_start_time > self.search_timeout:
            self.get_logger().warning("搜索目标超时")
            with self.state_lock:
                self.state = TrackingState.IDLE
            return
        
        # 获取最新图像
        with self.frame_lock:
            if self.latest_color_frame is None or self.latest_depth_frame is None:
                return
            
            color_frame = self.latest_color_frame.copy()
            depth_frame = self.latest_depth_frame.copy()
        
        # 检测目标
        detection_result = self._detect_target(color_frame, depth_frame)
        
        if detection_result['found']:
            # 找到目标，开始追踪
            self.target_distance_measured = detection_result['distance']
            self.target_angle = detection_result['angle']
            self.target_confidence = detection_result['confidence']
            
            with self.state_lock:
                self.state = TrackingState.TRACKING
                self.search_start_time = 0  # 重置搜索时间
            
            self.get_logger().info(f"找到目标: 距离={self.target_distance_measured:.2f}m, 角度={self.target_angle:.1f}°")
            
            # 发布初始控制命令
            self._publish_control_command()
        else:
            # 未找到目标，缓慢旋转搜索
            self._search_rotation()
    
    def _track_target(self):
        """追踪目标"""
        current_time = time.time()
        
        # 检查追踪时间
        if current_time - self.tracking_start_time > self.max_tracking_time:
            self.get_logger().warning("达到最大追踪时间")
            with self.state_lock:
                self.state = TrackingState.COMPLETED
            return
        
        # 获取最新图像
        with self.frame_lock:
            if self.latest_color_frame is None or self.latest_depth_frame is None:
                self.target_lost_count += 1
                if self.target_lost_count > 10:
                    with self.state_lock:
                        self.state = TrackingState.LOST
                return
            
            color_frame = self.latest_color_frame.copy()
            depth_frame = self.latest_depth_frame.copy()
        
        # 检测目标
        detection_result = self._detect_target(color_frame, depth_frame)
        
        if detection_result['found']:
            # 更新目标信息
            self.target_distance_measured = detection_result['distance']
            self.target_angle = detection_result['angle']
            self.target_confidence = detection_result['confidence']
            self.target_lost_count = 0
            
            # 检查是否达到目标距离
            if abs(self.target_distance_measured - self.target_distance) < self.distance_tolerance:
                self.get_logger().info(f"达到目标距离: {self.target_distance_measured:.2f}m")
                with self.state_lock:
                    self.state = TrackingState.COMPLETED
                return
            
            # 发布控制命令
            self._publish_control_command()
            
        else:
            # 目标丢失
            self.target_lost_count += 1
            if self.target_lost_count > 5:  # 连续5帧丢失
                self.get_logger().warning("目标丢失")
                with self.state_lock:
                    self.state = TrackingState.LOST
    
    def _handle_target_lost(self):
        """处理目标丢失"""
        # 停止移动
        self._stop_robot()
        
        # 短暂等待后重新搜索
        time.sleep(1.0)
        
        with self.state_lock:
            self.state = TrackingState.SEARCHING
            self.search_start_time = time.time()
        
        self.get_logger().info("重新搜索目标")
    
    def _detect_target(self, color_frame, depth_frame):
        """检测目标"""
        if self.use_simulation:
            return self._simulate_detection(color_frame, depth_frame)
        else:
            return self._perform_detection(color_frame, depth_frame)
    
    def _perform_detection(self, color_frame, depth_frame):
        """执行实际的检测"""
        h, w = color_frame.shape[:2]
        
        # 运行YOLO追踪
        results = self.model.track(
            color_frame,
            persist=True,
            classes=[self.target_class_id],
            verbose=False
        )
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            # 获取第一个检测框（假设是主要目标）
            box = results[0].boxes.xyxy[0].cpu().numpy()
            confidence = results[0].boxes.conf[0].cpu().numpy()
            
            # 计算中心点
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            
            # 获取深度
            distance = self._get_depth_at_point(depth_frame, cx, cy)
            
            # 计算角度（相对于图像中心）
            angle = (cx - w/2) / (w/2) * 30  # 转换为角度
            
            return {
                'found': True,
                'distance': distance,
                'angle': angle,
                'confidence': float(confidence),
                'box': box
            }
        
        return {'found': False}
    
    def _simulate_detection(self, color_frame, depth_frame):
        """模拟检测"""
        h, w = color_frame.shape[:2]
        
        # 模拟检测结果
        import random
        if random.random() > 0.3:  # 70%的概率检测到目标
            # 模拟目标在图像中的随机位置
            cx = random.randint(w//4, 3*w//4)
            cy = random.randint(h//4, 3*h//4)
            
            # 模拟距离（逐渐接近目标距离）
            current_time = time.time() - self.tracking_start_time
            distance = self.target_distance + (3.0 - current_time * 0.1)  # 逐渐接近
            
            # 计算角度
            angle = (cx - w/2) / (w/2) * 30
            
            return {
                'found': True,
                'distance': max(distance, 0.5),  # 最小距离0.5m
                'angle': angle,
                'confidence': 0.8,
                'box': [cx-50, cy-50, cx+50, cy+50]
            }
        
        return {'found': False}
    
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
    
    def _publish_control_command(self):
        """发布控制命令"""
        # 计算距离误差
        error_distance = self.target_distance_measured - self.target_distance
        
        # 计算角度误差
        error_angle = self.target_angle
        
        # PID控制（简化版）
        current_time = time.time()
        dt = 0.1  # 假设固定时间间隔
        
        # 线性速度控制
        deriv_distance = (error_distance - self.last_error_distance) / dt if dt > 0 else 0.0
        linear_vel = self.kp_linear * error_distance + self.kd_linear * deriv_distance
        
        # 角度速度控制
        deriv_angle = (error_angle - self.last_error_angle) / dt if dt > 0 else 0.0
        angular_vel = self.kp_angular * error_angle + self.kd_angular * deriv_angle
        
        # 限制速度范围
        linear_vel = max(min(linear_vel, self.max_linear_vel), -self.max_linear_vel)
        angular_vel = max(min(angular_vel, self.max_angular_vel), -self.max_angular_vel)
        
        # 如果距离已经很接近，减小速度
        if abs(error_distance) < self.distance_tolerance * 2:
            linear_vel *= 0.5
            angular_vel *= 0.5
        
        # 发布控制命令
        twist = Twist()
        twist.linear.x = float(linear_vel)
        twist.angular.z = float(angular_vel)
        self.cmd_vel_pub.publish(twist)
        
        # 保存当前误差
        self.last_error_distance = error_distance
        self.last_error_angle = error_angle
        
        self.get_logger().debug(
            f"控制命令: 线速度={linear_vel:.2f}, 角速度={angular_vel:.2f}, "
            f"距离误差={error_distance:.2f}, 角度误差={error_angle:.1f}"
        )
    
    def _search_rotation(self):
        """搜索旋转"""
        # 缓慢旋转以搜索目标
        twist = Twist()
        twist.angular.z = 0.3  # 缓慢右转
        self.cmd_vel_pub.publish(twist)
        
        self.get_logger().debug("正在旋转搜索目标")
    
    def _stop_robot(self):
        """停止机器人"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        time.sleep(0.1)  # 确保命令被发送
    
    def _stop_tracking(self):
        """停止追踪"""
        self.stop_event.set()
        if self.tracking_thread and self.tracking_thread.is_alive():
            self.tracking_thread.join(timeout=2.0)
        
        with self.state_lock:
            self.state = TrackingState.IDLE
        
        self._stop_robot()
        self.get_logger().info("追踪已停止")
    
    def destroy_node(self):
        """清理资源"""
        self._stop_tracking()
        
        if self.model is not None:
            # 清理模型资源
            pass
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = TrackNode()
        
        # 使用多线程执行器
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            node.get_logger().info("追踪节点正在关闭...")
        finally:
            executor.shutdown()
            node.destroy_node()
            
    except Exception as e:
        print(f"追踪节点启动失败: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
