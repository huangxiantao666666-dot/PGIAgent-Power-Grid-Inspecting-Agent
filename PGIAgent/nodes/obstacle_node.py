#!/usr/bin/env python3
"""
障碍物检测节点 - 提供check_obstacle()工具功能
使用激光雷达检测障碍物，分析安全方向
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import LaserScan
from PGIAgent.srv import CheckObstacle
import numpy as np
import math
import time
from threading import Lock
from enum import Enum


class SafetyLevel(Enum):
    """安全级别"""
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"


class ObstacleNode(Node):
    """障碍物检测节点，提供障碍物检测服务"""
    
    def __init__(self):
        super().__init__('obstacle_node')
        
        # 声明参数
        self.declare_parameters(
            namespace='',
            parameters=[
                ('lidar_topic', '/scan'),
                ('service_name', '/pgi_agent/check_obstacle'),
                ('safety_distance', 0.5),  # 安全距离（米）
                ('warning_distance', 1.0),  # 警告距离
                ('danger_distance', 0.3),   # 危险距离
                ('critical_distance', 0.2), # 临界距离
                ('angle_resolution', 45),    # 扇区角度分辨率（度）- 改为45度8个扇区
                ('front_sector', 60),       # 前方扇区角度
                ('min_valid_distance', 0.1), # 最小有效距离
                ('max_valid_distance', 5.0), # 最大有效距离
                ('use_simulation', False),  # 使用模拟模式
                ('scan_timeout', 1.0),      # 激光雷达数据超时时间（秒）
            ]
        )
        
        # 获取参数
        self.lidar_topic = self.get_parameter('lidar_topic').value
        self.service_name = self.get_parameter('service_name').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.warning_distance = self.get_parameter('warning_distance').value
        self.danger_distance = self.get_parameter('danger_distance').value
        self.critical_distance = self.get_parameter('critical_distance').value
        self.angle_resolution = self.get_parameter('angle_resolution').value
        self.front_sector = self.get_parameter('front_sector').value
        self.min_valid_distance = self.get_parameter('min_valid_distance').value
        self.max_valid_distance = self.get_parameter('max_valid_distance').value
        self.use_simulation = self.get_parameter('use_simulation').value
        self.scan_timeout = self.get_parameter('scan_timeout').value
        
        # 计算扇区数量（360度 / 分辨率）
        self.num_sectors = 360 // self.angle_resolution
        
        # 状态变量
        self.latest_scan = None
        self.scan_lock = Lock()
        self.scan_received = False
        self.last_scan_time = None
        
        # 创建订阅器
        if not self.use_simulation:
            self.lidar_sub = self.create_subscription(
                LaserScan, self.lidar_topic, self.lidar_callback, 10
            )
        
        # 创建服务
        self.obstacle_service = self.create_service(
            CheckObstacle,
            self.service_name,
            self.handle_obstacle_request,
            callback_group=ReentrantCallbackGroup()
        )
        
        # 初始化模拟数据
        self.simulation_data = self._initialize_simulation_data()
        
        self.get_logger().info(f"障碍物检测节点已启动，服务: {self.service_name}")
        self.get_logger().info(f"安全距离: {self.safety_distance}m, 警告距离: {self.warning_distance}m")
        if self.use_simulation:
            self.get_logger().info("运行在模拟模式")
    
    def _initialize_simulation_data(self):
        """初始化模拟数据 - 使用标准的激光雷达消息格式"""
        # 创建模拟的LaserScan消息
        scan_msg = LaserScan()
        
        # 标准参数：前方为0度，逆时针增加
        scan_msg.angle_min = -math.pi  # -180度
        scan_msg.angle_max = math.pi   # 180度
        scan_msg.angle_increment = math.radians(1)  # 1度分辨率
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.1
        scan_msg.range_min = self.min_valid_distance
        scan_msg.range_max = self.max_valid_distance
        
        # 计算点数
        num_points = int((scan_msg.angle_max - scan_msg.angle_min) / scan_msg.angle_increment) + 1
        
        # 初始化所有点都在最大距离
        ranges = np.ones(num_points) * self.max_valid_distance
        
        # 在正前方添加一个障碍物（0度附近）
        front_center_idx = int((0 - scan_msg.angle_min) / scan_msg.angle_increment)
        for i in range(-5, 6):
            if 0 <= front_center_idx + i < num_points:
                ranges[front_center_idx + i] = self.safety_distance * 0.8
        
        # 在左侧添加一个障碍物（-90度，即270度）
        left_idx = int((math.radians(-90) - scan_msg.angle_min) / scan_msg.angle_increment)
        for i in range(-3, 4):
            if 0 <= left_idx + i < num_points:
                ranges[left_idx + i] = self.warning_distance * 0.7
        
        # 在右侧添加一个障碍物（90度）
        right_idx = int((math.radians(90) - scan_msg.angle_min) / scan_msg.angle_increment)
        for i in range(-3, 4):
            if 0 <= right_idx + i < num_points:
                ranges[right_idx + i] = self.danger_distance * 0.9
        
        scan_msg.ranges = ranges.tolist()
        scan_msg.intensities = []
        
        return scan_msg
    
    def lidar_callback(self, msg):
        """激光雷达数据回调"""
        try:
            with self.scan_lock:
                self.latest_scan = msg
                self.scan_received = True
                self.last_scan_time = self.get_clock().now()
                
        except Exception as e:
            self.get_logger().error(f"激光雷达数据处理错误: {e}")
    
    def _get_latest_scan_with_timeout(self):
        """获取最新的激光雷达数据，带超时机制"""
        if self.use_simulation:
            return self.simulation_data
        
        start_time = self.get_clock().now()
        
        while (self.get_clock().now() - start_time).nanoseconds / 1e9 < self.scan_timeout:
            with self.scan_lock:
                if self.latest_scan is not None:
                    return self.latest_scan
            time.sleep(0.05)  # 50ms轮询间隔
        
        return None
    
    def handle_obstacle_request(self, request, response):
        """处理障碍物检测请求"""
        try:
            # 获取激光雷达数据（带超时）
            scan_data = self._get_latest_scan_with_timeout()
            
            if scan_data is None:
                response.success = False
                response.message = f"未收到激光雷达数据（超时 {self.scan_timeout}s）"
                # 返回默认安全值
                response.safe_direction = 0.0
                response.min_distance = self.max_valid_distance
                response.safe_sectors = [True] * self.num_sectors
                return response
            
            # 提取数据
            ranges = np.array(scan_data.ranges)
            
            # 生成对应的角度数组（绝对角度，弧度）
            angles = np.linspace(scan_data.angle_min, scan_data.angle_max, len(ranges))
            
            # 分析障碍物
            analysis_result = self._analyze_obstacles(ranges, angles)
            
            # 填充响应
            response.success = True
            response.message = analysis_result['message']
            response.safe_direction = analysis_result['safe_direction']  # 已经是相对角度
            response.min_distance = analysis_result['min_distance']
            response.safety_level = analysis_result['safety_level']
            response.obstacle_count = analysis_result['obstacle_count']
            
            # 确保safe_sectors是布尔列表
            if isinstance(analysis_result['safe_sectors'], list):
                response.safe_sectors = analysis_result['safe_sectors']
            else:
                response.safe_sectors = [True] * self.num_sectors
            
            self.get_logger().info(
                f"障碍物分析: {response.safety_level}, "
                f"最小距离: {response.min_distance:.2f}m, "
                f"安全方向: {response.safe_direction:.1f}°"
            )
            
        except Exception as e:
            self.get_logger().error(f"障碍物分析过程中发生错误: {e}")
            response.success = False
            response.message = f"分析失败: {str(e)}"
            response.safe_direction = 0.0
            response.min_distance = self.max_valid_distance
            response.safe_sectors = [True] * self.num_sectors
        
        return response
    
    def _absolute_to_relative_angle(self, absolute_angle_rad):
        """
        将绝对角度转换为相对角度（机器人坐标系：前为0度）
        输入：弧度，范围 [-π, π] 或 [0, 2π]
        输出：度，范围 [-180, 180]
        """
        # 确保角度在 [-π, π] 范围内
        if absolute_angle_rad > math.pi:
            absolute_angle_rad -= 2 * math.pi
        
        # 转换为度
        relative_deg = math.degrees(absolute_angle_rad)
        
        return relative_deg
    
    def _relative_to_absolute_angle(self, relative_deg):
        """
        将相对角度转换为绝对角度（用于调试）
        输入：度，范围 [-180, 180]
        输出：弧度，范围 [-π, π]
        """
        return math.radians(relative_deg)
    
    def _analyze_obstacles(self, ranges, angles):
        """分析障碍物"""
        # 过滤无效数据
        valid_mask = (ranges >= self.min_valid_distance) & (ranges <= self.max_valid_distance) & np.isfinite(ranges)
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        
        if len(valid_ranges) == 0:
            return {
                'message': "未检测到有效障碍物数据",
                'safe_direction': 0.0,
                'min_distance': self.max_valid_distance,
                'safety_level': SafetyLevel.SAFE.value,
                'obstacle_count': 0,
                'safe_sectors': [True] * self.num_sectors
            }
        
        # 计算最小距离
        min_distance = np.min(valid_ranges)
        
        # 确定安全级别
        safety_level = self._determine_safety_level(min_distance)
        
        # 分析各个扇区
        sector_analysis = self._analyze_sectors(ranges, angles)
        
        # 找到最安全的方向（返回相对角度）
        safe_direction = self._find_safest_direction(sector_analysis)
        
        # 统计障碍物数量
        obstacle_count = np.sum(valid_ranges < self.safety_distance)
        
        # 生成安全扇区掩码（布尔列表）
        safe_sectors = self._generate_sector_mask(sector_analysis)
        
        # 生成消息
        message = self._generate_analysis_message(
            safety_level, min_distance, obstacle_count, safe_direction
        )
        
        return {
            'message': message,
            'safe_direction': safe_direction,  # 已经是相对角度
            'min_distance': float(min_distance),
            'safety_level': safety_level.value,
            'obstacle_count': int(obstacle_count),
            'safe_sectors': safe_sectors
        }
    
    def _determine_safety_level(self, min_distance):
        """确定安全级别"""
        if min_distance < self.critical_distance:
            return SafetyLevel.CRITICAL
        elif min_distance < self.danger_distance:
            return SafetyLevel.DANGER
        elif min_distance < self.warning_distance:
            return SafetyLevel.WARNING
        else:
            return SafetyLevel.SAFE
    
    def _analyze_sectors(self, ranges, angles):
        """分析各个扇区"""
        sector_analysis = []
        
        for i in range(self.num_sectors):
            # 扇区角度范围（绝对角度）
            start_angle_abs = i * self.angle_resolution
            end_angle_abs = (i + 1) * self.angle_resolution
            
            # 转换为弧度，并转换到 [-π, π] 范围
            start_rad = math.radians(start_angle_abs - 180)  # 0度转换为 -180度
            end_rad = math.radians(end_angle_abs - 180)      # 360度转换为 180度
            
            # 找到该扇区内的数据点
            if start_rad < end_rad:
                mask = (angles >= start_rad) & (angles < end_rad)
            else:
                # 处理角度环绕（如从 350度 到 10度）
                mask = (angles >= start_rad) | (angles < end_rad)
            
            sector_ranges = ranges[mask]
            
            if len(sector_ranges) > 0:
                # 过滤无效数据
                valid_mask = (sector_ranges >= self.min_valid_distance) & (sector_ranges <= self.max_valid_distance) & np.isfinite(sector_ranges)
                valid_ranges = sector_ranges[valid_mask]
                
                if len(valid_ranges) > 0:
                    min_dist = np.min(valid_ranges)
                    avg_dist = np.mean(valid_ranges)
                else:
                    min_dist = self.max_valid_distance
                    avg_dist = self.max_valid_distance
            else:
                min_dist = self.max_valid_distance
                avg_dist = self.max_valid_distance
            
            # 计算扇区中心角（相对角度）
            center_angle_abs = (start_angle_abs + end_angle_abs) / 2
            center_angle_rel = self._absolute_to_relative_angle(math.radians(center_angle_abs - 180))
            
            sector_analysis.append({
                'sector_id': i,
                'start_angle_abs': start_angle_abs,
                'end_angle_abs': end_angle_abs,
                'center_angle_rel': center_angle_rel,  # 相对角度
                'min_distance': min_dist,
                'avg_distance': avg_dist,
                'is_safe': min_dist >= self.safety_distance
            })
        
        return sector_analysis
    
    def _find_safest_direction(self, sector_analysis):
        """找到最安全的方向（返回相对角度）"""
        # 优先考虑前方扇区（相对角度 -30 到 30度）
        front_sectors = []
        other_sectors = []
        
        for sector in sector_analysis:
            # 判断是否在前方范围内
            if abs(sector['center_angle_rel']) <= self.front_sector / 2:
                front_sectors.append(sector)
            else:
                other_sectors.append(sector)
        
        # 首先找前方完全安全的扇区
        safe_front = [s for s in front_sectors if s['is_safe']]
        if safe_front:
            # 选择距离最远的
            safest = max(safe_front, key=lambda x: x['min_distance'])
            return safest['center_angle_rel']
        
        # 其次找其他方向完全安全的扇区
        safe_others = [s for s in other_sectors if s['is_safe']]
        if safe_others:
            safest = max(safe_others, key=lambda x: x['min_distance'])
            return safest['center_angle_rel']
        
        # 如果没有完全安全的扇区，找所有扇区中距离最远的
        safest = max(sector_analysis, key=lambda x: x['min_distance'])
        return safest['center_angle_rel']
    
    def _generate_sector_mask(self, sector_analysis):
        """生成8个扇区的安全掩码（布尔列表）"""
        safe_mask = [False] * self.num_sectors
        
        for sector in sector_analysis:
            if sector['sector_id'] < self.num_sectors and sector['is_safe']:
                safe_mask[sector['sector_id']] = True
        
        return safe_mask
    
    def _generate_analysis_message(self, safety_level, min_distance, obstacle_count, safe_direction):
        """生成分析消息"""
        messages = {
            SafetyLevel.SAFE: f"前方安全，最小距离 {min_distance:.2f}m",
            SafetyLevel.WARNING: f"前方有障碍物警告，最小距离 {min_distance:.2f}m",
            SafetyLevel.DANGER: f"前方有障碍物危险，最小距离 {min_distance:.2f}m",
            SafetyLevel.CRITICAL: f"前方有障碍物临界，最小距离 {min_distance:.2f}m，建议立即停止"
        }
        
        base_message = messages.get(safety_level, "未知安全状态")
        
        if obstacle_count > 0:
            obstacle_info = f"，检测到 {obstacle_count} 个障碍物"
        else:
            obstacle_info = ""
        
        direction_info = f"，建议朝向 {safe_direction:.1f}° 方向移动"
        
        return base_message + obstacle_info + direction_info
    
    def destroy_node(self):
        """清理资源"""
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ObstacleNode()
        
        # 使用多线程执行器
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            node.get_logger().info("障碍物检测节点正在关闭...")
        finally:
            executor.shutdown()
            node.destroy_node()
            
    except Exception as e:
        print(f"障碍物检测节点启动失败: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()