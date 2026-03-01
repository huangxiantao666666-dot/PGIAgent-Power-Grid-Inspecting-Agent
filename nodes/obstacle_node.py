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
from pgi_agent_msgs.srv import CheckObstacle, CheckObstacleResponse
import numpy as np
import math
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
                ('angle_resolution', 5),    # 角度分辨率（度）
                ('front_sector', 60),       # 前方扇区角度
                ('min_valid_distance', 0.1), # 最小有效距离
                ('max_valid_distance', 5.0), # 最大有效距离
                ('use_simulation', False),  # 使用模拟模式
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
        
        # 状态变量
        self.latest_scan = None
        self.scan_lock = Lock()
        self.scan_received = False
        
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
        """初始化模拟数据"""
        # 创建模拟的激光雷达数据
        num_points = 360  # 360度
        ranges = np.ones(num_points) * self.max_valid_distance
        
        # 在前方添加一些障碍物
        front_start = 180 - self.front_sector // 2
        front_end = 180 + self.front_sector // 2
        
        # 在正前方添加一个障碍物
        ranges[175:185] = self.safety_distance * 0.8
        
        # 在左侧添加一个障碍物
        ranges[130:140] = self.warning_distance * 0.7
        
        # 在右侧添加一个障碍物
        ranges[220:230] = self.danger_distance * 0.9
        
        return ranges
    
    def lidar_callback(self, msg):
        """激光雷达数据回调"""
        try:
            with self.scan_lock:
                self.latest_scan = msg
                self.scan_received = True
                
        except Exception as e:
            self.get_logger().error(f"激光雷达数据处理错误: {e}")
    
    def handle_obstacle_request(self, request, response):
        """处理障碍物检测请求"""
        try:
            # 获取激光雷达数据
            if self.use_simulation:
                ranges = self.simulation_data
                angles = np.linspace(0, 2*math.pi, len(ranges))
            else:
                scan_data = self._get_latest_scan()
                if scan_data is None:
                    response.success = False
                    response.message = "未收到激光雷达数据"
                    return response
                
                ranges = np.array(scan_data.ranges)
                angles = np.linspace(scan_data.angle_min, scan_data.angle_max, len(ranges))
            
            # 分析障碍物
            analysis_result = self._analyze_obstacles(ranges, angles)
            
            # 填充响应
            response.success = True
            response.message = analysis_result['message']
            response.safe_direction = analysis_result['safe_direction']
            response.min_distance = analysis_result['min_distance']
            response.safety_level = analysis_result['safety_level']
            response.obstacle_count = analysis_result['obstacle_count']
            response.safe_sectors = analysis_result['safe_sectors']
            
            self.get_logger().info(
                f"障碍物分析: {response.safety_level}, "
                f"最小距离: {response.min_distance:.2f}m, "
                f"安全方向: {response.safe_direction}°"
            )
            
        except Exception as e:
            self.get_logger().error(f"障碍物分析过程中发生错误: {e}")
            response.success = False
            response.message = f"分析失败: {str(e)}"
        
        return response
    
    def _get_latest_scan(self):
        """获取最新的激光雷达数据"""
        with self.scan_lock:
            return self.latest_scan
    
    def _analyze_obstacles(self, ranges, angles):
        """分析障碍物"""
        # 过滤无效数据
        valid_mask = (ranges >= self.min_valid_distance) & (ranges <= self.max_valid_distance)
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        
        if len(valid_ranges) == 0:
            return {
                'message': "未检测到有效障碍物数据",
                'safe_direction': 0.0,
                'min_distance': self.max_valid_distance,
                'safety_level': SafetyLevel.SAFE.value,
                'obstacle_count': 0,
                'safe_sectors': [0, 360]
            }
        
        # 计算最小距离
        min_distance = np.min(valid_ranges)
        min_distance_idx = np.argmin(valid_ranges)
        min_distance_angle = math.degrees(valid_angles[min_distance_idx])
        
        # 确定安全级别
        safety_level = self._determine_safety_level(min_distance)
        
        # 分析各个方向
        sector_analysis = self._analyze_sectors(ranges, angles)
        
        # 找到最安全的方向
        safe_direction = self._find_safest_direction(sector_analysis)
        
        # 统计障碍物数量
        obstacle_count = np.sum(valid_ranges < self.safety_distance)
        
        # 找到安全扇区
        safe_sectors = self._find_safe_sectors(sector_analysis)
        
        # 生成消息
        message = self._generate_analysis_message(
            safety_level, min_distance, obstacle_count, safe_direction
        )
        
        return {
            'message': message,
            'safe_direction': safe_direction,
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
        num_sectors = 360 // self.angle_resolution
        sector_analysis = []
        
        for i in range(num_sectors):
            start_angle = i * self.angle_resolution
            end_angle = (i + 1) * self.angle_resolution
            
            # 转换为弧度
            start_rad = math.radians(start_angle)
            end_rad = math.radians(end_angle)
            
            # 找到该扇区内的数据点
            if start_rad < end_rad:
                mask = (angles >= start_rad) & (angles < end_rad)
            else:
                # 处理角度环绕
                mask = (angles >= start_rad) | (angles < end_rad)
            
            sector_ranges = ranges[mask]
            
            if len(sector_ranges) > 0:
                # 过滤无效数据
                valid_mask = (sector_ranges >= self.min_valid_distance) & (sector_ranges <= self.max_valid_distance)
                valid_ranges = sector_ranges[valid_mask]
                
                if len(valid_ranges) > 0:
                    min_dist = np.min(valid_ranges)
                    avg_dist = np.mean(valid_ranges)
                    max_dist = np.max(valid_ranges)
                else:
                    min_dist = self.max_valid_distance
                    avg_dist = self.max_valid_distance
                    max_dist = self.max_valid_distance
            else:
                min_dist = self.max_valid_distance
                avg_dist = self.max_valid_distance
                max_dist = self.max_valid_distance
            
            # 确定扇区安全级别
            sector_safety = self._determine_safety_level(min_dist)
            
            sector_analysis.append({
                'sector_id': i,
                'start_angle': start_angle,
                'end_angle': end_angle,
                'center_angle': (start_angle + end_angle) / 2,
                'min_distance': min_dist,
                'avg_distance': avg_dist,
                'max_distance': max_dist,
                'safety_level': sector_safety,
                'is_safe': min_dist >= self.safety_distance
            })
        
        return sector_analysis
    
    def _find_safest_direction(self, sector_analysis):
        """找到最安全的方向"""
        # 首先找完全安全的扇区
        safe_sectors = [s for s in sector_analysis if s['is_safe']]
        
        if safe_sectors:
            # 选择距离最远的扇区
            safest_sector = max(safe_sectors, key=lambda x: x['min_distance'])
            return safest_sector['center_angle']
        
        # 如果没有完全安全的扇区，选择相对最安全的
        safest_sector = max(sector_analysis, key=lambda x: x['min_distance'])
        return safest_sector['center_angle']
    
    def _find_safe_sectors(self, sector_analysis):
        """找到安全扇区"""
        safe_sectors = []
        
        for sector in sector_analysis:
            if sector['is_safe']:
                safe_sectors.append({
                    'start': sector['start_angle'],
                    'end': sector['end_angle'],
                    'center': sector['center_angle'],
                    'min_distance': sector['min_distance']
                })
        
        # 合并相邻的安全扇区
        merged_sectors = []
        if safe_sectors:
            current_sector = safe_sectors[0]
            
            for sector in safe_sectors[1:]:
                if sector['start'] <= current_sector['end'] + self.angle_resolution:
                    # 合并扇区
                    current_sector['end'] = sector['end']
                    current_sector['min_distance'] = min(
                        current_sector['min_distance'], sector['min_distance']
                    )
                else:
                    merged_sectors.append(current_sector)
                    current_sector = sector
            
            merged_sectors.append(current_sector)
        
        # 转换为简单的角度范围列表
        simple_sectors = []
        for sector in merged_sectors:
            simple_sectors.append(sector['start'])
            simple_sectors.append(sector['end'])
        
        return simple_sectors if simple_sectors else [0, 360]
    
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