#!/usr/bin/env python3
"""
移动工具节点 - 控制小车移动
提供 move(velocity, angle, seconds) 工具功能
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger
from pgi_agent_msgs.srv import MoveCommand
import time
import math
from threading import Lock


class MoveNode(Node):
    """移动工具节点，提供移动控制服务"""
    
    def __init__(self):
        super().__init__('move_node')
        
        # 声明参数
        self.declare_parameters(
            namespace='',
            parameters=[
                ('default_velocity', 0.2),
                ('default_seconds', 2.0),
                ('angular_scaling', 0.5),
                ('cmd_vel_topic', '/cmd_vel'),
                ('service_name', '/pgi_agent/move'),
                ('max_velocity', 0.5),
                ('emergency_stop_distance', 0.2),
            ]
        )
        
        # 获取参数
        self.default_velocity = self.get_parameter('default_velocity').value
        self.default_seconds = self.get_parameter('default_seconds').value
        self.angular_scaling = self.get_parameter('angular_scaling').value
        self.max_velocity = self.get_parameter('max_velocity').value
        
        # 创建服务
        self.service_name = self.get_parameter('service_name').value
        self.move_service = self.create_service(
            MoveCommand,
            self.service_name,
            self.handle_move_command,
            callback_group=ReentrantCallbackGroup()
        )
        
        # 创建发布器
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        
        # 状态变量
        self.is_moving = False
        self.move_lock = Lock()
        self.current_velocity = 0.0
        self.current_angle = 0.0
        
        # 创建停止服务
        self.stop_service = self.create_service(
            Trigger,
            f'{self.service_name}/stop',
            self.handle_stop_command,
            callback_group=ReentrantCallbackGroup()
        )
        
        self.get_logger().info(f"移动工具节点已启动，服务: {self.service_name}")
        self.get_logger().info(f"默认速度: {self.default_velocity} m/s, 默认时间: {self.default_seconds} s")
    
    def handle_move_command(self, request, response):
        """处理移动命令请求"""
        with self.move_lock:
            if self.is_moving:
                response.success = False
                response.message = "正在执行其他移动命令，请稍后再试"
                return response
            
            # 获取参数
            velocity = request.velocity if request.velocity > 0 else self.default_velocity
            angle = request.angle  # 角度，0为直行，正数为左转，负数为右转
            seconds = request.seconds if request.seconds > 0 else self.default_seconds
            
            # 安全检查
            velocity = min(abs(velocity), self.max_velocity)
            if velocity == 0:
                response.success = False
                response.message = "速度不能为0"
                return response
            
            # 标记为正在移动
            self.is_moving = True
            self.current_velocity = velocity
            self.current_angle = angle
            
            try:
                # 执行移动
                self.get_logger().info(f"开始移动: 速度={velocity}m/s, 角度={angle}°, 时间={seconds}s")
                self._execute_move(velocity, angle, seconds)
                
                response.success = True
                response.message = f"移动完成: 以速度{velocity}m/s, 角度{angle}°移动了{seconds}秒"
                
            except Exception as e:
                self.get_logger().error(f"移动过程中发生错误: {e}")
                response.success = False
                response.message = f"移动失败: {str(e)}"
                
            finally:
                self.is_moving = False
                self.current_velocity = 0.0
                self.current_angle = 0.0
        
        return response
    
    def _execute_move(self, velocity, angle, seconds):
        """执行移动操作"""
        start_time = time.time()
        
        # 计算角速度
        angular_velocity = 0.0
        if abs(angle) > 0.1:  # 如果有角度，计算角速度
            # 角度转换为弧度/秒，使用缩放因子
            angular_velocity = math.radians(angle) * self.angular_scaling
        
        # 发布速度命令
        twist = Twist()
        twist.linear.x = float(velocity)
        twist.angular.z = float(angular_velocity)
        
        # 持续发布命令直到时间到
        while time.time() - start_time < seconds:
            if not self.is_moving:  # 检查是否被停止
                break
                
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)  # 10Hz控制频率
        
        # 停止机器人
        self._stop_robot()
    
    def handle_stop_command(self, request, response):
        """处理停止命令"""
        with self.move_lock:
            if self.is_moving:
                self.is_moving = False
                self._stop_robot()
                response.success = True
                response.message = "移动已停止"
                self.get_logger().info("移动被手动停止")
            else:
                response.success = True
                response.message = "当前没有移动任务"
        
        return response
    
    def _stop_robot(self):
        """停止机器人"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        time.sleep(0.1)  # 确保命令被发送
    
    def destroy_node(self):
        """节点销毁时停止机器人"""
        self._stop_robot()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = MoveNode()
        
        # 使用多线程执行器以支持并发服务调用
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            node.get_logger().info("移动工具节点正在关闭...")
        finally:
            executor.shutdown()
            node.destroy_node()
            
    except Exception as e:
        print(f"移动工具节点启动失败: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()