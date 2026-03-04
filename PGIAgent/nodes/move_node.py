#!/usr/bin/env python3
"""
移动工具节点 - 控制小车移动
提供 move(velocity, angle, seconds) 工具功能
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup # 并发执行多个回调，允许多个回调同时运行
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger
from PGIAgent.srv import MoveCommand  # 修正导入
import math
from threading import Lock


class MoveNode(Node):
    """
    移动工具节点，提供移动控制服务，本服务通过定时器定时向Twist话题发布消息
    当客户端发送请求时，服务器会检查当前运动状态，如果发布运动请求且当前未处于运动状态，则开始运动
    客户端也可以发送停止运动的请求
    服务器采用线程锁来保证线程安全    
    """
    
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
                ('control_frequency', 10.0),  # 控制频率
            ]
        )
        
        # 获取参数
        self.default_velocity = self.get_parameter('default_velocity').value
        self.default_seconds = self.get_parameter('default_seconds').value
        self.angular_scaling = self.get_parameter('angular_scaling').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.control_frequency = self.get_parameter('control_frequency').value
        
        # 创建服务
        self.service_name = self.get_parameter('service_name').value
        self.move_service = self.create_service(
            MoveCommand,
            self.service_name,
            self.handle_move_command, # 回调函数
            callback_group=ReentrantCallbackGroup() # 回调组，允许并发执行多个回调
        )
        
        # 创建发布器
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        
        # 状态变量
        self.is_moving = False
        self.move_lock = Lock() # 线程锁，同一时间只能有一个获得了锁的线程运行代码
        self.target_twist = Twist()
        self.move_end_time = 0.0
        
        # 创建控制定时器（非阻塞）
        timer_period = 1.0 / self.control_frequency
        self.control_timer = self.create_timer(timer_period, self.control_callback)
        self.control_timer.cancel()  # 初始时取消
        
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
            # 检查是否正在移动
            if self.is_moving:
                response.success = False
                response.message = "正在执行其他移动命令，请稍后再试"
                return response
            
            # 参数验证
            if request.seconds < 0:
                response.success = False
                response.message = "移动时间不能为负数"
                return response
            
            # 获取参数
            velocity = request.velocity if request.velocity > 0 else self.default_velocity
            angle = request.angle
            seconds = request.seconds if request.seconds > 0 else self.default_seconds
            
            # 安全检查
            velocity = min(abs(velocity), self.max_velocity)
            if velocity == 0:
                response.success = False
                response.message = "速度不能为0"
                return response
            
            # 执行移动
            try:
                self._start_move(velocity, angle, seconds)
                
                response.success = True
                response.message = f"移动开始: 速度{velocity}m/s, 角度{angle}°, 时间{seconds}秒"
                self.get_logger().info(response.message)
                
            except Exception as e:
                self.get_logger().error(f"移动启动失败: {e}")
                response.success = False
                response.message = f"移动失败: {str(e)}"
        
        return response
    
    def _start_move(self, velocity, angle, seconds):
        """启动移动（非阻塞）"""
        # 计算目标Twist
        angular_velocity = 0.0
        if abs(angle) > 0.1:
            angular_velocity = math.radians(angle) * self.angular_scaling
        
        self.target_twist.linear.x = float(velocity)
        self.target_twist.angular.z = float(angular_velocity)
        
        # 设置结束时间
        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        self.move_end_time = current_time + seconds
        
        # 标记为移动中
        self.is_moving = True
        
        # 激活控制定时器
        self.control_timer.reset()
    
    def control_callback(self):
        """定时器回调，非阻塞控制，定时器的回调函数来发布消息"""
        if not self.is_moving:
            return
        
        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        
        if current_time >= self.move_end_time:
            # 时间到，停止
            self._stop_move()
        else:
            # 继续移动
            self.cmd_vel_pub.publish(self.target_twist)
    
    def _stop_move(self):
        """停止移动"""
        self.is_moving = False
        self.control_timer.cancel()
        
        # 发布零速度
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        
        self.get_logger().debug("移动停止")
    
    def handle_stop_command(self, request, response):
        """处理停止命令"""
        with self.move_lock:
            if self.is_moving:
                self._stop_move()
                response.success = True
                response.message = "移动已停止"
                self.get_logger().info("移动被手动停止")
            else:
                response.success = True
                response.message = "当前没有移动任务"
        
        return response
    
    def destroy_node(self):
        """节点销毁时停止机器人"""
        self._stop_move()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = MoveNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("移动工具节点正在关闭...")
    except Exception as e:
        node.get_logger().error(f"节点运行错误: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()