"""
工具函数封装
提供智能体可调用的工具函数，与ROS2服务交互
"""

import rclpy
from rclpy.node import Node
from rclpy.client import Client
import time
import json
from typing import Dict, Any, List, Optional, Tuple
import yaml
import os

from .state import AgentConfig, ToolType, DetectedObject, ObstacleInfo, OCRResult


class ToolManager:
    """工具管理器，负责与ROS2服务交互"""
    
    def __init__(self, config: AgentConfig, node: Optional[Node] = None):
        self.config = config
        self.node = node
        self.clients: Dict[str, Client] = {}
        
        # 如果提供了ROS节点，初始化服务客户端
        if node and config.ros_enabled:
            self._init_ros_clients()
    
    def _init_ros_clients(self):
        """初始化ROS服务客户端"""
        from std_srvs.srv import Trigger
        from pgi_agent_msgs.srv import (
            MoveCommand, YOLODetect, VLMDetect, 
            Track, CheckObstacle, OCR
        )
        
        # 创建服务客户端
        service_types = {
            self.config.move_service: MoveCommand,
            self.config.yolo_service: YOLODetect,
            self.config.vlm_service: VLMDetect,
            self.config.track_service: Track,
            self.config.obstacle_service: CheckObstacle,
            self.config.ocr_service: OCR,
        }
        
        for service_name, service_type in service_types.items():
            try:
                client = self.node.create_client(service_type, service_name)
                self.clients[service_name] = client
                
                # 等待服务可用
                if not client.wait_for_service(timeout_sec=5.0):
                    self.node.get_logger().warn(f"服务 {service_name} 不可用")
            except Exception as e:
                self.node.get_logger().error(f"创建服务客户端失败 {service_name}: {e}")
    
    def move(self, velocity: Optional[float] = None, 
             angle: float = 0.0, 
             seconds: Optional[float] = None) -> Dict[str, Any]:
        """
        移动工具
        Args:
            velocity: 移动速度 (m/s)，正数为前进，负数为后退
            angle: 转向角度 (度)，0为直行，正数为左转，负数为右转
            seconds: 移动时间 (秒)
        Returns:
            移动结果
        """
        if not self.config.ros_enabled:
            return self._simulate_move(velocity, angle, seconds)
        
        # 使用默认值
        velocity = velocity if velocity is not None else self.config.default_move_velocity
        seconds = seconds if seconds is not None else self.config.default_move_seconds
        
        # 安全检查
        velocity = max(min(velocity, self.config.max_velocity), -self.config.max_velocity)
        
        try:
            client = self.clients.get(self.config.move_service)
            if not client:
                return {"success": False, "message": "移动服务不可用"}
            
            request = MoveCommand.Request()
            request.velocity = float(velocity)
            request.angle = float(angle)
            request.seconds = float(seconds)
            
            future = client.call_async(request)
            
            # 等待响应
            start_time = time.time()
            while not future.done():
                if time.time() - start_time > 30:  # 30秒超时
                    return {"success": False, "message": "移动服务调用超时"}
                time.sleep(0.1)
            
            response = future.result()
            return {
                "success": response.success,
                "message": response.message,
                "velocity": velocity,
                "angle": angle,
                "seconds": seconds
            }
            
        except Exception as e:
            return {"success": False, "message": f"移动失败: {str(e)}"}
    
    def yolo_detect(self, threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        YOLO物体检测工具
        Args:
            threshold: 置信度阈值 (0.0-1.0)
        Returns:
            检测结果
        """
        if not self.config.ros_enabled:
            return self._simulate_yolo_detect(threshold)
        
        threshold = threshold if threshold is not None else self.config.yolo_threshold
        
        try:
            client = self.clients.get(self.config.yolo_service)
            if not client:
                return {"success": False, "message": "YOLO检测服务不可用"}
            
            request = YOLODetect.Request()
            request.threshold = float(threshold)
            
            future = client.call_async(request)
            
            # 等待响应
            start_time = time.time()
            while not future.done():
                if time.time() - start_time > 10:  # 10秒超时
                    return {"success": False, "message": "YOLO检测服务调用超时"}
                time.sleep(0.1)
            
            response = future.result()
            
            # 解析检测结果
            objects = []
            if response.success and len(response.objects) > 0:
                for i in range(len(response.objects)):
                    obj = DetectedObject(
                        name=response.objects[i],
                        confidence=0.8,  # 默认置信度
                        distance=response.distances[i] if i < len(response.distances) else 0.0,
                        position=response.positions[i] if i < len(response.positions) else "未知"
                    )
                    objects.append(obj.__dict__)
            
            return {
                "success": response.success,
                "message": response.message,
                "objects": objects,
                "threshold": threshold,
                "count": len(objects)
            }
            
        except Exception as e:
            return {"success": False, "message": f"YOLO检测失败: {str(e)}"}
    
    def vlm_detect(self) -> Dict[str, Any]:
        """
        视觉大模型场景理解工具
        Returns:
            场景描述结果
        """
        if not self.config.ros_enabled:
            return self._simulate_vlm_detect()
        
        try:
            client = self.clients.get(self.config.vlm_service)
            if not client:
                return {"success": False, "message": "VLM检测服务不可用"}
            
            request = VLMDetect.Request()
            
            future = client.call_async(request)
            
            # 等待响应
            start_time = time.time()
            while not future.done():
                if time.time() - start_time > 30:  # 30秒超时
                    return {"success": False, "message": "VLM检测服务调用超时"}
                time.sleep(0.1)
            
            response = future.result()
            
            return {
                "success": response.success,
                "message": response.message,
                "description": response.description if response.success else "",
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {"success": False, "message": f"VLM检测失败: {str(e)}"}
    
    def track(self, target: Optional[str] = None) -> Dict[str, Any]:
        """
        目标追踪工具
        Args:
            target: 追踪目标 (如"person", "electric_box")
        Returns:
            追踪结果
        """
        if not self.config.ros_enabled:
            return self._simulate_track(target)
        
        target = target if target is not None else "person"
        
        try:
            client = self.clients.get(self.config.track_service)
            if not client:
                return {"success": False, "message": "追踪服务不可用"}
            
            request = Track.Request()
            request.target = target
            
            future = client.call_async(request)
            
            # 等待响应
            start_time = time.time()
            while not future.done():
                if time.time() - start_time > 60:  # 60秒超时
                    return {"success": False, "message": "追踪服务调用超时"}
                time.sleep(0.1)
            
            response = future.result()
            
            return {
                "success": response.success,
                "message": response.message,
                "target": target,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {"success": False, "message": f"追踪失败: {str(e)}"}
    
    def check_obstacle(self) -> Dict[str, Any]:
        """
        障碍物检测工具
        Returns:
            障碍物检测结果
        """
        if not self.config.ros_enabled:
            return self._simulate_check_obstacle()
        
        try:
            client = self.clients.get(self.config.obstacle_service)
            if not client:
                return {"success": False, "message": "障碍物检测服务不可用"}
            
            request = CheckObstacle.Request()
            
            future = client.call_async(request)
            
            # 等待响应
            start_time = time.time()
            while not future.done():
                if time.time() - start_time > 5:  # 5秒超时
                    return {"success": False, "message": "障碍物检测服务调用超时"}
                time.sleep(0.1)
            
            response = future.result()
            
            # 解析障碍物信息
            obstacle_info = None
            if response.success:
                obstacle_info = ObstacleInfo(
                    safe_direction=response.safe_direction,
                    min_distance=response.min_distance,
                    safe_sectors=list(response.safe_sectors),
                    sector_distances=[1.0] * 8  # 模拟距离
                )
            
            return {
                "success": response.success,
                "message": response.message,
                "obstacle_info": obstacle_info.__dict__ if obstacle_info else None,
                "safe_direction": response.safe_direction if response.success else 0.0,
                "min_distance": response.min_distance if response.success else 0.0
            }
            
        except Exception as e:
            return {"success": False, "message": f"障碍物检测失败: {str(e)}"}
    
    def ocr(self) -> Dict[str, Any]:
        """
        OCR文字识别工具
        Returns:
            OCR识别结果
        """
        if not self.config.ros_enabled:
            return self._simulate_ocr()
        
        try:
            client = self.clients.get(self.config.ocr_service)
            if not client:
                return {"success": False, "message": "OCR服务不可用"}
            
            request = OCR.Request()
            
            future = client.call_async(request)
            
            # 等待响应
            start_time = time.time()
            while not future.done():
                if time.time() - start_time > 10:  # 10秒超时
                    return {"success": False, "message": "OCR服务调用超时"}
                time.sleep(0.1)
            
            response = future.result()
            
            # 解析OCR结果
            ocr_results = []
            if response.success and len(response.texts) > 0:
                for i in range(len(response.texts)):
                    result = OCRResult(
                        text=response.texts[i],
                        confidence=response.confidences[i] if i < len(response.confidences) else 0.8
                    )
                    ocr_results.append(result.__dict__)
            
            return {
                "success": response.success,
                "message": response.message,
                "texts": [r["text"] for r in ocr_results],
                "results": ocr_results,
                "count": len(ocr_results)
            }
            
        except Exception as e:
            return {"success": False, "message": f"OCR识别失败: {str(e)}"}
    
    # 模拟工具函数（用于测试）
    def _simulate_move(self, velocity: Optional[float], angle: float, seconds: Optional[float]) -> Dict[str, Any]:
        """模拟移动"""
        velocity = velocity if velocity is not None else self.config.default_move_velocity
        seconds = seconds if seconds is not None else self.config.default_move_seconds
        
        time.sleep(seconds)  # 模拟移动时间
        
        return {
            "success": True,
            "message": f"模拟移动完成: 速度={velocity}m/s, 角度={angle}°, 时间={seconds}s",
            "velocity": velocity,
            "angle": angle,
            "seconds": seconds
        }
    
    def _simulate_yolo_detect(self, threshold: Optional[float]) -> Dict[str, Any]:
        """模拟YOLO检测"""
        threshold = threshold if threshold is not None else self.config.yolo_threshold
        
        # 模拟检测结果
        objects = [
            DetectedObject("person", 0.85, 2.5, "画面中央").__dict__,
            DetectedObject("electric_box", 0.92, 3.0, "右上方").__dict__,
            DetectedObject("warning_sign", 0.78, 1.8, "左下方").__dict__
        ]
        
        return {
            "success": True,
            "message": f"模拟YOLO检测完成，阈值={threshold}",
            "objects": objects,
            "threshold": threshold,
            "count": len(objects)
        }
    
    def _simulate_vlm_detect(self) -> Dict[str, Any]:
        """模拟VLM检测"""
        description = "这是一个变电站场景。画面中央有一个电力控制箱，箱体表面有警告标志。右侧有一个变压器设备，看起来运行正常。地面上有电缆通道，需要注意安全。"
        
        return {
            "success": True,
            "message": "模拟VLM检测完成",
            "description": description,
            "timestamp": time.time()
        }
    
    def _simulate_track(self, target: Optional[str]) -> Dict[str, Any]:
        """模拟追踪"""
        target = target if target is not None else "person"
        
        time.sleep(2)  # 模拟追踪时间
        
        return {
            "success": True,
            "message": f"模拟追踪完成: 目标={target}",
            "target": target,
            "timestamp": time.time()
        }
    
    def _simulate_check_obstacle(self) -> Dict[str, Any]:
        """模拟障碍物检测"""
        obstacle_info = ObstacleInfo(
            safe_direction=15.0,
            min_distance=1.2,
            safe_sectors=[True, True, True, False, False, True, True, True],
            sector_distances=[2.0, 1.8, 1.5, 0.3, 0.4, 1.6, 1.9, 2.1]
        )
        
        return {
            "success": True,
            "message": "模拟障碍物检测完成",
            "obstacle_info": obstacle_info.__dict__,
            "safe_direction": 15.0,
            "min_distance": 1.2
        }
    
    def _simulate_ocr(self) -> Dict[str, Any]:
        """模拟OCR"""
        ocr_results = [
            OCRResult("高压危险", 0.95).__dict__,
            OCRResult("禁止入内", 0.88).__dict__,
            OCRResult("变电站A区", 0.92).__dict__
        ]
        
        return {
            "success": True,
            "message": "模拟OCR识别完成",
            "texts": ["高压危险", "禁止入内", "变电站A区"],
            "results": ocr_results,
            "count": len(ocr_results)
        }


def create_tool_functions(tool_manager: ToolManager) -> Dict[str, Any]:
    """创建工具函数字典，供智能体调用"""
    
    def move_wrapper(velocity: Optional[float] = None, 
                    angle: float = 0.0, 
                    seconds: Optional[float] = None) -> str:
        """移动工具包装函数"""
        result = tool_manager.move(velocity, angle, seconds)
        return json.dumps(result, ensure_ascii=False)
    
    def yolo_detect_wrapper(threshold: Optional[float] = None) -> str:
        """YOLO检测工具包装函数"""
        result = tool_manager.yolo_detect(threshold)
        return json.dumps(result, ensure_ascii=False)
    
    def vlm_detect_wrapper() -> str:
        """VLM检测工具包装函数"""
        result = tool_manager.vlm_detect()
        return json.dumps(result, ensure_ascii=False)
    
    def track_wrapper(target: Optional[str] = None) -> str:
        """追踪工具包装函数"""
        result = tool_manager.track(target)
        return json.dumps(result, ensure_ascii=False)
    
    def check_obstacle_wrapper() -> str:
        """障碍物检测工具包装函数"""
        result = tool_manager.check_obstacle()
        return json.dumps(result, ensure_ascii=False)
    
    def ocr_wrapper() -> str:
        """OCR工具包装函数"""
        result = tool_manager.ocr()
        return json.dumps(result, ensure_ascii=False)
    
    return {
        "move": move_wrapper,
        "yolo_detect": yolo_detect_wrapper,
        "VLM_detect": vlm_detect_wrapper,
        "track": track_wrapper,
        "check_obstacle": check_obstacle_wrapper,
        "ocr": ocr_wrapper
    }


def load_tool_config(config_path: str) -> AgentConfig:
    """从配置文件加载工具配置"""
    if not os.path.exists(config_path):
        return AgentConfig()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # 从配置数据创建AgentConfig
    config = AgentConfig()
    
    # 更新配置
    if 'agent' in config_data:
        agent_config = config_data['agent']
        if 'max_iterations' in agent_config:
            config.max_iterations = agent_config['max_iterations']
        if 'use_reflection' in agent_config.get('planning', {}):
            config.use_reflection = agent_config['planning']['use_reflection']
    
    if 'tools' in config_data:
        tools_config = config_data['tools']
        if 'move' in tools_config:
            move_config = tools_config['move']
            if 'default_velocity' in move_config:
                config.default_move_velocity = move_config['default_velocity']
            if 'default_seconds' in move_config:
                config.default_move_seconds = move_config['default_seconds']
        
        if 'yolo_detect' in tools_config:
            yolo_config = tools_config['yolo_detect']
            if 'default_threshold' in yolo_config:
                config.yolo_threshold = yolo_config['default_threshold']
        
        if 'track' in tools_config:
            track_config = tools_config['track']
            if 'tracking_distance' in track_config:
                config.tracking_distance = track_config['tracking_distance']
    
    if 'ros_topics' in config_data:
        ros_config = config_data['ros_topics']
        if 'control' in ros_config:
            control_config = ros_config['control']
            if 'cmd_vel' in control_config:
                # 可以在这里设置话题名称，但当前配置结构不支持
                pass
    
    return config
