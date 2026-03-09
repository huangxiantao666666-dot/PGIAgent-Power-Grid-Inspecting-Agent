# PGIAgent/agent/tools.py
import rclpy
from rclpy.node import Node
from rclpy.client import Client
import time
import json
from typing import Dict, Any, List, Optional, Tuple
import yaml
import os
import asyncio

from PGIAgent.srv import (
    MoveCommand, YOLODetect, VLMDetect, 
    Track, CheckObstacle, OCR
)

from .state import AgentConfig, DetectedObject, ObstacleInfo, OCRResult

from langchain_core.tools import tool


class ToolManager:
    """工具管理器单例，负责与ROS2服务交互"""
    
    # 类变量保存唯一实例
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """控制实例创建过程，实现单例"""
        if cls._instance is None:
            # 首次调用，创建实例
            print("🔄 创建 ToolManager 单例实例...")
            cls._instance = super().__new__(cls)
        else:
            print(f"✅ 复用已存在的 ToolManager 实例 (ID: {id(cls._instance)})")
        
        return cls._instance
    
    def __init__(self, config: Optional[AgentConfig] = None, node: Optional[Node] = None):
        """
        初始化工具管理器（单例）
        注意：__init__ 每次都会调用，需要用 _initialized 标志避免重复初始化
        """
        # 避免重复初始化
        if ToolManager._initialized:
            return
        
        print("🔧 初始化 ToolManager...")
        self.config = config or AgentConfig()
        self.node = node
        self.clients: Dict[str, Client] = {}
        
        # 如果提供了ROS节点，初始化服务客户端
        if node and self.config.ros_enabled:
            self._init_ros_clients()
        
        # 标记已初始化
        ToolManager._initialized = True
        print(f"✅ ToolManager 初始化完成 (ID: {id(self)})")
    
    def _init_ros_clients(self):
        """初始化ROS服务客户端"""
        service_map = {
            self.config.move_service: MoveCommand,
            self.config.yolo_service: YOLODetect,
            self.config.vlm_service: VLMDetect,
            self.config.track_service: Track,
            self.config.obstacle_service: CheckObstacle,
            self.config.ocr_service: OCR,
        }
        
        for service_name, service_type in service_map.items():
            try:
                client = self.node.create_client(service_type, service_name)
                self.clients[service_name] = client
                
                if not client.wait_for_service(timeout_sec=2.0):
                    self.node.get_logger().warn(f"服务 {service_name} 不可用")
            except Exception as e:
                self.node.get_logger().error(f"创建服务客户端失败 {service_name}: {e}")
    
    def _call_service(self, service_name: str, request, timeout: float = 10.0) -> Optional[Any]:
        """通用服务调用方法"""
        client = self.clients.get(service_name)
        if not client:
            return None
        
        future = client.call_async(request)
        
        start_time = time.time()
        while not future.done():
            if time.time() - start_time > timeout:
                return None
            time.sleep(0.1)
        
        return future.result()
    
    # 异步版本
    async def _call_service_async(self, service_name: str, request, timeout: float = 10.0) -> Optional[Any]:
        """异步通用服务调用方法"""
        client = self.clients.get(service_name)
        if not client:
            return None
        
        future = client.call_async(request)
        
        start_time = time.time()
        while not future.done():
            if time.time() - start_time > timeout:
                return None
            await asyncio.sleep(0.01)
        
        return future.result()
    
    # ========== 工具方法 ==========
    
    def move(self, velocity: Optional[float] = None, angle: float = 0.0, seconds: Optional[float] = None) -> Dict[str, Any]:
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
            request = MoveCommand.Request()
            request.velocity = float(velocity)
            request.angle = float(angle)
            request.seconds = float(seconds)
            
            response = self._call_service(self.config.move_service, request, timeout=30.0)
            
            if response is None:
                return {"success": False, "message": "移动服务调用超时"}
            
            return {
                "success": response.success,
                "message": response.message,
                "velocity": velocity,
                "angle": angle,
                "seconds": seconds
            }
            
        except Exception as e:
            return {"success": False, "message": f"移动失败: {str(e)}"}
    
    async def move_async(self, velocity: Optional[float] = None, angle: float = 0.0, seconds: Optional[float] = None) -> Dict[str, Any]:
        """
        异步移动工具
        Args:
            velocity: 移动速度 (m/s)，正数为前进，负数为后退
            angle: 转向角度 (度)，0为直行，正数为左转，负数为右转
            seconds: 移动时间 (秒)
        Returns:
            移动结果
        """
        if not self.config.ros_enabled:
            return await self._simulate_move_async(velocity, angle, seconds)
        
        velocity = velocity if velocity is not None else self.config.default_move_velocity
        seconds = seconds if seconds is not None else self.config.default_move_seconds
        velocity = max(min(velocity, self.config.max_velocity), -self.config.max_velocity)
        
        try:
            request = MoveCommand.Request()
            request.velocity = float(velocity)
            request.angle = float(angle)
            request.seconds = float(seconds)
            
            response = await self._call_service_async(self.config.move_service, request, timeout=30.0)
            
            if response is None:
                return {"success": False, "message": "移动服务调用超时"}
            
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
            request = YOLODetect.Request()
            request.threshold = float(threshold)
            
            response = self._call_service(self.config.yolo_service, request, timeout=10.0)
            
            if response is None:
                return {"success": False, "message": "YOLO检测服务调用超时"}
            
            # 解析检测结果
            objects = []
            if response.success and len(response.objects) > 0:
                for i in range(len(response.objects)):
                    obj = DetectedObject(
                        name=response.objects[i],
                        confidence=response.confidences[i] if i < len(response.confidences) else 0.8,
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
    
    async def yolo_detect_async(self, threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        YOLO物体检测工具
        Args:
            threshold: 置信度阈值 (0.0-1.0)
        Returns:
            检测结果
        """
        if not self.config.ros_enabled:
            return await self._simulate_yolo_detect_async(threshold)
        
        threshold = threshold if threshold is not None else self.config.yolo_threshold
        
        try:
            request = YOLODetect.Request()
            request.threshold = float(threshold)
            
            response = await self._call_service_async(self.config.yolo_service, request, timeout=10.0)
            
            if response is None:
                return {"success": False, "message": "YOLO检测服务调用超时"}
            
            objects = []
            if response.success and len(response.objects) > 0:
                for i in range(len(response.objects)):
                    obj = DetectedObject(
                        name=response.objects[i],
                        confidence=response.confidences[i] if i < len(response.confidences) else 0.8,
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
            
            response = self._call_service(self.config.vlm_service, request, timeout=30.0)
            
            return {
                "success": response.success,
                "message": response.message,
                "description": response.description if response.success else "",
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {"success": False, "message": f"VLM检测失败: {str(e)}"}
        
        
    async def vlm_detect_async(self) -> Dict[str, Any]:
        """
        异步视觉大模型场景理解工具
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
            
            response = await self._call_service_async(self.config.vlm_service, request, timeout=30.0)
            
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
            
            response = self._call_service(self.config.track_service, request, timeout=30.0)
            
            return {
                "success": response.success,
                "message": response.message,
                "target": target,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {"success": False, "message": f"追踪失败: {str(e)}"}
        
    async def track_async(self, target: Optional[str] = None) -> Dict[str, Any]:
        """
        异步目标追踪工具
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
            
            response = await self._call_service_async(self.config.track_service, request, timeout=30.0)
            
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
            
            response = self._call_service(self.config.obstacle_service, request, timeout=30.0)
            
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
        
    async def check_obstacle_async(self) -> Dict[str, Any]:
        """
        异步障碍物检测工具
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
            
            response = await self._call_service_async(self.config.obstacle_service, request, timeout=30.0)
            
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
            
            response = self._call_service(self.config.ocr_service, request, timeout=30.0)
            
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
        
    async def ocr_async(self) -> Dict[str, Any]:
        """
        异步OCR文字识别工具
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
            
            response = await self._call_service_async(self.config.ocr_service, request, timeout=30.0)
            
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
    
    # ========== 模拟方法 ==========
    
    # ========== 模拟方法 ==========
    
    # ----- 移动模拟 -----
    def _simulate_move(self, velocity: Optional[float], angle: float, seconds: Optional[float]) -> Dict[str, Any]:
        """模拟移动"""
        velocity = velocity if velocity is not None else self.config.default_move_velocity
        seconds = seconds if seconds is not None else self.config.default_move_seconds
        
        time.sleep(min(seconds, 0.5))
        
        return {
            "success": True,
            "message": f"模拟移动完成: 速度={velocity}m/s, 角度={angle}°, 时间={seconds}s",
            "velocity": velocity,
            "angle": angle,
            "seconds": seconds
        }
    
    async def _simulate_move_async(self, velocity: Optional[float], angle: float, seconds: Optional[float]) -> Dict[str, Any]:
        """异步模拟移动"""
        velocity = velocity if velocity is not None else self.config.default_move_velocity
        seconds = seconds if seconds is not None else self.config.default_move_seconds
        
        await asyncio.sleep(min(seconds, 0.5))
        
        return {
            "success": True,
            "message": f"模拟移动完成: 速度={velocity}m/s, 角度={angle}°, 时间={seconds}s",
            "velocity": velocity,
            "angle": angle,
            "seconds": seconds
        }
    
    # ----- YOLO检测模拟 -----
    def _simulate_yolo_detect(self, threshold: Optional[float]) -> Dict[str, Any]:
        """模拟YOLO检测"""
        threshold = threshold if threshold is not None else self.config.yolo_threshold
        
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
    
    async def _simulate_yolo_detect_async(self, threshold: Optional[float]) -> Dict[str, Any]:
        """异步模拟YOLO检测"""
        await asyncio.sleep(0.3)
        return self._simulate_yolo_detect(threshold)
    
    # ----- VLM检测模拟 -----
    def _simulate_vlm_detect(self) -> Dict[str, Any]:
        """模拟VLM检测"""
        # 模拟不同的场景描述
        descriptions = [
            "这是一个变电站场景。画面中央有一个电力控制箱，箱体表面有警告标志。右侧有一个变压器设备，看起来运行正常。地面上有电缆通道，需要注意安全。",
            "检测到输电线路区域。前方有高压电塔，电线清晰可见。周围有安全警示牌，写着'高压危险'。环境晴朗，视野良好。",
            "巡检区域发现异常。变压器表面有轻微锈蚀，需要进一步检查。仪表读数正常，但指示灯显示黄色警告。建议靠近详细检查。",
            "设备运行正常。所有仪表读数在正常范围内，无异常报警。可以继续下一区域巡检。",
        ]
        
        # 随机选择一个描述
        import random
        description = random.choice(descriptions)
        
        return {
            "success": True,
            "message": "模拟VLM检测完成",
            "description": description,
            "timestamp": time.time()
        }
    
    async def _simulate_vlm_detect_async(self) -> Dict[str, Any]:
        """异步模拟VLM检测"""
        await asyncio.sleep(0.5)  # 模拟VLM推理延迟
        return self._simulate_vlm_detect()
    
    # ----- 目标追踪模拟 -----
    def _simulate_track(self, target: Optional[str]) -> Dict[str, Any]:
        """模拟目标追踪"""
        target = target if target is not None else "person"
        
        time.sleep(0.3)  # 模拟追踪启动延迟
        
        return {
            "success": True,
            "message": f"模拟追踪完成: 目标={target}，已锁定并开始跟随",
            "target": target,
            "timestamp": time.time(),
            "tracking_status": "active",
            "target_distance": 2.5  # 模拟距离
        }
    
    async def _simulate_track_async(self, target: Optional[str]) -> Dict[str, Any]:
        """异步模拟目标追踪"""
        target = target if target is not None else "person"
        
        await asyncio.sleep(0.3)
        
        return {
            "success": True,
            "message": f"模拟追踪完成: 目标={target}，已锁定并开始跟随",
            "target": target,
            "timestamp": time.time(),
            "tracking_status": "active",
            "target_distance": 2.5
        }
    
    # ----- 障碍物检测模拟 -----
    def _simulate_check_obstacle(self) -> Dict[str, Any]:
        """模拟障碍物检测"""
        import random
        
        # 随机生成不同的障碍物情况
        scenarios = [
            {
                "safe_direction": 15.0,
                "min_distance": 1.2,
                "safe_sectors": [True, True, True, False, False, True, True, True],
                "sector_distances": [2.0, 1.8, 1.5, 0.3, 0.4, 1.6, 1.9, 2.1],
                "message": "前方有障碍物，建议左转15度"
            },
            {
                "safe_direction": -10.0,
                "min_distance": 0.8,
                "safe_sectors": [False, True, True, True, True, True, False, False],
                "sector_distances": [0.5, 1.2, 1.8, 2.0, 1.9, 1.5, 0.6, 0.4],
                "message": "左侧有障碍物，建议右转10度"
            },
            {
                "safe_direction": 0.0,
                "min_distance": 3.5,
                "safe_sectors": [True, True, True, True, True, True, True, True],
                "sector_distances": [3.5, 3.6, 3.8, 4.0, 4.1, 3.9, 3.7, 3.5],
                "message": "前方畅通，可以直行"
            }
        ]
        
        scenario = random.choice(scenarios)
        
        obstacle_info = ObstacleInfo(
            safe_direction=scenario["safe_direction"],
            min_distance=scenario["min_distance"],
            safe_sectors=scenario["safe_sectors"],
            sector_distances=scenario["sector_distances"]
        )
        
        return {
            "success": True,
            "message": scenario["message"],
            "obstacle_info": obstacle_info.__dict__,
            "safe_direction": scenario["safe_direction"],
            "min_distance": scenario["min_distance"]
        }
    
    async def _simulate_check_obstacle_async(self) -> Dict[str, Any]:
        """异步模拟障碍物检测"""
        await asyncio.sleep(0.2)
        return self._simulate_check_obstacle()
    
    # ----- OCR识别模拟 -----
    def _simulate_ocr(self) -> Dict[str, Any]:
        """模拟OCR识别"""
        import random
        
        # 模拟不同的OCR识别结果
        scenarios = [
            {
                "texts": ["高压危险", "禁止入内", "变电站A区"],
                "confidences": [0.95, 0.88, 0.92],
                "positions": ["左侧上方", "右侧中间", "下方中间"]
            },
            {
                "texts": ["变压器: 110kV/10kV", "电流: 150A", "功率: 5000kW"],
                "confidences": [0.92, 0.85, 0.89],
                "positions": ["画面中央", "右侧下方", "左侧上方"]
            },
            {
                "texts": ["注意安全", "必须佩戴安全帽", "当心触电"],
                "confidences": [0.97, 0.93, 0.96],
                "positions": ["上方中间", "左侧中间", "右侧中间"]
            },
            {
                "texts": ["设备编号: TR-2024-001", "生产日期: 2024-01-15", "下次维护: 2024-07-15"],
                "confidences": [0.94, 0.91, 0.90],
                "positions": ["左侧上方", "中间", "右侧下方"]
            }
        ]
        
        scenario = random.choice(scenarios)
        
        # 创建OCRResult对象列表
        ocr_results = []
        for i in range(len(scenario["texts"])):
            result = OCRResult(
                text=scenario["texts"][i],
                confidence=scenario["confidences"][i],
                position=scenario["positions"][i]
            )
            ocr_results.append(result.__dict__)
        
        return {
            "success": True,
            "message": f"模拟OCR识别完成，识别到{len(ocr_results)}个文本",
            "texts": scenario["texts"],
            "confidences": scenario["confidences"],
            "positions": scenario["positions"],
            "results": ocr_results,
            "count": len(ocr_results)
        }
    
    async def _simulate_ocr_async(self) -> Dict[str, Any]:
        """异步模拟OCR识别"""
        await asyncio.sleep(0.4)  # 模拟OCR处理延迟
        return self._simulate_ocr()
    

    
    @classmethod
    def reset_instance(cls):
        """重置单例（主要用于测试）"""
        cls._instance = None
        cls._initialized = False
        print("🔄 ToolManager 单例已重置")


# ========== 工具函数封装 ==========

# 全局变量保存单例实例
_tool_manager_instance = None

def get_tool_manager(config: Optional[AgentConfig] = None, node: Optional[Node] = None) -> ToolManager:
    """
    获取ToolManager单例实例
    
    这是推荐的方式，在创建agent时调用此函数获取实例
    """
    global _tool_manager_instance
    if _tool_manager_instance is None:
        _tool_manager_instance = ToolManager(config, node)
    return _tool_manager_instance


# 创建工具函数（供LangGraph使用）
def create_tool_functions(tool_manager: Optional[ToolManager] = None) -> Dict[str, Any]:
    """
    创建工具函数字典，供智能体调用
    
    Args:
        tool_manager: ToolManager实例，如果为None则自动获取单例
    """
    tm = tool_manager or get_tool_manager()
    
    @tool
    def move_wrapper(velocity: Optional[float] = None, angle: float = 0.0, seconds: Optional[float] = None) -> str:
        """移动工具包装函数"""
        result = tm.move(velocity, angle, seconds)
        return json.dumps(result, ensure_ascii=False)
    
    @tool
    def yolo_detect_wrapper(threshold: Optional[float] = None) -> str:
        """YOLO检测工具包装函数"""
        result = tm.yolo_detect(threshold)
        return json.dumps(result, ensure_ascii=False)
    
    @tool
    def vlm_detect_wrapper() -> str:
        """VLM检测工具包装函数"""
        result = tm.vlm_detect()
        return json.dumps(result, ensure_ascii=False)
    
    @tool
    def track_wrapper(target: Optional[str] = None) -> str:
        """追踪工具包装函数"""
        result = tm.track(target)
        return json.dumps(result, ensure_ascii=False)
    
    @tool
    def check_obstacle_wrapper() -> str:
        """障碍物检测工具包装函数"""
        result = tm.check_obstacle()
        return json.dumps(result, ensure_ascii=False)
    
    @tool
    def ocr_wrapper() -> str:
        """OCR工具包装函数"""
        result = tm.ocr()
        return json.dumps(result, ensure_ascii=False)
    
    return {
        "move": move_wrapper,
        "yolo_detect": yolo_detect_wrapper,
        "VLM_detect": vlm_detect_wrapper,
        "track": track_wrapper,
        "check_obstacle": check_obstacle_wrapper,
        "ocr": ocr_wrapper
    }


# 异步工具函数（用于LangGraph的异步调用）
def create_async_tool_functions(tool_manager: Optional[ToolManager] = None) -> Dict[str, Any]:
    """
    创建异步工具函数字典，供智能体调用
    """
    tm = tool_manager or get_tool_manager()
    
    @tool
    async def move_wrapper(velocity: Optional[float] = None, angle: float = 0.0, seconds: Optional[float] = None) -> str:
        """异步移动工具包装函数"""
        result = await tm.move_async(velocity, angle, seconds)
        return json.dumps(result, ensure_ascii=False)
    
    @tool
    async def yolo_detect_wrapper(threshold: Optional[float] = None) -> str:
        """异步YOLO检测工具包装函数"""
        result = await tm.yolo_detect_async(threshold)
        return json.dumps(result, ensure_ascii=False)
    
    @tool
    async def vlm_detect_wrapper() -> str:
        """异步VLM检测工具包装函数"""
        result = await tm.vlm_detect_async()
        return json.dumps(result, ensure_ascii=False)
    
    @tool
    async def track_wrapper(target: Optional[str] = None) -> str:
        """异步追踪工具包装函数"""
        result = await tm.track_async(target)
        return json.dumps(result, ensure_ascii=False)
    
    @tool
    async def check_obstacle_wrapper() -> str:
        """障碍物检测工具包装函数"""
        result = await tm.check_obstacle_async()
        return json.dumps(result, ensure_ascii=False)
    
    @tool
    async def ocr_wrapper() -> str:
        """OCR工具包装函数"""
        result = await tm.ocr_async()
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
    
    config = AgentConfig()
    
    # 更新配置...
    if 'agent' in config_data:
        agent_config = config_data['agent']
        config.max_iterations = agent_config.get('max_iterations', config.max_iterations)
        config.use_reflection = agent_config.get('planning', {}).get('use_reflection', config.use_reflection)
    
    if 'tools' in config_data:
        tools_config = config_data['tools']
        if 'move' in tools_config:
            move_config = tools_config['move']
            config.default_move_velocity = move_config.get('default_velocity', config.default_move_velocity)
            config.default_move_seconds = move_config.get('default_seconds', config.default_move_seconds)
        
        if 'yolo_detect' in tools_config:
            yolo_config = tools_config['yolo_detect']
            config.yolo_threshold = yolo_config.get('default_threshold', config.yolo_threshold)
        
        if 'track' in tools_config:
            track_config = tools_config['track']
            config.tracking_distance = track_config.get('tracking_distance', config.tracking_distance)
    
    return config