#!/usr/bin/env python3
"""
视觉大模型节点 - 提供VLM_detect()工具功能
订阅相机图像，调用视觉大模型API分析场景
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from pgi_agent_msgs.srv import VLMDetect, VLMDetectResponse
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import base64
import json
from threading import Lock
import os
from enum import Enum

# 尝试导入API客户端
try:
    import openai
    import httpx
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告: openai或httpx未安装，使用模拟模式")


class VLMProvider(Enum):
    """视觉大模型提供商"""
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    LOCAL = "local"
    SIMULATION = "simulation"


class VLMNode(Node):
    """视觉大模型节点，提供场景分析服务"""
    
    def __init__(self):
        super().__init__('vlm_node')
        
        # 声明参数
        self.declare_parameters(
            namespace='',
            parameters=[
                ('provider', 'qwen'),
                ('model', 'qwen-vl-max'),
                ('api_key', ''),
                ('base_url', ''),
                ('camera_topic', '/depth_cam/rgb/image_raw'),
                ('service_name', '/pgi_agent/vlm_detect'),
                ('use_simulation', False),
                ('max_tokens', 1000),
                ('temperature', 0.1),
                ('timeout', 30),
                ('image_quality', 'high'),  # high, low
                ('max_image_size', 1024),
            ]
        )
        
        # 获取参数
        self.provider = self.get_parameter('provider').value
        self.model = self.get_parameter('model').value
        self.api_key = self.get_parameter('api_key').value
        self.base_url = self.get_parameter('base_url').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.service_name = self.get_parameter('service_name').value
        self.use_simulation = self.get_parameter('use_simulation').value or not OPENAI_AVAILABLE
        self.max_tokens = self.get_parameter('max_tokens').value
        self.temperature = self.get_parameter('temperature').value
        self.timeout = self.get_parameter('timeout').value
        self.image_quality = self.get_parameter('image_quality').value
        self.max_image_size = self.get_parameter('max_image_size').value
        
        # 初始化工具
        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_lock = Lock()
        
        # 初始化API客户端
        self.client = None
        if not self.use_simulation and OPENAI_AVAILABLE:
            self._initialize_client()
        
        # 创建订阅器
        self.image_sub = self.create_subscription(
            Image, self.camera_topic, self.image_callback, 10
        )
        
        # 创建服务
        self.vlm_service = self.create_service(
            VLMDetect,
            self.service_name,
            self.handle_vlm_request,
            callback_group=ReentrantCallbackGroup()
        )
        
        # 状态变量
        self.last_request_time = 0
        self.request_count = 0
        
        self.get_logger().info(f"VLM节点已启动，服务: {self.service_name}")
        self.get_logger().info(f"提供商: {self.provider}, 模型: {self.model}")
        if self.use_simulation:
            self.get_logger().info("运行在模拟模式")
    
    def _initialize_client(self):
        """初始化API客户端"""
        try:
            if self.provider in ['openai', 'deepseek', 'qwen']:
                # 使用OpenAI兼容的API
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url if self.base_url else None,
                    timeout=httpx.Timeout(self.timeout)
                )
                self.get_logger().info(f"API客户端初始化成功: {self.provider}")
            else:
                self.get_logger().warning(f"不支持的提供商: {self.provider}")
                self.use_simulation = True
                
        except Exception as e:
            self.get_logger().error(f"API客户端初始化失败: {e}")
            self.use_simulation = True
            self.get_logger().info("切换到模拟模式")
    
    def image_callback(self, msg):
        """图像回调"""
        try:
            with self.frame_lock:
                self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"图像处理错误: {e}")
    
    def handle_vlm_request(self, request, response):
        """处理VLM分析请求"""
        try:
            # 检查是否有最新图像
            with self.frame_lock:
                if self.latest_frame is None:
                    response.success = False
                    response.message = "未收到相机图像"
                    response.description = ""
                    return response
                
                frame = self.latest_frame.copy()
            
            # 执行VLM分析
            if self.use_simulation:
                analysis_result = self._simulate_vlm_analysis(frame)
            else:
                analysis_result = self._perform_vlm_analysis(frame)
            
            # 填充响应
            response.success = True
            response.message = "场景分析完成"
            response.description = analysis_result['description']
            response.detailed_analysis = analysis_result.get('detailed_analysis', '')
            response.objects_detected = analysis_result.get('objects', [])
            response.scene_type = analysis_result.get('scene_type', 'unknown')
            
            self.get_logger().info(f"VLM分析完成: {response.scene_type}")
            
        except Exception as e:
            self.get_logger().error(f"VLM分析过程中发生错误: {e}")
            response.success = False
            response.message = f"分析失败: {str(e)}"
            response.description = ""
        
        return response
    
    def _perform_vlm_analysis(self, frame):
        """执行实际的VLM分析"""
        # 预处理图像
        processed_frame = self._preprocess_image(frame)
        
        # 编码图像为base64
        image_base64 = self._encode_image_to_base64(processed_frame)
        
        # 构建提示词
        prompt = self._build_vlm_prompt()
        
        # 调用API
        try:
            self.request_count += 1
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            self.last_request_time = current_time
            
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": self.image_quality
                            }
                        }
                    ]
                }
            ]
            
            # 调用API
            api_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # 解析响应
            analysis_text = api_response.choices[0].message.content
            
            # 提取结构化信息
            structured_result = self._parse_vlm_response(analysis_text)
            structured_result['description'] = analysis_text
            
            self.get_logger().info(f"API调用成功，耗时: {time_since_last:.2f}s")
            
            return structured_result
            
        except Exception as e:
            self.get_logger().error(f"API调用失败: {e}")
            raise
    
    def _simulate_vlm_analysis(self, frame):
        """模拟VLM分析（用于测试）"""
        h, w = frame.shape[:2]
        
        # 生成模拟分析结果
        scene_types = [
            "变电站设备区", "控制室", "户外电力设施", 
            "设备维护现场", "安全通道", "设备存储区"
        ]
        
        objects_list = [
            ["变压器", "断路器", "电缆", "绝缘子"],
            ["控制面板", "显示器", "开关", "指示灯"],
            ["电杆", "输电线路", "避雷器", "接地装置"],
            ["维护人员", "工具车", "安全帽", "工具箱"],
            ["安全标志", "通道线", "应急灯", "灭火器"],
            ["备用设备", "电缆盘", "绝缘材料", "防护装备"]
        ]
        
        # 随机选择场景
        import random
        scene_idx = random.randint(0, len(scene_types) - 1)
        scene_type = scene_types[scene_idx]
        objects = objects_list[scene_idx]
        
        # 生成详细描述
        descriptions = [
            f"这是一个{scene_type}，可以看到{', '.join(objects)}等设备。",
            f"场景分析：{scene_type}，主要设备包括{', '.join(objects[:3])}等。",
            f"当前处于{scene_type}，现场有{len(objects)}个主要物体：{', '.join(objects)}。"
        ]
        
        description = descriptions[random.randint(0, len(descriptions) - 1)]
        
        # 添加更多细节
        details = [
            "设备状态正常，无异常情况。",
            "部分设备正在运行，指示灯正常。",
            "现场整洁，安全设施完备。",
            "有维护人员在作业，请注意安全距离。"
        ]
        
        detailed_analysis = details[random.randint(0, len(details) - 1)]
        
        self.get_logger().info(f"模拟VLM分析: {scene_type}")
        
        return {
            'description': description,
            'detailed_analysis': detailed_analysis,
            'objects': objects,
            'scene_type': scene_type
        }
    
    def _preprocess_image(self, frame):
        """预处理图像"""
        h, w = frame.shape[:2]
        
        # 调整图像大小
        if max(h, w) > self.max_image_size:
            scale = self.max_image_size / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 根据质量设置调整
        if self.image_quality == 'low':
            # 进一步压缩
            frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
        
        return frame
    
    def _encode_image_to_base64(self, frame):
        """将图像编码为base64"""
        # 编码为JPEG
        success, buffer = cv2.imencode('.jpg', frame, 
                                      [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise ValueError("图像编码失败")
        
        # 转换为base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    
    def _build_vlm_prompt(self):
        """构建VLM提示词"""
        prompt = """你是一个电力巡检机器人，请分析当前场景：
1. 描述场景中的主要物体和设备
2. 识别电力设备类型和状态
3. 注意安全相关元素（警告标志、安全设备等）
4. 评估现场安全状况
5. 提供巡检建议

请用中文回答，结构清晰。"""
        
        return prompt
    
    def _parse_vlm_response(self, response_text):
        """解析VLM响应，提取结构化信息"""
        # 尝试提取对象列表
        objects = []
        scene_type = "unknown"
        
        # 简单的关键词匹配（实际应用中可以使用更复杂的方法）
        keywords = {
            '变压器': ['变压器', '变电器', 'transformer'],
            '断路器': ['断路器', '开关', 'breaker'],
            '电缆': ['电缆', '电线', 'cable'],
            '控制面板': ['控制面板', '控制台', 'control panel'],
            '维护人员': ['人员', '工人', '维护', 'personnel'],
            '安全标志': ['标志', '警告', '安全', 'sign']
        }
        
        for obj, keywords_list in keywords.items():
            for keyword in keywords_list:
                if keyword in response_text.lower():
                    objects.append(obj)
                    break
        
        # 确定场景类型
        scene_keywords = {
            '变电站': ['变电站', '变电所', 'substation'],
            '控制室': ['控制室', '控制中心', 'control room'],
            '户外': ['户外', '室外', '野外', 'outdoor'],
            '设备区': ['设备区', '设备间', 'equipment area']
        }
        
        for scene, scene_keys in scene_keywords.items():
            for key in scene_keys:
                if key in response_text.lower():
                    scene_type = scene
                    break
        
        return {
            'objects': list(set(objects)),  # 去重
            'scene_type': scene_type,
            'detailed_analysis': response_text  # 保留原始文本作为详细分析
        }
    
    def destroy_node(self):
        """清理资源"""
        if self.client is not None:
            # 清理客户端资源
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = VLMNode()
        
        # 使用多线程执行器
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            node.get_logger().info("VLM节点正在关闭...")
        finally:
            executor.shutdown()
            node.destroy_node()
            
    except Exception as e:
        print(f"VLM节点启动失败: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()