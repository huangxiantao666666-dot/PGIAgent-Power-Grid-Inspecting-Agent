#!/usr/bin/env python3
"""
OCR文字识别节点 - 提供ocr()工具功能
识别图像中的文字，特别针对电力设备标签
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from PGIAgent.srv import OCR
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import base64
import re
from threading import Lock
import os
from collections import deque
import concurrent.futures
import yaml

# 尝试导入OCR库
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("警告: easyocr未安装，使用模拟模式")

try:
    import pytesseract
    from PIL import Image as PILImage
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("警告: pytesseract或PIL未安装")

# 尝试导入API客户端
try:
    import openai
    import httpx
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告: openai或httpx未安装，VLM OCR将不可用")
    

def load_config_with_env(config_path):
    with open(config_path, 'r') as f:
        config = f.read()
    
    # 替换环境变量
    for key, value in os.environ.items():
        placeholder = f"${{{key}}}"
        if placeholder in config:
            config = config.replace(placeholder, value)
    
    return yaml.safe_load(config)    

try:
    with open(r"config/tools_param.yaml", 'r', encoding='utf-8') as file:
        config = load_config_with_env(r"config/tools_param.yaml")
    print(f"成功加载配置文件: {r'config/tools_param.yaml'}")
except FileNotFoundError:
    print(f"错误: 文件 {r'config/tools_param.yaml'} 不存在")


class OCRNode(Node):
    """OCR文字识别节点，提供文字识别服务"""
    
    def __init__(self):
        super().__init__('ocr_node')
        
        # 从配置文件读取参数
        ocr_params = config.get('ocr_node', {}).get('ros__parameters', {})
        
        # 声明参数（使用yaml中的值作为默认值）
        self.declare_parameters(
            namespace='',
            parameters=[
                ('rgb_topic', ocr_params.get('rgb_topic', '/depth_cam/rgb/image_raw')),
                ('service_name', ocr_params.get('service_name', '/pgi_agent/ocr')),
                ('ocr_engine', ocr_params.get('ocr_engine', 'easyocr')),
                ('languages', ocr_params.get('languages', ['en', 'ch_sim'])),
                ('confidence_threshold', ocr_params.get('confidence_threshold', 0.5)),
                ('preprocess_enabled', ocr_params.get('preprocess_enabled', True)),
                ('resize_factor', ocr_params.get('resize_factor', 2.0)),
                ('denoise', ocr_params.get('denoise', True)),
                ('adaptive_threshold', ocr_params.get('adaptive_threshold', True)),
                ('roi_enabled', ocr_params.get('roi_enabled', False)),
                ('roi_x', ocr_params.get('roi_x', 0.3)),
                ('roi_y', ocr_params.get('roi_y', 0.3)),
                ('roi_width', ocr_params.get('roi_width', 0.4)),
                ('roi_height', ocr_params.get('roi_height', 0.4)),
                ('tesseract_path', ocr_params.get('tesseract_path', '')),
                ('gpu', ocr_params.get('gpu', True)),
                ('batch_size', ocr_params.get('batch_size', 1)),
                ('max_text_length', ocr_params.get('max_text_length', 1000)),
                ('use_simulation', ocr_params.get('use_simulation', False)),
                # VLM OCR 参数
                ('vlm_provider', ocr_params.get('vlm_provider', 'qwen')),
                ('vlm_api_key', ocr_params.get('vlm_api_key', '')),
                ('vlm_base_url', ocr_params.get('vlm_base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')),
                ('vlm_model', ocr_params.get('vlm_model', 'qwen-vl-flash-2026-01-22')),
                ('vlm_ocr_prompt', ocr_params.get('vlm_ocr_prompt', '请识别图中所有的文字内容，包括仪表读数、标签、设备编号、安全警示等电力设备相关文字。请直接列出识别到的文字内容。')),
                ('vlm_max_tokens', ocr_params.get('vlm_max_tokens', 1000)),
                ('vlm_temperature', ocr_params.get('vlm_temperature', 0.1)),
                ('vlm_timeout', ocr_params.get('vlm_timeout', 30)),
                ('vlm_image_quality', ocr_params.get('vlm_image_quality', 'high')),
                ('vlm_max_image_size', ocr_params.get('vlm_max_image_size', 1024)),
            ]
        )
        
        # 获取参数
        self.camera_topic = self.get_parameter('rgb_topic').value
        self.service_name = self.get_parameter('service_name').value
        self.ocr_engine = self.get_parameter('ocr_engine').value
        self.languages = self.get_parameter('languages').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.preprocess_enabled = self.get_parameter('preprocess_enabled').value
        self.resize_factor = self.get_parameter('resize_factor').value
        self.denoise = self.get_parameter('denoise').value
        self.adaptive_threshold = self.get_parameter('adaptive_threshold').value
        self.roi_enabled = self.get_parameter('roi_enabled').value
        self.roi_x = self.get_parameter('roi_x').value
        self.roi_y = self.get_parameter('roi_y').value
        self.roi_width = self.get_parameter('roi_width').value
        self.roi_height = self.get_parameter('roi_height').value
        self.use_simulation = self.get_parameter('use_simulation').value
        self.tesseract_path = self.get_parameter('tesseract_path').value
        self.gpu = self.get_parameter('gpu').value
        self.batch_size = self.get_parameter('batch_size').value
        self.max_text_length = self.get_parameter('max_text_length').value
        
        # VLM OCR 参数
        self.vlm_provider = self.get_parameter('vlm_provider').value
        self.vlm_api_key = self.get_parameter('vlm_api_key').value
        self.vlm_base_url = self.get_parameter('vlm_base_url').value
        self.vlm_model = self.get_parameter('vlm_model').value
        self.vlm_ocr_prompt = self.get_parameter('vlm_ocr_prompt').value
        self.vlm_max_tokens = self.get_parameter('vlm_max_tokens').value
        self.vlm_temperature = self.get_parameter('vlm_temperature').value
        self.vlm_timeout = self.get_parameter('vlm_timeout').value
        self.vlm_image_quality = self.get_parameter('vlm_image_quality').value
        self.vlm_max_image_size = self.get_parameter('vlm_max_image_size').value
        
        # 检查OCR引擎可用性
        if self.ocr_engine == 'easyocr' and not EASYOCR_AVAILABLE:
            self.get_logger().warning("easyocr不可用，切换到模拟模式")
            self.use_simulation = True
        elif self.ocr_engine == 'tesseract' and not TESSERACT_AVAILABLE:
            self.get_logger().warning("tesseract不可用，切换到模拟模式")
            self.use_simulation = True
        elif self.ocr_engine == 'vlm' and not OPENAI_AVAILABLE:
            self.get_logger().warning("openai/httpx不可用，切换到模拟模式")
            self.use_simulation = True
            
        
        # 新增：性能统计
        self.stats = {
            'total_requests': 0,
            'total_time': 0.0,
            'max_time': 0.0,
            'success_count': 0
        }
        
        # 新增：图像缓存
        self.frame_queue = deque(maxlen=3)
        
        # 新增：定期性能报告
        self.stats_timer = self.create_timer(60.0, self._report_stats)
        
        # 初始化工具
        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_lock = Lock()
        
        # 初始化OCR引擎
        self.reader = None
        if not self.use_simulation:
            self._initialize_ocr_engine()
        
        # 创建订阅器
        self.image_sub = self.create_subscription(
            Image, self.camera_topic, self.image_callback, 10
        )
        
        # 创建服务
        self.ocr_service = self.create_service(
            OCR,
            self.service_name,
            self.handle_ocr_request,
            callback_group=ReentrantCallbackGroup()
        )
        
        # 电力设备关键词（用于过滤和验证）
        self.power_keywords = [
            '变压器', '断路器', '开关', '电缆', '绝缘子',
            '电压', '电流', '功率', '频率', '相位',
            'kV', 'kVA', 'kW', 'Hz', 'A', 'V',
            '危险', '高压', '禁止', '注意', '安全'
        ]
        
        self.get_logger().info(f"OCR节点已启动，服务: {self.service_name}")
        self.get_logger().info(f"OCR引擎: {self.ocr_engine}, 语言: {self.languages}")
        if self.use_simulation:
            self.get_logger().info("运行在模拟模式")
    
    def _initialize_ocr_engine(self):
        """初始化OCR引擎"""
        try:
            if self.ocr_engine == 'easyocr':
                self.get_logger().info(f"初始化easyocr，语言: {self.languages}")
                self.reader = easyocr.Reader(
                    self.languages,
                    gpu=True,  # 使用GPU加速
                    model_storage_directory='./models/easyocr'
                )
                self.get_logger().info("easyocr初始化成功")
                
            elif self.ocr_engine == 'tesseract':
                if self.tesseract_path:
                    pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
                
                # 设置Tesseract参数
                self.tesseract_config = '--oem 3 --psm 6'
                if 'ch_sim' in self.languages:
                    self.tesseract_config += ' -l chi_sim+eng'
                else:
                    self.tesseract_config += ' -l eng'
                
                self.get_logger().info("tesseract初始化成功")
                
            elif self.ocr_engine == 'vlm':
                # 初始化VLM客户端
                self.get_logger().info(f"初始化VLM OCR，提供商: {self.vlm_provider}")
                import openai
                import httpx
                self.vlm_client = openai.OpenAI(
                    api_key=self.vlm_api_key,
                    base_url=self.vlm_base_url,
                    timeout=httpx.Timeout(self.vlm_timeout)
                )
                self.get_logger().info(f"VLM OCR初始化成功，模型: {self.vlm_model}")
                
        except Exception as e:
            self.get_logger().error(f"OCR引擎初始化失败: {e}")
            self.use_simulation = True
            self.get_logger().info("切换到模拟模式")
    
    def image_callback(self, msg):
        """图像回调"""
        try:
            with self.frame_lock:
                self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"图像处理错误: {e}")
    
    def handle_ocr_request(self, request, response):
        """处理OCR识别请求"""
        try:
            # 检查是否有最新图像
            with self.frame_lock:
                if self.latest_frame is None:
                    response.success = False
                    response.message = "未收到相机图像"
                    response.texts = []
                    response.confidences = []
                    response.positions = []
                    return response  # ✅ 必须显式返回
                
                frame = self.latest_frame.copy()
            
            # 执行OCR识别
            if self.use_simulation:
                ocr_result = self._simulate_ocr(frame)
            else:
                ocr_result = self._perform_ocr(frame)
            
            # 过滤和验证结果
            # filtered_result = self._filter_ocr_results(ocr_result)
            filtered_result = ocr_result
            
            # 填充响应
            response.success = True
            response.message = f"识别到 {len(filtered_result['texts'])} 个文本"
            response.texts = filtered_result['texts']
            response.confidences = filtered_result['confidences']
            response.positions = filtered_result['positions']
            
            self.get_logger().info(
                f"OCR识别完成: {response.message}"
            )
            
            return response  # ✅ 成功时返回
        
        except Exception as e:
            self.get_logger().error(f"OCR识别过程中发生错误: {e}")
            response.success = False
            response.message = f"识别失败: {str(e)}"
            response.texts = []
            response.confidences = []
            response.positions = []
            
            return response  # ✅ 失败时也返回
    
    def _perform_ocr_with_timeout(self, frame, timeout=5.0):
        """带超时的OCR执行"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self._perform_ocr, frame)
            return future.result(timeout=timeout)
    
    def _report_stats(self):
        """定期报告性能统计"""
        if self.stats['total_requests'] > 0:
            avg_time = self.stats['total_time'] / self.stats['total_requests']
            success_rate = self.stats['success_count'] / self.stats['total_requests']
            
            self.get_logger().info(
                f"OCR性能统计: "
                f"请求={self.stats['total_requests']}, "
                f"成功率={success_rate:.1%}, "
                f"平均耗时={avg_time:.2f}s, "
                f"最大耗时={self.stats['max_time']:.2f}s"
            )
    
    def _perform_ocr(self, frame):
        """执行实际的OCR识别"""
        # 预处理图像
        processed_frame = self._preprocess_image(frame)
        
        # 提取ROI（如果启用）
        if self.roi_enabled:
            roi_frame = self._extract_roi(processed_frame)
        else:
            roi_frame = processed_frame
        
        # 执行OCR识别
        if self.ocr_engine == 'easyocr':
            return self._perform_easyocr(roi_frame, frame.shape)
        elif self.ocr_engine == 'tesseract':
            return self._perform_tesseract(roi_frame, frame.shape)
        elif self.ocr_engine == 'vlm':
            return self._perform_vlm_ocr(frame)
        else:
            return self._simulate_ocr(frame)
    
    def _perform_vlm_ocr(self, frame):
        """使用VLM进行OCR识别"""
        # 预处理图像
        processed_frame = self._preprocess_vlm_image(frame)
        
        # 编码图像为base64
        image_base64 = self._encode_image_to_base64(processed_frame)
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.vlm_ocr_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": self.vlm_image_quality
                        }
                    }
                ]
            }
        ]
        
        try:
            # 调用VLM API
            api_response = self.vlm_client.chat.completions.create(
                model=self.vlm_model,
                messages=messages,
                max_tokens=self.vlm_max_tokens,
                temperature=self.vlm_temperature
            )
            
            # 解析响应
            ocr_text = api_response.choices[0].message.content
            
            self.get_logger().info(f"VLM OCR识别完成: {ocr_text[:100]}...")
            
            # 解析VLM响应，提取文本和位置信息
            return self._parse_vlm_ocr_response(ocr_text, frame.shape)
            
        except Exception as e:
            self.get_logger().error(f"VLM OCR调用失败: {e}")
            return {
                'texts': [],
                'confidences': [],
                'positions': []
            }
    
    def _preprocess_vlm_image(self, frame):
        """VLM图像预处理"""
        h, w = frame.shape[:2]
        
        # 调整图像大小
        if max(h, w) > self.vlm_max_image_size:
            scale = self.vlm_max_image_size / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 根据质量设置调整
        if self.vlm_image_quality == 'low':
            frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
        
        return frame
    
    def _encode_image_to_base64(self, frame):
        """将图像编码为base64"""
        # 编码为JPEG
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise ValueError("图像编码失败")
        
        # 转换为base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    
    def _parse_vlm_ocr_response(self, ocr_text, original_shape):
        """解析VLM OCR响应，提取文本列表"""
        texts = []
        confidences = []
        positions = []
        
        h, w = original_shape[:2]
        
        # VLM返回的是文本描述，需要按行或按项目符号分割
        # 常见格式：
        # 1. 文本1
        # 2. 文本2
        # 或者：
        # 文本1, 文本2, 文本3
        
        # 尝试按行分割
        lines = ocr_text.replace('\n', ' ').split('.')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 移除可能的编号（1., 2., - 等）
            line = re.sub(r'^[0-9]+[\.\)]\s*', '', line)
            line = re.sub(r'^[-•]\s*', '', line)
            
            if len(line) > 1:  # 过滤单字符
                texts.append(line[:self.max_text_length])
                # VLM不返回置信度，使用默认值
                confidences.append(0.8)
                # VLM不返回位置信息，使用默认位置
                positions.append("中间")
        
        return {
            'texts': texts,
            'confidences': confidences,
            'positions': positions
        }
    
    def _preprocess_image(self, frame):
        """预处理图像"""
        processed = frame.copy()
        
        if self.preprocess_enabled:
            # 转换为灰度图
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            
            # 应用自适应阈值
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # 降噪
            denoised = cv2.medianBlur(binary, 3)
            
            # 转换为BGR格式返回
            processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        
        return processed
    
    def _extract_roi(self, frame):
        """提取感兴趣区域"""
        h, w = frame.shape[:2]
        
        roi_x = int(w * self.roi_x)
        roi_y = int(h * self.roi_y)
        roi_w = int(w * self.roi_width)
        roi_h = int(h * self.roi_height)
        
        # 确保ROI在图像范围内
        roi_x = max(0, min(roi_x, w - 1))
        roi_y = max(0, min(roi_y, h - 1))
        roi_w = min(roi_w, w - roi_x)
        roi_h = min(roi_h, h - roi_y)
        
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        self.get_logger().debug(f"提取ROI: ({roi_x},{roi_y}) {roi_w}x{roi_h}")
        
        return roi
    
    def _perform_easyocr(self, frame, original_shape):
        """使用easyocr进行识别"""
        h, w = frame.shape[:2]
        
        # 执行OCR
        results = self.reader.readtext(
            frame,
            detail=1,
            paragraph=False,
            min_size=20,
            text_threshold=self.confidence_threshold
        )
        
        texts = []
        confidences = []
        positions = []
        
        for result in results:
            bbox, text, confidence = result
            
            # 过滤低置信度结果
            if confidence < self.confidence_threshold:
                continue
            
            # 过滤过长的文本
            if len(text) > self.max_text_length:
                continue
            
            # 计算边界框中心位置（相对于原始图像）
            if self.roi_enabled:
                # 需要将ROI坐标转换回原始图像坐标
                original_h, original_w = original_shape[:2]
                roi_x = int(original_w * self.roi_x)
                roi_y = int(original_h * self.roi_y)
                
                # 转换边界框坐标
                bbox_array = np.array(bbox)
                bbox_array[:, 0] += roi_x  # x坐标
                bbox_array[:, 1] += roi_y  # y坐标
                
                # 计算中心点
                center_x = np.mean(bbox_array[:, 0])
                center_y = np.mean(bbox_array[:, 1])
            else:
                # 直接计算中心点
                bbox_array = np.array(bbox)
                center_x = np.mean(bbox_array[:, 0])
                center_y = np.mean(bbox_array[:, 1])
            
            # 描述位置
            position = self._describe_position(center_x, center_y, w, h)
            
            texts.append(text)
            confidences.append(float(confidence))
            positions.append(position)
            
            self.get_logger().debug(
                f"识别到文本: '{text}' (置信度: {confidence:.2f}), "
                f"位置: {position}"
            )
        
        return {
            'texts': texts,
            'confidences': confidences,
            'positions': positions
        }
    
    def _perform_tesseract(self, frame, original_shape):
        """使用tesseract进行识别"""
        # 转换为PIL图像
        pil_image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 执行OCR
        data = pytesseract.image_to_data(
            pil_image,
            config=self.tesseract_config,
            output_type=pytesseract.Output.DICT
        )
        
        texts = []
        confidences = []
        positions = []
        
        h, w = frame.shape[:2]
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            confidence = float(data['conf'][i]) / 100.0  # 转换为0-1范围
            
            # 过滤空文本和低置信度
            if not text or confidence < self.confidence_threshold:
                continue
            
            # 过滤过长的文本
            if len(text) > self.max_text_length:
                continue
            
            # 获取边界框
            x = data['left'][i]
            y = data['top'][i]
            width = data['width'][i]
            height = data['height'][i]
            
            # 计算中心点
            center_x = x + width / 2
            center_y = y + height / 2
            
            # 描述位置
            position = self._describe_position(center_x, center_y, w, h)
            
            texts.append(text)
            confidences.append(confidence)
            positions.append(position)
            
            self.get_logger().debug(
                f"识别到文本: '{text}' (置信度: {confidence:.2f}), "
                f"位置: {position}"
            )
        
        return {
            'texts': texts,
            'confidences': confidences,
            'positions': positions
        }
    
    def _simulate_ocr(self, frame):
        """模拟OCR识别（用于测试）"""
        h, w = frame.shape[:2]
        
        # 模拟电力设备标签
        simulated_texts = [
            "变压器: 110kV/10kV",
            "电流: 150A",
            "功率: 5000kW",
            "危险! 高压",
            "禁止靠近"
        ]
        
        simulated_confidences = [0.85, 0.78, 0.92, 0.95, 0.88]
        
        # 生成随机位置
        positions = []
        for _ in range(len(simulated_texts)):
            center_x = np.random.randint(w // 4, 3 * w // 4)
            center_y = np.random.randint(h // 4, 3 * h // 4)
            positions.append(self._describe_position(center_x, center_y, w, h))
        
        self.get_logger().info(f"模拟OCR识别: {len(simulated_texts)} 个文本")
        
        return {
            'texts': simulated_texts,
            'confidences': simulated_confidences,
            'positions': positions
        }
    
    def _describe_position(self, x, y, img_width, img_height):
        """描述文本在图像中的位置"""
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
    
    def _filter_ocr_results(self, ocr_result):
        """过滤和验证OCR结果"""
        filtered_texts = []
        filtered_confidences = []
        filtered_positions = []
        
        # 检查ocr_result是否包含必要的键
        if not all(key in ocr_result for key in ['texts', 'confidences', 'positions']):
            self.get_logger().warn("OCR结果格式不正确")
            return {
                'texts': filtered_texts,
                'confidences': filtered_confidences,
                'positions': filtered_positions
            }
        
        for text, confidence, position in zip(
            ocr_result['texts'],
            ocr_result['confidences'],
            ocr_result['positions']
        ):
            # 基础过滤：置信度阈值
            if confidence < self.confidence_threshold:
                continue
            
            # 文本长度检查
            if len(text) > self.max_text_length:
                continue
            
            # 空文本检查
            if not text or text.isspace():
                continue
            
            # # 检查是否包含电力设备关键词
            # contains_keyword = any(
            #     keyword in text for keyword in self.power_keywords
            # )
            
            # # 检查是否包含数字和单位（电力设备常见）
            # has_number = any(char.isdigit() for char in text)
            # has_unit = any(unit in text for unit in ['kV', 'V', 'A', 'W', 'Hz', 'kVA', 'kW'])
            
            # # 保留条件：
            # # 1. 包含电力关键词
            # # 2. 或包含数字和单位（可能是仪表读数）
            # # 3. 或置信度非常高（>0.9）
                
            filtered_texts.append(text)
            filtered_confidences.append(confidence)
            filtered_positions.append(position)
        
        return {
            'texts': filtered_texts,
            'confidences': filtered_confidences,
            'positions': filtered_positions
        }

def main(args=None):
    """
    OCR节点的主函数
    初始化ROS2，创建节点，使用多线程执行器运行
    """
    # 初始化ROS2
    rclpy.init(args=args)
    
    node = None
    executor = None
    
    try:
        # 创建OCR节点
        node = OCRNode()
        
        # 使用多线程执行器支持并发服务调用
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        
        node.get_logger().info("OCR节点已启动，等待请求...")
        
        # 开始事件循环
        executor.spin()
        
    except KeyboardInterrupt:
        # 用户按Ctrl+C时的优雅退出
        if node:
            node.get_logger().info("用户中断，正在关闭OCR节点...")
        
    except Exception as e:
        # 其他异常
        if node:
            node.get_logger().error(f"OCR节点运行出错: {e}")
        else:
            print(f"OCR节点创建失败: {e}")
        
    finally:
        # 清理资源
        if executor:
            executor.shutdown()
        
        if node:
            node.destroy_node()
        
        # 关闭ROS2
        rclpy.shutdown()
        
        if node:
            node.get_logger().info("OCR节点已关闭")


if __name__ == '__main__':
    main()