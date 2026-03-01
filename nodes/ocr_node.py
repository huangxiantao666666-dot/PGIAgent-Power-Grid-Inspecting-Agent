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
from pgi_agent_msgs.srv import OCR, OCRResponse
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from threading import Lock
import os

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


class OCRNode(Node):
    """OCR文字识别节点，提供文字识别服务"""
    
    def __init__(self):
        super().__init__('ocr_node')
        
        # 声明参数
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_topic', '/depth_cam/rgb/image_raw'),
                ('service_name', '/pgi_agent/ocr'),
                ('ocr_engine', 'easyocr'),  # easyocr, tesseract, simulation
                ('languages', ['ch_sim', 'en']),
                ('confidence_threshold', 0.5),
                ('preprocess_enabled', True),
                ('roi_enabled', False),
                ('roi_x', 0.3),
                ('roi_y', 0.3),
                ('roi_width', 0.4),
                ('roi_height', 0.4),
                ('use_simulation', False),
                ('tesseract_path', ''),
                ('max_text_length', 1000),
            ]
        )
        
        # 获取参数
        self.camera_topic = self.get_parameter('camera_topic').value
        self.service_name = self.get_parameter('service_name').value
        self.ocr_engine = self.get_parameter('ocr_engine').value
        self.languages = self.get_parameter('languages').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.preprocess_enabled = self.get_parameter('preprocess_enabled').value
        self.roi_enabled = self.get_parameter('roi_enabled').value
        self.roi_x = self.get_parameter('roi_x').value
        self.roi_y = self.get_parameter('roi_y').value
        self.roi_width = self.get_parameter('roi_width').value
        self.roi_height = self.get_parameter('roi_height').value
        self.use_simulation = self.get_parameter('use_simulation').value
        self.tesseract_path = self.get_parameter('tesseract_path').value
        self.max_text_length = self.get_parameter('max_text_length').value
        
        # 检查OCR引擎可用性
        if self.ocr_engine == 'easyocr' and not EASYOCR_AVAILABLE:
            self.get_logger().warning("easyocr不可用，切换到模拟模式")
            self.use_simulation = True
        elif self.ocr_engine == 'tesseract' and not TESSERACT_AVAILABLE:
            self.get_logger().warning("tesseract不可用，切换到模拟模式")
            self.use_simulation = True
        
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
                    return response
                
                frame = self.latest_frame.copy()
            
            # 执行OCR识别
            if self.use_simulation:
                ocr_result = self._simulate_ocr(frame)
            else:
                ocr_result = self._perform_ocr(frame)
            
            # 过滤和验证结果
            filtered_result = self._filter_ocr_results(ocr_result)
            
            # 填充响应
            response.success = True
            response.message = f"识别到 {len(filtered_result['texts'])} 个文本"
            response.texts = filtered_result['texts']
            response.confidences = filtered_result['confidences']
            response.positions = filtered_result['positions']
            response.engine = self.ocr_engine
            
            self.get_logger().info(
                f"OCR识别完成: {response.message}, "
                f"引擎: {response.engine}"
            )
            
        except Exception as e:
            self.get_logger().error(f"OCR识别过程中发生错误: {e}")
            response.success = False
            response.message = f"识别失败: {str(e)}"
            response.texts = []
            response.confidences = []
            response.positions = []
        
        return response
    
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
        else:
            return self._simulate_ocr(frame)
    
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
        
        for text, confidence, position in zip(
            ocr_result['texts'],
            ocr_result['confidences'],
            ocr_result['positions']
        ):
            # 检查是否包含电力设备关键词
            contains_keyword = any(
                keyword in text for keyword in self.power_keywords
            )
            
            # 如果包含关键词或置信度足够高，则保留
            if contains_keyword or confidence >= self.confidence_threshold:
                filtered_texts.append(text)
                filtered_confidences.append(confidence)
                filtered_positions.append(position)
        
        return {
            'texts': filtered_texts,
            'confidences': filtered_confidences,
            'positions': filtered_positions
        }
    
    def destroy_node(self):
        """清理资源"""
        if self.reader is not None:
            # 清理OCR资源
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = OCRNode()
        
        # 使用多线程执行器
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            node.get_logger().info("OCR节点正在关闭...")
        finally:
            executor.shutdown()
            node.destroy_node()
            
    except Exception as e:
        print(f"OCR节点启动失败: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()