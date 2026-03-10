#!/usr/bin/env python3
"""
综合测试脚本 - 测试所有6个服务节点
模拟agent/tools.py中的调用方式，验证服务节点是否正常工作
"""

import rclpy
from rclpy.node import Node
from rclpy.client import Client
import time
import json
from typing import Dict, Any, List, Optional
import yaml
import os

# 导入所有服务类型
from PGIAgent.srv import (
    MoveCommand, YOLODetect, VLMDetect, 
    Track, CheckObstacle, OCR
)

class ServiceTestClient(Node):
    """服务测试客户端"""
    
    def __init__(self):
        super().__init__('service_test_client')
        
        # 服务映射
        self.services = {
            'move': {'name': '/pgi_agent/move', 'type': MoveCommand},
            'yolo_detect': {'name': '/pgi_agent/yolo_detect', 'type': YOLODetect},
            'vlm_detect': {'name': '/pgi_agent/vlm_detect', 'type': VLMDetect},
            'track': {'name': '/pgi_agent/track', 'type': Track},
            'check_obstacle': {'name': '/pgi_agent/check_obstacle', 'type': CheckObstacle},
            'ocr': {'name': '/pgi_agent/ocr', 'type': OCR},
        }
        
        # 客户端字典
        self.clients = {}
        
        # 测试结果
        self.test_results = {}
        
        # 加载配置
        self.load_config()
        
    def load_config(self):
        """加载配置文件"""
        try:
            config_path = "config/tools_param.yaml"
            with open(config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
            self.get_logger().info(f"成功加载配置文件: {config_path}")
        except FileNotFoundError:
            self.get_logger().error(f"配置文件不存在，使用默认配置")
            self.config = {}
    
    def initialize_clients(self):
        """初始化所有服务客户端"""
        self.get_logger().info("正在初始化服务客户端...")
        
        for service_name, service_info in self.services.items():
            try:
                client = self.create_client(
                    service_info['type'], 
                    service_info['name']
                )
                self.clients[service_name] = client
                
                # 等待服务可用
                timeout = 5.0
                start_time = time.time()
                while not client.wait_for_service(timeout_sec=0.5):
                    if time.time() - start_time > timeout:
                        self.get_logger().warn(f"服务 {service_info['name']} 在 {timeout}秒内不可用")
                        break
                
                self.get_logger().info(f"✅ 服务客户端 {service_name} 初始化完成")
                
            except Exception as e:
                self.get_logger().error(f"❌ 初始化服务客户端 {service_name} 失败: {e}")
                self.clients[service_name] = None
    
    def test_move_service(self):
        """测试移动服务"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("测试移动服务")
        self.get_logger().info("="*60)
        
        try:
            client = self.clients.get('move')
            if not client:
                return {"success": False, "message": "移动服务客户端未初始化"}
            
            # 获取配置参数
            move_config = self.config.get('move_node', {}).get('ros__parameters', {})
            default_velocity = move_config.get('default_velocity', 0.2)
            default_seconds = move_config.get('default_seconds', 2.0)
            
            request = MoveCommand.Request()
            request.velocity = float(default_velocity)
            request.angle = 0.0
            request.seconds = float(default_seconds)
            
            future = client.call_async(request)
            
            # 等待响应
            start_time = time.time()
            while not future.done():
                if time.time() - start_time > 10.0:
                    return {"success": False, "message": "移动服务调用超时"}
                time.sleep(0.1)
            
            response = future.result()
            
            result = {
                "success": response.success,
                "message": response.message,
                "requested_velocity": request.velocity,
                "requested_angle": request.angle,
                "requested_seconds": request.seconds
            }
            
            self.get_logger().info(f"移动服务测试结果:")
            self.get_logger().info(f"  成功: {result['success']}")
            self.get_logger().info(f"  消息: {result['message']}")
            self.get_logger().info(f"  请求参数: 速度={result['requested_velocity']}, 角度={result['requested_angle']}, 时间={result['requested_seconds']}")
            
            return result
            
        except Exception as e:
            result = {"success": False, "message": f"移动服务测试失败: {str(e)}"}
            self.get_logger().error(f"❌ 移动服务测试失败: {e}")
            return result
    
    def test_yolo_detect_service(self):
        """测试YOLO检测服务"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("测试YOLO检测服务")
        self.get_logger().info("="*60)
        
        try:
            client = self.clients.get('yolo_detect')
            if not client:
                return {"success": False, "message": "YOLO检测服务客户端未初始化"}
            
            # 获取配置参数
            yolo_config = self.config.get('yolo_detect_node', {}).get('ros__parameters', {})
            default_threshold = yolo_config.get('conf_threshold', 0.8)
            
            request = YOLODetect.Request()
            request.threshold = float(default_threshold)
            
            future = client.call_async(request)
            
            # 等待响应
            start_time = time.time()
            while not future.done():
                if time.time() - start_time > 10.0:
                    return {"success": False, "message": "YOLO检测服务调用超时"}
                time.sleep(0.1)
            
            response = future.result()
            
            result = {
                "success": response.success,
                "message": response.message,
                "threshold": request.threshold,
                "objects_count": len(response.objects) if response.objects else 0,
                "objects": list(response.objects) if response.objects else [],
                "distances": list(response.distances) if response.distances else [],
                "positions": list(response.positions) if response.positions else []
            }
            
            self.get_logger().info(f"YOLO检测服务测试结果:")
            self.get_logger().info(f"  成功: {result['success']}")
            self.get_logger().info(f"  消息: {result['message']}")
            self.get_logger().info(f"  阈值: {result['threshold']}")
            self.get_logger().info(f"  检测到物体数量: {result['objects_count']}")
            if result['objects_count'] > 0:
                for i, (obj, dist, pos) in enumerate(zip(result['objects'], result['distances'], result['positions'])):
                    self.get_logger().info(f"    {i+1}. {obj} - 距离: {dist:.2f}m - 位置: {pos}")
            
            return result
            
        except Exception as e:
            result = {"success": False, "message": f"YOLO检测服务测试失败: {str(e)}"}
            self.get_logger().error(f"❌ YOLO检测服务测试失败: {e}")
            return result
    
    def test_vlm_detect_service(self):
        """测试VLM检测服务"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("测试VLM检测服务")
        self.get_logger().info("="*60)
        
        try:
            client = self.clients.get('vlm_detect')
            if not client:
                return {"success": False, "message": "VLM检测服务客户端未初始化"}
            
            request = VLMDetect.Request()
            
            future = client.call_async(request)
            
            # 等待响应
            start_time = time.time()
            while not future.done():
                if time.time() - start_time > 30.0:
                    return {"success": False, "message": "VLM检测服务调用超时"}
                time.sleep(0.1)
            
            response = future.result()
            
            result = {
                "success": response.success,
                "message": response.message,
                "description": response.description if hasattr(response, 'description') else "",
                "detailed_analysis": response.detailed_analysis if hasattr(response, 'detailed_analysis') else "",
                "objects_detected": response.objects_detected if hasattr(response, 'objects_detected') else [],
                "scene_type": response.scene_type if hasattr(response, 'scene_type') else "unknown"
            }
            
            self.get_logger().info(f"VLM检测服务测试结果:")
            self.get_logger().info(f"  成功: {result['success']}")
            self.get_logger().info(f"  消息: {result['message']}")
            if result['description']:
                self.get_logger().info(f"  描述: {result['description'][:100]}...")
            if result['detailed_analysis']:
                self.get_logger().info(f"  详细分析: {result['detailed_analysis'][:100]}...")
            self.get_logger().info(f"  检测到的物体: {result['objects_detected']}")
            self.get_logger().info(f"  场景类型: {result['scene_type']}")
            
            return result
            
        except Exception as e:
            result = {"success": False, "message": f"VLM检测服务测试失败: {str(e)}"}
            self.get_logger().error(f"❌ VLM检测服务测试失败: {e}")
            return result
    
    def test_track_service(self):
        """测试追踪服务"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("测试追踪服务")
        self.get_logger().info("="*60)
        
        try:
            client = self.clients.get('track')
            if not client:
                return {"success": False, "message": "追踪服务客户端未初始化"}
            
            # 获取配置参数
            track_config = self.config.get('track_node', {}).get('ros__parameters', {})
            default_target = track_config.get('default_target', 'person')
            
            request = Track.Request()
            request.target = default_target
            
            future = client.call_async(request)
            
            # 等待响应
            start_time = time.time()
            while not future.done():
                if time.time() - start_time > 10.0:
                    return {"success": False, "message": "追踪服务调用超时"}
                time.sleep(0.1)
            
            response = future.result()
            
            result = {
                "success": response.success,
                "message": response.message,
                "target": request.target
            }
            
            self.get_logger().info(f"追踪服务测试结果:")
            self.get_logger().info(f"  成功: {result['success']}")
            self.get_logger().info(f"  消息: {result['message']}")
            self.get_logger().info(f"  追踪目标: {result['target']}")
            
            return result
            
        except Exception as e:
            result = {"success": False, "message": f"追踪服务测试失败: {str(e)}"}
            self.get_logger().error(f"❌ 追踪服务测试失败: {e}")
            return result
    
    def test_check_obstacle_service(self):
        """测试障碍物检测服务"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("测试障碍物检测服务")
        self.get_logger().info("="*60)
        
        try:
            client = self.clients.get('check_obstacle')
            if not client:
                return {"success": False, "message": "障碍物检测服务客户端未初始化"}
            
            request = CheckObstacle.Request()
            
            future = client.call_async(request)
            
            # 等待响应
            start_time = time.time()
            while not future.done():
                if time.time() - start_time > 10.0:
                    return {"success": False, "message": "障碍物检测服务调用超时"}
                time.sleep(0.1)
            
            response = future.result()
            
            result = {
                "success": response.success,
                "message": response.message,
                "safe_direction": response.safe_direction,
                "min_distance": response.min_distance,
                "sector_ranges": list(response.sector_ranges) if response.sector_ranges else [],
                "has_obstacle": list(response.has_obstacle) if response.has_obstacle else []
            }
            
            self.get_logger().info(f"障碍物检测服务测试结果:")
            self.get_logger().info(f"  成功: {result['success']}")
            self.get_logger().info(f"  消息: {result['message']}")
            self.get_logger().info(f"  安全方向: {result['safe_direction']}°")
            self.get_logger().info(f"  最小距离: {result['min_distance']}m")
            self.get_logger().info(f"  扇区信息:")
            
            if result['sector_ranges'] and result['has_obstacle']:
                for i, (sector_range, has_obstacle) in enumerate(zip(result['sector_ranges'], result['has_obstacle'])):
                    status = "有障碍物" if has_obstacle else "无障碍物"
                    self.get_logger().info(f"    扇区 {i}: {sector_range} - {status}")
            
            return result
            
        except Exception as e:
            result = {"success": False, "message": f"障碍物检测服务测试失败: {str(e)}"}
            self.get_logger().error(f"❌ 障碍物检测服务测试失败: {e}")
            return result
    
    def test_ocr_service(self):
        """测试OCR服务"""
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("测试OCR服务")
        self.get_logger().info("="*60)
        
        try:
            client = self.clients.get('ocr')
            if not client:
                return {"success": False, "message": "OCR服务客户端未初始化"}
            
            request = OCR.Request()
            
            future = client.call_async(request)
            
            # 等待响应
            start_time = time.time()
            while not future.done():
                if time.time() - start_time > 30.0:
                    return {"success": False, "message": "OCR服务调用超时"}
                time.sleep(0.1)
            
            response = future.result()
            
            result = {
                "success": response.success,
                "message": response.message,
                "texts_count": len(response.texts) if response.texts else 0,
                "texts": list(response.texts) if response.texts else [],
                "confidences": list(response.confidences) if response.confidences else [],
                "positions": list(response.positions) if response.positions else []
            }
            
            self.get_logger().info(f"OCR服务测试结果:")
            self.get_logger().info(f"  成功: {result['success']}")
            self.get_logger().info(f"  消息: {result['message']}")
            self.get_logger().info(f"  识别到文本数量: {result['texts_count']}")
            if result['texts_count'] > 0:
                for i, (text, confidence, position) in enumerate(zip(result['texts'], result['confidences'], result['positions'])):
                    self.get_logger().info(f"    {i+1}. '{text}' - 置信度: {confidence:.2f} - 位置: {position}")
            
            return result
            
        except Exception as e:
            result = {"success": False, "message": f"OCR服务测试失败: {str(e)}"}
            self.get_logger().error(f"❌ OCR服务测试失败: {e}")
            return result
    
    def run_all_tests(self):
        """运行所有测试"""
        self.get_logger().info("🚀 开始综合测试所有服务节点...")
        
        # 初始化客户端
        self.initialize_clients()
        
        # 等待服务启动
        time.sleep(2)
        
        # 测试各个服务
        tests = [
            ("移动服务", self.test_move_service),
            ("YOLO检测服务", self.test_yolo_detect_service),
            ("VLM检测服务", self.test_vlm_detect_service),
            ("追踪服务", self.test_track_service),
            ("障碍物检测服务", self.test_check_obstacle_service),
            ("OCR服务", self.test_ocr_service),
        ]
        
        all_results = {}
        
        for test_name, test_func in tests:
            try:
                self.get_logger().info(f"\n{'='*80}")
                self.get_logger().info(f"开始测试: {test_name}")
                self.get_logger().info(f"{'='*80}")
                
                result = test_func()
                all_results[test_name] = result
                
                # 记录结果
                status = "✅ 通过" if result['success'] else "❌ 失败"
                self.get_logger().info(f"\n{status} {test_name}: {result['message']}")
                
            except Exception as e:
                self.get_logger().error(f"❌ 测试 {test_name} 时发生异常: {e}")
                all_results[test_name] = {"success": False, "message": f"测试异常: {str(e)}"}
        
        # 生成测试报告
        self.generate_test_report(all_results)
        
        return all_results
    
    def generate_test_report(self, results):
        """生成测试报告"""
        self.get_logger().info("\n" + "="*80)
        self.get_logger().info("测试报告总结")
        self.get_logger().info("="*80)
        
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result['success'])
        failed_tests = total_tests - passed_tests
        
        self.get_logger().info(f"总测试数: {total_tests}")
        self.get_logger().info(f"通过测试: {passed_tests}")
        self.get_logger().info(f"失败测试: {failed_tests}")
        self.get_logger().info(f"成功率: {(passed_tests/total_tests)*100:.1f}%")
        
        self.get_logger().info("\n详细结果:")
        for test_name, result in results.items():
            status = "✅ 通过" if result['success'] else "❌ 失败"
            self.get_logger().info(f"  {status} {test_name}: {result['message']}")
        
        # 保存JSON报告
        try:
            report_path = "test_results.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.get_logger().info(f"\n详细测试报告已保存到: {report_path}")
        except Exception as e:
            self.get_logger().error(f"保存测试报告失败: {e}")


def main():
    """主函数"""
    rclpy.init()
    
    try:
        test_client = ServiceTestClient()
        results = test_client.run_all_tests()
        
        # 检查整体结果
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result['success'])
        
        print(f"\n{'='*80}")
        print("测试完成!")
        print(f"{'='*80}")
        print(f"总测试数: {total_tests}")
        print(f"通过测试: {passed_tests}")
        print(f"失败测试: {total_tests - passed_tests}")
        print(f"成功率: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("🎉 所有服务节点测试通过!")
            exit_code = 0
        else:
            print("⚠️  部分服务节点测试失败，请检查相关节点和服务")
            exit_code = 1
            
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        exit_code = 1
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        exit_code = 1
    finally:
        if 'test_client' in locals():
            test_client.destroy_node()
        rclpy.shutdown()
    
    exit(exit_code)


if __name__ == '__main__':
    main()