#!/usr/bin/env python3
"""
PGIAgent 测试脚本
测试电网巡检智能体的基本功能
"""

import sys
import os
import time
import json

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PGIAgent.agent import create_agent, run_agent_task


def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试智能体基本功能 ===")
    
    # 创建智能体
    print("1. 创建智能体实例...")
    agent = create_agent()
    
    # 测试可用工具
    print("2. 获取可用工具...")
    tools = agent.get_available_tools()
    print(f"   可用工具: {', '.join(tools)}")
    
    # 测试简单任务
    print("3. 测试简单任务...")
    task = "检查当前场景"
    result = agent.run(task, max_iterations=5)
    
    print(f"   任务: {result['task']}")
    print(f"   成功: {result['success']}")
    print(f"   迭代次数: {result['iterations']}")
    print(f"   状态摘要:\n{result['summary']}")
    
    return result['success']


def test_power_inspection():
    """测试电力巡检任务"""
    print("\n=== 测试电力巡检任务 ===")
    
    # 运行电力巡检任务
    task = "检查变电站A区的设备状态"
    print(f"任务: {task}")
    
    result = run_agent_task(task, max_iterations=10)
    
    print(f"成功: {result['success']}")
    print(f"迭代次数: {result['iterations']}")
    print(f"工具调用次数: {len(result['state']['tool_calls'])}")
    
    # 打印工具调用详情
    if result['state']['tool_calls']:
        print("工具调用记录:")
        for i, tool_call in enumerate(result['state']['tool_calls']):
            print(f"  {i+1}. {tool_call['tool_type']}: {tool_call.get('success', False)}")
            if not tool_call.get('success', False):
                print(f"     错误: {tool_call.get('error_message', '未知错误')}")
    
    return result['success']


def test_person_tracking():
    """测试人员追踪任务"""
    print("\n=== 测试人员追踪任务 ===")
    
    task = "追踪维护人员并保持安全距离"
    print(f"任务: {task}")
    
    result = run_agent_task(task, max_iterations=8)
    
    print(f"成功: {result['success']}")
    print(f"迭代次数: {result['iterations']}")
    
    # 检查是否调用了追踪工具
    track_calls = [tc for tc in result['state']['tool_calls'] if tc['tool_type'] == 'track']
    if track_calls:
        print(f"追踪工具调用: {len(track_calls)}次")
        for call in track_calls:
            print(f"  成功: {call.get('success', False)}, 结果: {call.get('result', {}).get('message', '无')}")
    
    return result['success']


def test_obstacle_avoidance():
    """测试障碍物避障"""
    print("\n=== 测试障碍物避障 ===")
    
    task = "移动到目标位置并避开障碍物"
    print(f"任务: {task}")
    
    result = run_agent_task(task, max_iterations=6)
    
    print(f"成功: {result['success']}")
    print(f"迭代次数: {result['iterations']}")
    
    # 检查是否调用了障碍物检测
    obstacle_calls = [tc for tc in result['state']['tool_calls'] if tc['tool_type'] == 'check_obstacle']
    if obstacle_calls:
        print(f"障碍物检测调用: {len(obstacle_calls)}次")
        for call in obstacle_calls:
            if call.get('success', False):
                result_data = call.get('result', {})
                safe_dir = result_data.get('safe_direction', 0)
                min_dist = result_data.get('min_distance', 0)
                print(f"  安全方向: {safe_dir}度, 最小距离: {min_dist}米")
    
    return result['success']


def test_ocr_function():
    """测试OCR功能"""
    print("\n=== 测试OCR功能 ===")
    
    task = "读取电力控制箱上的警告标志"
    print(f"任务: {task}")
    
    result = run_agent_task(task, max_iterations=6)
    
    print(f"成功: {result['success']}")
    print(f"迭代次数: {result['iterations']}")
    
    # 检查OCR结果
    ocr_calls = [tc for tc in result['state']['tool_calls'] if tc['tool_type'] == 'ocr']
    if ocr_calls:
        print(f"OCR调用: {len(ocr_calls)}次")
        for call in ocr_calls:
            if call.get('success', False):
                result_data = call.get('result', {})
                texts = result_data.get('texts', [])
                count = result_data.get('count', 0)
                print(f"  识别到 {count} 条文本: {texts}")
    
    return result['success']


def test_comprehensive_task():
    """测试综合任务"""
    print("\n=== 测试综合巡检任务 ===")
    
    task = "巡检整个变电站区域，检查所有设备状态，读取警告标志，确保安全"
    print(f"任务: {task}")
    
    start_time = time.time()
    result = run_agent_task(task, max_iterations=15)
    elapsed_time = time.time() - start_time
    
    print(f"执行时间: {elapsed_time:.2f}秒")
    print(f"成功: {result['success']}")
    print(f"迭代次数: {result['iterations']}")
    
    # 统计工具使用情况
    tool_stats = {}
    for tool_call in result['state']['tool_calls']:
        tool_type = tool_call['tool_type']
        tool_stats[tool_type] = tool_stats.get(tool_type, 0) + 1
    
    print("工具使用统计:")
    for tool, count in tool_stats.items():
        print(f"  {tool}: {count}次")
    
    # 打印执行计划
    if result['state']['plan']:
        print("执行计划:")
        for i, step in enumerate(result['state']['plan']):
            status = "✓" if i < result['state']['current_step'] else "○"
            print(f"  {status} {i+1}. {step}")
    
    return result['success']


def save_test_results(results):
    """保存测试结果"""
    output_file = "test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n测试结果已保存到: {output_file}")


def main():
    """主测试函数"""
    print("PGIAgent 智能体系统测试")
    print("=" * 50)
    
    test_results = {}
    
    try:
        # 运行各个测试
        test_results['basic_functionality'] = test_basic_functionality()
        time.sleep(1)
        
        test_results['power_inspection'] = test_power_inspection()
        time.sleep(1)
        
        test_results['person_tracking'] = test_person_tracking()
        time.sleep(1)
        
        test_results['obstacle_avoidance'] = test_obstacle_avoidance()
        time.sleep(1)
        
        test_results['ocr_function'] = test_ocr_function()
        time.sleep(1)
        
        test_results['comprehensive_task'] = test_comprehensive_task()
        
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        test_results['error'] = str(e)
    
    # 统计测试结果
    print("\n" + "=" * 50)
    print("测试结果统计:")
    
    passed = sum(1 for result in test_results.values() if result is True)
    total = len([v for v in test_results.values() if isinstance(v, bool)])
    
    print(f"通过: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%" if total > 0 else "成功率: N/A")
    
    # 保存结果
    save_test_results(test_results)
    
    # 总体评估
    if passed == total:
        print("\n✅ 所有测试通过！智能体系统功能正常。")
        return 0
    else:
        print(f"\n⚠️  有 {total-passed} 个测试失败，请检查问题。")
        return 1


if __name__ == "__main__":
    sys.exit(main())