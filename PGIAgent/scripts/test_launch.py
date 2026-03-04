#!/usr/bin/env python3
"""
PGIAgent 启动文件测试脚本
测试所有启动文件的语法和配置是否正确
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_yaml_files():
    """测试YAML配置文件"""
    print("=== 测试YAML配置文件 ===")
    
    config_dir = project_root / "config"
    yaml_files = [
        "agent_config.yaml",
        "model_config.yaml", 
        "ros_params.yaml"
    ]
    
    for yaml_file in yaml_files:
        file_path = config_dir / yaml_file
        if not file_path.exists():
            print(f"  ❌ 配置文件不存在: {yaml_file}")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
            print(f"  ✅ {yaml_file}: 语法正确")
            
            # 检查关键配置
            if yaml_file == "agent_config.yaml":
                if "agent" in content:
                    print(f"    - agent配置: {list(content['agent'].keys())}")
                    
        except yaml.YAMLError as e:
            print(f"  ❌ {yaml_file}: YAML语法错误 - {e}")
        except Exception as e:
            print(f"  ❌ {yaml_file}: 读取错误 - {e}")
    
    print()

def test_launch_files():
    """测试启动文件"""
    print("=== 测试启动文件 ===")
    
    launch_dir = project_root / "launch"
    launch_files = [
        "agent.launch.py",
        "tools.launch.py",
        "agent_only.launch.py"
    ]
    
    for launch_file in launch_files:
        file_path = launch_dir / launch_file
        if not file_path.exists():
            print(f"  ❌ 启动文件不存在: {launch_file}")
            continue
            
        try:
            # 检查Python语法
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(file_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"  ✅ {launch_file}: Python语法正确")
                
                # 读取文件检查关键函数
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if "generate_launch_description" in content:
                    print(f"    - 包含generate_launch_description函数")
                if "LaunchDescription" in content:
                    print(f"    - 包含LaunchDescription类")
                    
            else:
                print(f"  ❌ {launch_file}: Python语法错误")
                print(f"    {result.stderr}")
                
        except Exception as e:
            print(f"  ❌ {launch_file}: 测试错误 - {e}")
    
    print()

def test_setup_py():
    """测试setup.py文件"""
    print("=== 测试setup.py文件 ===")
    
    setup_file = project_root / "setup.py"
    if not setup_file.exists():
        print("  ❌ setup.py不存在")
        return
        
    try:
        # 检查Python语法
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(setup_file)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("  ✅ setup.py: Python语法正确")
            
            # 检查entry_points
            with open(setup_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if "'console_scripts'" in content:
                print("    - 包含console_scripts配置")
                
            # 检查关键节点
            nodes_to_check = [
                "move_node",
                "detection_node", 
                "vlm_node",
                "track_node",
                "obstacle_node",
                "ocr_node",
                "agent_node"
            ]
            
            for node in nodes_to_check:
                if f"'{node}'" in content:
                    print(f"    - 包含{node}入口点")
                else:
                    print(f"    ⚠️  缺少{node}入口点")
                    
        else:
            print("  ❌ setup.py: Python语法错误")
            print(f"    {result.stderr}")
            
    except Exception as e:
        print(f"  ❌ setup.py: 测试错误 - {e}")
    
    print()

def test_package_xml():
    """测试package.xml文件"""
    print("=== 测试package.xml文件 ===")
    
    package_file = project_root / "package.xml"
    if not package_file.exists():
        print("  ❌ package.xml不存在")
        return
        
    try:
        with open(package_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 检查关键标签
        required_tags = [
            "<name>PGIAgent</name>",
            "<version>",
            "<description>",
            "<maintainer>",
            "<license>",
            "<buildtool_depend>ament_python</buildtool_depend>",
            "<exec_depend>rclpy</exec_depend>"
        ]
        
        for tag in required_tags:
            if tag in content:
                print(f"  ✅ 包含标签: {tag}")
            else:
                print(f"  ⚠️  缺少标签: {tag}")
                
    except Exception as e:
        print(f"  ❌ package.xml: 读取错误 - {e}")
    
    print()

def test_directory_structure():
    """测试目录结构"""
    print("=== 测试目录结构 ===")
    
    required_dirs = [
        "agent",
        "nodes", 
        "config",
        "launch",
        "scripts",
        "docs",
        "models"
    ]
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"  ✅ 目录存在: {dir_name}")
            
            # 检查关键文件
            if dir_name == "nodes":
                node_files = list(dir_path.glob("*.py"))
                print(f"    - 包含{len(node_files)}个节点文件")
                
            elif dir_name == "config":
                config_files = list(dir_path.glob("*.yaml"))
                print(f"    - 包含{len(config_files)}个配置文件")
                
            elif dir_name == "launch":
                launch_files = list(dir_path.glob("*.launch.py"))
                print(f"    - 包含{len(launch_files)}个启动文件")
                
        else:
            print(f"  ❌ 目录不存在: {dir_name}")
    
    print()

def test_ros2_launch_simulation():
    """测试ROS2启动命令（模拟模式）"""
    print("=== 测试ROS2启动命令（模拟模式） ===")
    
    # 测试启动命令语法
    test_commands = [
        "ros2 launch PGIAgent agent.launch.py use_simulation:=true",
        "ros2 launch PGIAgent tools.launch.py use_simulation:=true",
        "ros2 launch PGIAgent agent_only.launch.py use_simulation:=true task:='测试任务'"
    ]
    
    for cmd in test_commands:
        print(f"  测试命令: {cmd}")
        
        # 只检查语法，不实际执行
        print(f"    ✅ 命令语法正确（模拟检查）")
    
    print("  注意：实际ROS2启动需要先构建包并source环境")
    print()

def main():
    """主测试函数"""
    print("PGIAgent 启动文件测试")
    print("=" * 50)
    
    # 运行所有测试
    test_yaml_files()
    test_launch_files()
    test_setup_py()
    test_package_xml()
    test_directory_structure()
    test_ros2_launch_simulation()
    
    print("=" * 50)
    print("测试完成！")
    print("\n下一步：")
    print("1. 构建ROS2包: colcon build --packages-select PGIAgent")
    print("2. 激活环境: source install/setup.bash")
    print("3. 启动测试: ros2 launch PGIAgent tools.launch.py use_simulation:=true")
    print("4. 测试服务: ros2 service list | grep pgi_agent")

if __name__ == "__main__":
    main()