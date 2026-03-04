from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'PGIAgent'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament_index资源索引
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # 配置文件
        (os.path.join('share', package_name, 'config'), 
         glob('config/*.yaml') + glob('config/*.yaml.example')),
        
        # 启动文件
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.launch.py')),
        
        # 资源文件
        (os.path.join('share', package_name, 'resource'), 
         glob('resource/*')),
        
        # ⭐⭐⭐ 重要：服务定义文件 ⭐⭐⭐
        (os.path.join('share', package_name, 'srv'), 
         glob('srv/*.srv')),
        
        # 模型文件（如果有）
        (os.path.join('share', package_name, 'models'),
         glob('models/*')),
        
        # 文档文件（如果有）
        (os.path.join('share', package_name, 'docs'),
         glob('docs/*.md')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='PGIAgent Maintainer',
    maintainer_email='admin@example.com',
    description='Power Grid Inspection Agent for JetAuto robot with ROS2 and LangGraph integration',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # ✅ 工具节点（6个核心ROS2节点）- 这些是真正的Node
            'move_node = PGIAgent.nodes.move_node:main',
            'detection_node = PGIAgent.nodes.detection_node:main',
            'vlm_node = PGIAgent.nodes.vlm_node:main',
            'track_node = PGIAgent.nodes.track_node:main',
            'obstacle_node = PGIAgent.nodes.obstacle_node:main',
            'ocr_node = PGIAgent.nodes.ocr_node:main',
            
            # ⚠️ 智能体节点 - 如果agent不是Node，应该移除或改为其他方式
            # 'agent_node = PGIAgent.agent.agent_node:main',        # 移除
            # 'task_manager_node = PGIAgent.agent.task_manager_node:main',  # 移除
            
            # ⚠️ 辅助节点 - 这些是脚本还是Node？
            # 'service_tester_node = PGIAgent.scripts.service_tester:main',  # 如果是Node就保留
            # 'web_interface_node = PGIAgent.scripts.web_interface:main',    # 如果是Node就保留
            
            # ✅ 测试脚本 - 这些应该是工具脚本，不是Node
            # 'test_agent = PGIAgent.scripts.test_agent:main',      # 如果只是测试脚本，可以保留或改为普通脚本
            # 'test_tools = PGIAgent.scripts.test_tools:main',      # 如果只是测试脚本，可以保留或改为普通脚本
            
            # ✅ 工具脚本 - 这些应该是工具脚本，不是Node
            # 'agent_runner = PGIAgent.scripts.agent_runner:main',   # 如果只是工具脚本，可以保留或改为普通脚本
            # 'model_converter = PGIAgent.scripts.model_converter:main',  # 如果只是工具脚本，可以保留或改为普通脚本
        ],
    },
)