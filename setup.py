from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'PGIAgent'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 配置文件
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml.example')),
        # 启动文件
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # 资源文件
        (os.path.join('share', package_name, 'resource'), glob('resource/*')),
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
            # 工具节点
            'move_node = PGIAgent.nodes.move_node:main',
            'yolo_detect_node = PGIAgent.nodes.yolo_detect_node:main',
            'vlm_detect_node = PGIAgent.nodes.vlm_detect_node:main',
            'track_node = PGIAgent.nodes.track_node:main',
            'obstacle_check_node = PGIAgent.nodes.obstacle_check_node:main',
            'ocr_node = PGIAgent.nodes.ocr_node:main',
            'agent_bridge_node = PGIAgent.nodes.agent_bridge_node:main',
            # 现有节点
            'perception_node = PGIAgent.detection_node:main',
            'pid_control_node = PGIAgent.pid_control_node:main',
            # 智能体启动脚本
            'agent_runner = PGIAgent.scripts.agent_runner:main',
        ],
    },
)