#!/bin/bash
# PGIAgent 依赖安装脚本
# 适用于 Ubuntu 20.04/22.04 和 Jetson Orin Nano

set -e

echo "=== PGIAgent 依赖安装脚本 ==="
echo "系统检测: $(uname -a)"

# 检查Python版本
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "Python版本: $PYTHON_VERSION"

# 检查是否在ROS2环境中
if [ -z "$ROS_DISTRO" ]; then
    echo "警告: 未检测到ROS2环境"
    echo "请先设置ROS2环境: source /opt/ros/<distro>/setup.bash"
    read -p "是否继续安装Python依赖? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "ROS2环境: $ROS_DISTRO"
fi

# 创建虚拟环境（可选）
read -p "是否创建Python虚拟环境? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    echo "虚拟环境已激活"
fi

# 安装系统依赖
echo "安装系统依赖..."
sudo apt-get update

# ROS2依赖
if [ -n "$ROS_DISTRO" ]; then
    echo "安装ROS2依赖..."
    sudo apt-get install -y \
        ros-$ROS_DISTRO-rclpy \
        ros-$ROS_DISTRO-sensor-msgs \
        ros-$ROS_DISTRO-geometry-msgs \
        ros-$ROS_DISTRO-cv-bridge \
        ros-$ROS_DISTRO-image-transport \
        ros-$ROS_DISTRO-nav-msgs \
        ros-$ROS_DISTRO-tf2-ros \
        ros-$ROS_DISTRO-laser-geometry
fi

# 通用系统依赖
echo "安装通用系统依赖..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libopencv-dev \
    libtesseract-dev \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-eng

# 安装Python依赖
echo "安装Python依赖..."
pip install --upgrade pip

# 从requirements.txt安装
if [ -f "../requirements.txt" ]; then
    echo "从requirements.txt安装依赖..."
    pip install -r ../requirements.txt
else
    echo "requirements.txt未找到，安装核心依赖..."
    pip install \
        rclpy \
        numpy \
        opencv-python \
        Pillow \
        pyyaml \
        python-dotenv \
        langgraph \
        langchain \
        langchain-core \
        openai \
        httpx \
        pydantic \
        typing-extensions
fi

# 安装可选依赖
read -p "是否安装可选依赖(OCR, YOLO等)? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "安装可选依赖..."
    pip install \
        pytesseract \
        easyocr \
        ultralytics \
        torch torchvision --index-url https://download.pytorch.org/whl/cu118
fi

# 安装开发工具
read -p "是否安装开发工具? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "安装开发工具..."
    pip install \
        black \
        flake8 \
        mypy \
        pytest
fi

# Jetson Orin Nano特定优化
if [[ $(uname -n) == *"orin"* ]] || [[ $(uname -n) == *"jetson"* ]]; then
    echo "检测到Jetson设备，安装优化依赖..."
    
    # 安装JetPack（包含优化过的OpenCV、CUDA等）
    echo "安装NVIDIA JetPack..."
    sudo apt-get install -y \
        nvidia-jetpack \
        python3-pycuda \
        libopencv-dev \
        python3-opencv
    
    # 安装TensorRT
    echo "安装TensorRT..."
    sudo apt-get install -y \
        tensorrt \
        python3-libnvinfer \
        python3-libnvinfer-dev
    
    # 安装Jetson优化过的PyTorch
    echo "安装Jetson专用PyTorch..."
    # 方法1：通过NVIDIA PyPI源安装
    pip install --upgrade nvidia-pyindex
    pip install nvidia-pytorch
    
    # 方法2：或者使用预编译的wheel
    # wget https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
    # pip install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
    
    # 安装Jetson优化过的TorchVision
    echo "安装Jetson专用TorchVision..."
    pip install nvidia-torchvision
    
    # 安装Jetson优化过的TensorRT
    echo "安装TensorRT Python绑定..."
    pip install nvidia-tensorrt
    
    # 安装Jetson专用CUDA工具
    echo "安装CUDA工具..."
    sudo apt-get install -y \
        cuda-toolkit-12-2 \
        cuda-cudart-12-2 \
        cuda-cupti-12-2 \
        cuda-nvtx-12-2
    
    # 设置环境变量
    echo "设置Jetson环境变量..."
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    
    # 验证安装
    echo "验证Jetson安装..."
    python3 -c "import cv2; print(f'Jetson OpenCV: {cv2.__version__}')"
    python3 -c "import torch; print(f'Jetson PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    python3 -c "import tensorrt; print('TensorRT: OK')" 2>/dev/null || echo "TensorRT: 需要额外安装"
fi

# 配置环境变量
echo "配置环境变量..."
if [ -f ".env.example" ]; then
    if [ ! -f ".env" ]; then
        echo "创建.env文件..."
        cp .env.example .env
        echo "请编辑.env文件，添加API密钥"
    else
        echo ".env文件已存在"
    fi
fi

# 创建模型目录
echo "创建模型目录..."
mkdir -p ../models
mkdir -p ../models/easyocr

# 下载示例模型（可选）
read -p "是否下载示例YOLO模型? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "下载YOLOv11n模型..."
    wget -P ../models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt
fi

# 验证安装
echo "验证安装..."
python3 -c "import rclpy; print('ROS2 Python绑定: OK')" 2>/dev/null || echo "ROS2 Python绑定: 未安装"
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python3 -c "import langgraph; print('LangGraph: OK')" 2>/dev/null || echo "LangGraph: 未安装"

echo ""
echo "=== 安装完成 ==="
echo "下一步:"
echo "1. 编辑 .env 文件，添加API密钥"
echo "2. 运行测试: python scripts/test_agent.py"
echo "3. 构建ROS2包: colcon build --packages-select PGIAgent"
echo "4. 启动智能体: ros2 launch PGIAgent agent.launch.py"

if [ -d "venv" ]; then
    echo ""
    echo "虚拟环境使用说明:"
    echo "激活: source venv/bin/activate"
    echo "退出: deactivate"
fi