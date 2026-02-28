# Jetson Orin Nano 部署指南

本文档介绍如何在 Jetson Orin Nano 上部署和优化 PGIAgent 系统。

## 系统要求

- **硬件**: NVIDIA Jetson Orin Nano (8GB/16GB)
- **操作系统**: JetPack 5.1.2 或更高版本 (Ubuntu 20.04)
- **存储**: 至少 32GB 可用空间
- **网络**: 稳定的互联网连接

## 1. 系统准备

### 1.1 更新系统
```bash
sudo apt update
sudo apt upgrade -y
sudo apt autoremove -y
```

### 1.2 安装基础工具
```bash
sudo apt install -y \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    python3-pip \
    python3-dev \
    python3-venv
```

## 2. ROS2 安装

### 2.1 安装 ROS2 Humble
```bash
# 设置语言环境
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# 添加 ROS2 仓库
sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install -y curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 安装 ROS2
sudo apt update
sudo apt install -y \
    ros-humble-desktop \
    ros-dev-tools \
    python3-colcon-common-extensions
```

### 2.2 设置环境变量
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## 3. Jetson 优化依赖安装

### 3.1 安装 JetPack 组件
```bash
# 安装完整的 JetPack
sudo apt install -y \
    nvidia-jetpack \
    nvidia-cuda \
    nvidia-cudnn \
    nvidia-tensorrt \
    libopencv-dev \
    python3-opencv
```

### 3.2 安装 Jetson 优化的 PyTorch
```bash
# 方法1: 通过 NVIDIA PyPI 源
pip3 install --upgrade pip
pip3 install nvidia-pyindex
pip3 install nvidia-pytorch nvidia-torchvision nvidia-tensorrt

# 方法2: 使用预编译的 wheel (推荐)
# 下载适合 JetPack 版本的 PyTorch
wget https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
pip3 install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

# 下载 TorchVision
wget https://developer.download.nvidia.com/compute/redist/jp/v50/vision/torchvision-0.16.0a0+91660a7.nv23.06-cp38-cp38-linux_aarch64.whl
pip3 install torchvision-0.16.0a0+91660a7.nv23.06-cp38-cp38-linux_aarch64.whl
```

### 3.3 安装 TensorRT Python 绑定
```bash
# 安装 TensorRT
sudo apt install -y \
    tensorrt \
    python3-libnvinfer \
    python3-libnvinfer-dev

# 安装 Python 绑定
pip3 install nvidia-tensorrt
```

## 4. PGIAgent 安装

### 4.1 克隆代码
```bash
cd ~
git clone <repository-url>
cd PGIAgent
```

### 4.2 使用优化安装脚本
```bash
# 运行 Jetson 优化安装脚本
chmod +x scripts/install_deps.sh
./scripts/install_deps.sh
```

### 4.3 手动安装依赖（备选）
```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装核心依赖
pip3 install --upgrade pip
pip3 install -r requirements_jetson.txt  # 见下文
```

## 5. Jetson 优化配置

### 5.1 创建 Jetson 专用 requirements 文件
创建 `requirements_jetson.txt`:
```txt
# Jetson Orin Nano 优化依赖
# 核心依赖
rclpy>=1.0.0
numpy>=1.24.0
pyyaml>=6.0
python-dotenv>=1.0.0
Pillow>=10.0.0

# Jetson 优化的计算机视觉
# 使用系统安装的 OpenCV (python3-opencv)
# 使用系统安装的 PyTorch (nvidia-pytorch)

# LangGraph 智能体框架
langgraph>=0.0.40
langchain>=0.1.0
langchain-core>=0.1.0
langchain-openai>=0.0.5

# 大模型 API
openai>=1.12.0
httpx>=0.25.0

# YOLO (使用 TensorRT 优化)
ultralytics>=8.0.0

# OCR
pytesseract>=0.3.10
easyocr>=1.7.0

# 开发工具
black>=23.0.0
flake8>=6.0.0
pytest>=7.4.0
```

### 5.2 环境变量优化
在 `.env` 文件中添加 Jetson 优化配置:
```bash
# Jetson 性能优化
USE_TENSORRT=true
USE_FP16=true
GPU_MEMORY_LIMIT_MB=2048
JETSON_MODE=true

# 图像处理优化
IMAGE_RESIZE_WIDTH=640
IMAGE_RESIZE_HEIGHT=480
ENABLE_IMAGE_COMPRESSION=true
USE_HARDWARE_ACCELERATION=true

# 模型路径
YOLO_ENGINE_PATH=./models/yolo11n.engine  # TensorRT 引擎
```

### 5.3 创建 TensorRT 引擎
```bash
# 转换 YOLO 模型为 TensorRT 引擎
python3 scripts/convert_to_tensorrt.py \
    --model ./models/yolo11n.pt \
    --engine ./models/yolo11n.engine \
    --fp16 \
    --batch-size 1 \
    --img-size 320
```

## 6. 性能优化技巧

### 6.1 电源模式设置
```bash
# 查看当前电源模式
sudo nvpmodel -q

# 设置为最大性能模式 (50W)
sudo nvpmodel -m 0

# 设置风扇速度
sudo jetson_clocks --fan
```

### 6.2 内存优化
```bash
# 创建交换空间 (如果需要)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 6.3 GPU 内存管理
```python
# 在代码中限制 GPU 内存使用
import torch
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.8)  # 使用 80% GPU 内存
```

## 7. 测试与验证

### 7.1 硬件测试
```bash
# 测试 GPU
nvidia-smi

# 测试 CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 测试 TensorRT
python3 -c "import tensorrt; print(f'TensorRT: {tensorrt.__version__}')"

# 测试 OpenCV
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}'); print(f'CUDA: {cv2.cuda.getCudaEnabledDeviceCount()}')"
```

### 7.2 PGIAgent 测试
```bash
# 运行测试脚本
python3 scripts/test_agent.py

# 测试性能
python3 scripts/benchmark_jetson.py
```

## 8. 故障排除

### 8.1 常见问题

**问题1**: PyTorch CUDA 不可用
```bash
# 解决方案: 重新安装 Jetson 专用 PyTorch
pip3 uninstall torch torchvision
pip3 install nvidia-pytorch nvidia-torchvision
```

**问题2**: 内存不足
```bash
# 解决方案: 减少批量大小，启用交换空间
# 修改 config/jetson_config.yaml 中的 batch_size: 1
```

**问题3**: TensorRT 引擎创建失败
```bash
# 解决方案: 使用 FP32 模式
python3 scripts/convert_to_tensorrt.py --fp32
```

### 8.2 性能监控
```bash
# 实时监控
sudo jetson_stats

# GPU 使用率
tegrastats

# 温度监控
cat /sys/class/thermal/thermal_zone*/temp
```

## 9. 部署建议

### 9.1 生产环境配置
1. **电源**: 使用官方电源适配器
2. **散热**: 确保良好散热，考虑使用散热风扇
3. **存储**: 使用高速 microSD 卡或 SSD
4. **网络**: 使用有线网络连接

### 9.2 启动脚本
创建启动脚本 `start_agent.sh`:
```bash
#!/bin/bash
# 设置性能模式
sudo nvpmodel -m 0
sudo jetson_clocks

# 激活虚拟环境
source venv/bin/activate

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

# 启动智能体
ros2 launch PGIAgent agent.launch.py
```

### 9.3 自动启动
```bash
# 创建 systemd 服务
sudo nano /etc/systemd/system/pgiagent.service
```

服务文件内容:
```ini
[Unit]
Description=PGIAgent Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/PGIAgent
ExecStart=/bin/bash /home/$USER/PGIAgent/start_agent.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## 10. 性能基准

在 Jetson Orin Nano 8GB 上的预期性能:

| 任务 | 帧率 (FPS) | 内存使用 | GPU 使用 |
|------|------------|----------|----------|
| YOLO 检测 | 15-20 | 2GB | 60% |
| VLM 推理 | 2-5 | 3GB | 80% |
| 移动控制 | 30+ | <1GB | 10% |
| 完整巡检 | 5-10 | 4GB | 70% |

## 参考资料

- [NVIDIA Jetson Orin Nano 文档](https://docs.nvidia.com/jetson/archives/r35.3.1/DeveloperGuide/index.html)
- [JetPack SDK](https://developer.nvidia.com/embedded/jetpack)
- [ROS2 on Jetson](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)
- [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)