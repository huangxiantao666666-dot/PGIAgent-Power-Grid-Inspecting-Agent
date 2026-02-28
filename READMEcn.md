# PGIAgent - 电网巡检机器小车智能体

基于 ROS2 和 LangGraph 的智能体系统，为 JetAuto 麦轮式机器人（配备 Jetson Orin Nano）提供电网巡检能力。

## 项目概述

PGIAgent 是一个完整的智能体系统，将大语言模型（LLM）与机器人硬件能力相结合，实现自主的电网巡检任务。系统采用模块化设计，支持多种视觉模型、移动控制和环境感知工具。

### 核心特性

- **🤖 智能决策**: 基于 LangGraph 的工作流引擎，支持规划、执行、反思循环
- **👁️ 多模态感知**: 集成 YOLOv11 物体检测、视觉大模型（VLM）场景理解、OCR 文字识别
- **🚗 安全移动**: 激光雷达障碍物检测、PID 控制追踪、安全距离保持
- **🔧 工具化架构**: 所有能力封装为可调用的工具函数
- **⚡ Jetson 优化**: 针对 Jetson Orin Nano 的 TensorRT 加速和性能优化
- **🌐 云端/本地混合**: 支持 DeepSeek、Qwen 等云端 API，也支持本地模型推理

## 系统架构

```
PGIAgent/
├── agent/                    # 智能体核心
│   ├── state.py             # 状态管理
│   ├── tools.py             # 工具函数封装
│   ├── agent_graph.py       # LangGraph 工作流
│   ├── prompts.py           # 提示词模板
│   └── __init__.py
├── nodes/                   # ROS2 节点
│   ├── move_node.py         # 移动控制节点
│   ├── detection_node.py    # 物体检测节点
│   ├── vlm_node.py          # 视觉大模型节点
│   ├── track_node.py        # 目标追踪节点
│   ├── obstacle_node.py     # 障碍物检测节点
│   └── ocr_node.py          # OCR 节点
├── msg/                     # ROS2 服务定义
│   ├── MoveCommand.srv
│   ├── YOLODetect.srv
│   ├── VLMDetect.srv
│   ├── Track.srv
│   ├── CheckObstacle.srv
│   └── OCR.srv
├── config/                  # 配置文件
│   ├── agent_config.yaml    # 智能体配置
│   ├── model_config.yaml    # 模型配置
│   └── ros_params.yaml      # ROS 参数
├── scripts/                 # 工具脚本
│   ├── test_agent.py        # 测试脚本
│   ├── install_deps.sh      # 依赖安装
│   └── benchmark_jetson.py  # 性能测试
├── docs/                    # 文档
│   └── jetson_setup.md      # Jetson 部署指南
├── models/                  # 模型文件
├── launch/                  # ROS2 启动文件
├── package.xml              # ROS2 包定义
├── setup.py                 # Python 包配置
├── requirements.txt         # Python 依赖
├── .env.example             # 环境变量示例
└── .gitignore
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd PGIAgent

# 安装依赖
chmod +x scripts/install_deps.sh
./scripts/install_deps.sh

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，添加 API 密钥
```

### 2. 测试智能体

```bash
# 运行测试
python scripts/test_agent.py

# 测试输出示例:
# === 测试智能体基本功能 ===
# 1. 创建智能体实例...
# 2. 获取可用工具...
#    可用工具: move, yolo_detect, VLM_detect, track, check_obstacle, ocr
# 3. 测试简单任务...
#    任务: 检查当前场景
#    成功: True
#    迭代次数: 5
```

### 3. ROS2 集成

```bash
# 构建 ROS2 包
colcon build --packages-select PGIAgent
source install/setup.bash

# 启动所有节点
ros2 launch PGIAgent agent.launch.py

# 单独启动智能体
ros2 run PGIAgent agent_node
```

## 工具函数

智能体可以调用以下工具函数：

### 1. 移动控制
```python
move(velocity=0.2, angle=0.0, seconds=2.0)
```
- `velocity`: 移动速度 (m/s)，正数为前进，负数为后退
- `angle`: 转向角度 (度)，0为直行，正数为左转，负数为右转
- `seconds`: 移动时间 (秒)

### 2. YOLO 物体检测
```python
yolo_detect(threshold=0.8)
```
- `threshold`: 置信度阈值 (0.0-1.0)
- 返回: 物体列表、距离、位置描述

### 3. 视觉大模型场景理解
```python
VLM_detect()
```
- 返回: 详细的场景描述

### 4. 目标追踪
```python
track(target="person")
```
- `target`: 追踪目标 ("person", "electric_box", "transformer")
- 返回: 追踪结果

### 5. 障碍物检测
```python
check_obstacle()
```
- 返回: 安全方向、最小障碍物距离、安全扇区

### 6. OCR 文字识别
```python
ocr()
```
- 返回: 识别出的文本列表和置信度

## 使用示例

### 基本使用
```python
from PGIAgent.agent import create_agent

# 创建智能体
agent = create_agent()

# 执行任务
result = agent.run("检查变电站A区的设备状态", max_iterations=10)

print(f"任务: {result['task']}")
print(f"成功: {result['success']}")
print(f"迭代次数: {result['iterations']}")
print(f"状态摘要:\n{result['summary']}")
```

### 自定义配置
```python
from PGIAgent.agent import AgentConfig, create_agent

# 自定义配置
config = AgentConfig(
    llm_provider="deepseek",
    llm_model="deepseek-chat",
    max_iterations=15,
    default_move_velocity=0.3,
    yolo_threshold=0.7
)

# 创建智能体
agent = create_agent(config=config)
```

### ROS2 节点集成
```python
import rclpy
from PGIAgent.agent import create_agent

class AgentNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('pgi_agent_node')
        
        # 创建智能体，传入 ROS 节点
        self.agent = create_agent(ros_node=self)
        
        # 创建服务
        self.task_service = self.create_service(
            Trigger, '/pgi_agent/execute_task',
            self.execute_task_callback
        )
    
    def execute_task_callback(self, request, response):
        task = request.task
        result = self.agent.run(task)
        
        response.success = result['success']
        response.message = result['summary']
        return response
```

## 配置说明

### 智能体配置 (config/agent_config.yaml)
```yaml
agent:
  max_iterations: 20
  use_reflection: true
  reflection_depth: 2
  
  planning:
    use_llm_planning: true
    max_plan_steps: 10
    
  safety:
    max_velocity: 0.5
    max_angular_velocity: 1.0
    min_obstacle_distance: 0.3
    emergency_stop_distance: 0.2
```

### 模型配置 (config/model_config.yaml)
```yaml
models:
  llm:
    provider: "deepseek"  # deepseek, qwen, openai, local
    model: "deepseek-chat"
    api_key: "${DEEPSEEK_API_KEY}"
    
  vlm:
    provider: "qwen"
    model: "qwen-vl-max"
    
  yolo:
    model_path: "./models/yolo11n.pt"
    engine_path: "./models/yolo11n.engine"
    conf_threshold: 0.8
    img_size: 320
    
  ocr:
    provider: "easyocr"
    languages: ["ch_sim", "en"]
```

## Jetson Orin Nano 优化

### 性能优化
1. **TensorRT 加速**: 将 YOLO 模型转换为 TensorRT 引擎
2. **FP16 推理**: 使用半精度浮点数减少内存使用
3. **内存优化**: 限制 GPU 内存使用，启用交换空间
4. **电源管理**: 设置最大性能模式

### 部署步骤
详细步骤见 [docs/jetson_setup.md](docs/jetson_setup.md)

```bash
# 转换为 TensorRT 引擎
python scripts/convert_to_tensorrt.py \
    --model ./models/yolo11n.pt \
    --engine ./models/yolo11n.engine \
    --fp16

# 设置性能模式
sudo nvpmodel -m 0  # 最大性能
sudo jetson_clocks
```

## 任务示例

### 1. 变电站巡检
```
任务: "巡检变电站A区，检查所有设备状态，读取警告标志"

执行计划:
1. 使用YOLO检测当前场景中的电力设备
2. 使用VLM详细分析设备状态
3. 检查障碍物，确定安全移动方向
4. 移动到第一个设备前
5. 使用OCR读取设备标签
6. 记录设备状态
7. 重复步骤4-6检查所有设备
8. 生成巡检报告
```

### 2. 人员追踪
```
任务: "追踪维护人员并保持安全距离"

执行计划:
1. 使用YOLO检测人员位置
2. 启动追踪模式
3. 保持1.2米安全距离跟随
4. 持续检查障碍物
5. 人员停止时保持观察
```

### 3. 设备检查
```
任务: "检查变压器设备状态"

执行计划:
1. 移动到变压器前方安全位置
2. 使用VLM观察变压器外观
3. 使用OCR读取铭牌信息
4. 检查指示灯状态
5. 记录检查结果
```

## 开发指南

### 添加新工具
1. 在 `agent/tools.py` 中添加工具函数
2. 在 `agent/state.py` 中定义工具类型
3. 在 `agent/agent_graph.py` 中集成到工作流
4. 创建对应的 ROS2 服务定义和节点

### 扩展视觉模型
1. 在 `config/model_config.yaml` 中添加模型配置
2. 在 `agent/tools.py` 中实现模型调用
3. 创建对应的 ROS2 服务节点

### 自定义工作流
```python
from langgraph.graph import StateGraph
from PGIAgent.agent.state import AgentState

# 创建自定义工作流
workflow = StateGraph(AgentState)

# 添加自定义节点
workflow.add_node("custom_node", self.custom_node_function)

# 设置边和条件
workflow.add_edge("start", "custom_node")
workflow.add_edge("custom_node", "end")
```

## 故障排除

### 常见问题

1. **ROS2 服务不可用**
   ```bash
   # 检查服务状态
   ros2 service list
   ros2 service call /pgi_agent/move ...
   ```

2. **模型加载失败**
   ```bash
   # 检查模型路径
   ls -la ./models/
   # 重新下载模型
   wget -P ./models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt
   ```

3. **API 调用失败**
   ```bash
   # 检查环境变量
   echo $DEEPSEEK_API_KEY
   # 测试 API 连接
   python -c "import openai; openai.api_key='your_key'; print('OK')"
   ```

4. **Jetson 性能问题**
   ```bash
   # 监控资源使用
   tegrastats
   nvidia-smi
   # 优化设置
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```

### 调试模式
```bash
# 启用调试日志
export LOG_LEVEL=DEBUG
export ENABLE_VISUALIZATION=true

# 运行测试
python scripts/test_agent.py --debug
```

## 性能基准

| 组件 | Jetson Orin Nano 8GB | 说明 |
|------|---------------------|------|
| YOLOv11 检测 | 15-20 FPS | TensorRT 加速，320x320 输入 |
| VLM 推理 | 2-5 FPS | Qwen-VL API 调用 |
| 移动控制 | 30+ FPS | ROS2 话题发布 |
| 完整巡检任务 | 5-10 FPS | 综合所有组件 |
| 内存使用 | 4-6 GB | 峰值使用量 |
| GPU 使用率 | 60-80% | 典型工作负载 |

## 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 致谢

- [JetAuto 机器人平台](https://www.jetson.ai/jetauto)
- [ROS2](https://docs.ros.org/) - 机器人操作系统
- [LangGraph](https://langchain-ai.github.io/langgraph/) - 智能体工作流
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - 物体检测
- [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing) - 边缘AI平台



---

**注意**: 本项目为研究原型，实际部署前请进行充分测试和安全评估。电力巡检场景涉及高压设备，务必遵守现场安全规程。