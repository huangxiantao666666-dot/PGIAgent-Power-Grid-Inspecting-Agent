# PGIAgent - Power Grid Inspection Agent

A ROS2 and LangGraph-based intelligent agent system for JetAuto Mecanum Wheel Robot (equipped with Jetson Orin Nano) providing autonomous power grid inspection capabilities.

## Project Overview

PGIAgent is a complete intelligent agent system that integrates large language models (LLMs) with robot hardware capabilities to achieve autonomous power grid inspection tasks. The system adopts a modular design supporting multiple vision models, motion control, and environmental perception tools.

### Core Features

- **🤖 Intelligent Decision Making**: LangGraph-based workflow engine supporting planning, execution, reflection cycles
- **👁️ Multimodal Perception**: Integrated YOLOv11 object detection, Vision Language Model (VLM) scene understanding, OCR text recognition
- **🚗 Safe Navigation**: LiDAR obstacle detection, PID control tracking, safe distance maintenance
- **🔧 Tool-based Architecture**: All capabilities encapsulated as callable tool functions
- **⚡ Jetson Optimization**: TensorRT acceleration and performance optimization for Jetson Orin Nano
- **🌐 Cloud/Local Hybrid**: Support for DeepSeek, Qwen cloud APIs, and local model inference

## System Architecture

```
PGIAgent/
├── agent/                    # Agent Core
│   ├── state.py             # State management
│   ├── tools.py             # Tool function encapsulation
│   ├── agent_graph.py       # LangGraph workflow
│   ├── prompts.py           # Prompt templates
│   └── __init__.py
├── nodes/                   # ROS2 Nodes
│   ├── move_node.py         # Motion control node
│   ├── detection_node.py    # Object detection node
│   ├── vlm_node.py          # Vision language model node
│   ├── track_node.py        # Target tracking node
│   ├── obstacle_node.py     # Obstacle detection node
│   └── ocr_node.py          # OCR node
├── msg/                     # ROS2 Service Definitions
│   ├── MoveCommand.srv
│   ├── YOLODetect.srv
│   ├── VLMDetect.srv
│   ├── Track.srv
│   ├── CheckObstacle.srv
│   └── OCR.srv
├── config/                  # Configuration Files
│   ├── agent_config.yaml    # Agent configuration
│   ├── model_config.yaml    # Model configuration
│   └── ros_params.yaml      # ROS parameters
├── scripts/                 # Utility Scripts
│   ├── test_agent.py        # Test script
│   ├── install_deps.sh      # Dependency installation
│   └── benchmark_jetson.py  # Performance testing
├── docs/                    # Documentation
│   └── jetson_setup.md      # Jetson deployment guide
├── models/                  # Model files
├── launch/                  # ROS2 launch files
├── package.xml              # ROS2 package definition
├── setup.py                 # Python package configuration
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variables example
└── .gitignore             
```

## Quick Start

### 1. Environment Setup

```bash
# Clone the project
git clone <repository-url>
cd PGIAgent

# Install dependencies
chmod +x scripts/install_deps.sh
./scripts/install_deps.sh

# Configure environment variables
cp .env.example .env
# Edit .env file to add API keys
```

### 2. Test the Agent

```bash
# Run tests
python scripts/test_agent.py

# Test output example:
# === Testing Agent Basic Functionality ===
# 1. Creating agent instance...
# 2. Getting available tools...
#    Available tools: move, yolo_detect, VLM_detect, track, check_obstacle, ocr
# 3. Testing simple task...
#    Task: Inspect current scene
#    Success: True
#    Iterations: 5
```

### 3. ROS2 Integration

```bash
# Build ROS2 package
colcon build --packages-select PGIAgent
source install/setup.bash

# Launch all nodes
ros2 launch PGIAgent agent.launch.py

# Launch agent separately
ros2 run PGIAgent agent_node
```

## Tool Functions

The agent can call the following tool functions:

### 1. Motion Control
```python
move(velocity=0.2, angle=0.0, seconds=2.0)
```
- `velocity`: Movement speed (m/s), positive for forward, negative for backward
- `angle`: Steering angle (degrees), 0 for straight, positive for left turn, negative for right turn
- `seconds`: Movement duration (seconds)

### 2. YOLO Object Detection
```python
yolo_detect(threshold=0.8)
```
- `threshold`: Confidence threshold (0.0-1.0)
- Returns: Object list, distances, position descriptions

### 3. Vision Language Model Scene Understanding
```python
VLM_detect()
```
- Returns: Detailed scene description

### 4. Target Tracking
```python
track(target="person")
```
- `target`: Tracking target ("person", "electric_box", "transformer")
- Returns: Tracking result

### 5. Obstacle Detection
```python
check_obstacle()
```
- Returns: Safe direction, minimum obstacle distance, safe sectors

### 6. OCR Text Recognition
```python
ocr()
```
- Returns: Recognized text list and confidence scores

## Usage Examples

### Basic Usage
```python
from PGIAgent.agent import create_agent

# Create agent
agent = create_agent()

# Execute task
result = agent.run("Inspect equipment status in substation area A", max_iterations=10)

print(f"Task: {result['task']}")
print(f"Success: {result['success']}")
print(f"Iterations: {result['iterations']}")
print(f"State Summary:\n{result['summary']}")
```

### Custom Configuration
```python
from PGIAgent.agent import AgentConfig, create_agent

# Custom configuration
config = AgentConfig(
    llm_provider="deepseek",
    llm_model="deepseek-chat",
    max_iterations=15,
    default_move_velocity=0.3,
    yolo_threshold=0.7
)

# Create agent
agent = create_agent(config=config)
```

### ROS2 Node Integration
```python
import rclpy
from PGIAgent.agent import create_agent

class AgentNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('pgi_agent_node')
        
        # Create agent with ROS node
        self.agent = create_agent(ros_node=self)
        
        # Create service
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

## Configuration

### Agent Configuration (config/agent_config.yaml)
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

### Model Configuration (config/model_config.yaml)
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

## Jetson Orin Nano Optimization

### Performance Optimization
1. **TensorRT Acceleration**: Convert YOLO model to TensorRT engine
2. **FP16 Inference**: Use half-precision floating point to reduce memory usage
3. **Memory Optimization**: Limit GPU memory usage, enable swap space
4. **Power Management**: Set to maximum performance mode

### Deployment Steps
Detailed steps in [docs/jetson_setup.md](docs/jetson_setup.md)

```bash
# Convert to TensorRT engine
python scripts/convert_to_tensorrt.py \
    --model ./models/yolo11n.pt \
    --engine ./models/yolo11n.engine \
    --fp16

# Set performance mode
sudo nvpmodel -m 0  # Maximum performance
sudo jetson_clocks
```

## Task Examples

### 1. Substation Inspection
```
Task: "Inspect substation area A, check all equipment status, read warning signs"

Execution Plan:
1. Use YOLO to detect electrical equipment in current scene
2. Use VLM to analyze equipment status in detail
3. Check obstacles, determine safe movement direction
4. Move to first equipment
5. Use OCR to read equipment labels
6. Record equipment status
7. Repeat steps 4-6 for all equipment
8. Generate inspection report
```

### 2. Personnel Tracking
```
Task: "Track maintenance personnel and maintain safe distance"

Execution Plan:
1. Use YOLO to detect personnel position
2. Start tracking mode
3. Follow while maintaining 1.2m safe distance
4. Continuously check obstacles
5. Maintain observation when personnel stops
```

### 3. Equipment Check
```
Task: "Check transformer equipment status"

Execution Plan:
1. Move to safe position in front of transformer
2. Use VLM to observe transformer appearance
3. Use OCR to read nameplate information
4. Check indicator light status
5. Record inspection results
```

## Development Guide

### Adding New Tools
1. Add tool function in `agent/tools.py`
2. Define tool type in `agent/state.py`
3. Integrate into workflow in `agent/agent_graph.py`
4. Create corresponding ROS2 service definition and node

### Extending Vision Models
1. Add model configuration in `config/model_config.yaml`
2. Implement model calls in `agent/tools.py`
3. Create corresponding ROS2 service node

### Custom Workflow
```python
from langgraph.graph import StateGraph
from PGIAgent.agent.state import AgentState

# Create custom workflow
workflow = StateGraph(AgentState)

# Add custom node
workflow.add_node("custom_node", self.custom_node_function)

# Set edges and conditions
workflow.add_edge("start", "custom_node")
workflow.add_edge("custom_node", "end")
```

## Troubleshooting

### Common Issues

1. **ROS2 Services Unavailable**
   ```bash
   # Check service status
   ros2 service list
   ros2 service call /pgi_agent/move ...
   ```

2. **Model Loading Failed**
   ```bash
   # Check model path
   ls -la ./models/
   # Re-download model
   wget -P ./models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt
   ```

3. **API Call Failed**
   ```bash
   # Check environment variables
   echo $DEEPSEEK_API_KEY
   # Test API connection
   python -c "import openai; openai.api_key='your_key'; print('OK')"
   ```

4. **Jetson Performance Issues**
   ```bash
   # Monitor resource usage
   tegrastats
   nvidia-smi
   # Optimize settings
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export ENABLE_VISUALIZATION=true

# Run tests
python scripts/test_agent.py --debug
```

## Performance Benchmarks

| Component | Jetson Orin Nano 8GB | Description |
|-----------|---------------------|-------------|
| YOLOv11 Detection | 15-20 FPS | TensorRT accelerated, 320x320 input |
| VLM Inference | 2-5 FPS | Qwen-VL API calls |
| Motion Control | 30+ FPS | ROS2 topic publishing |
| Complete Inspection Task | 5-10 FPS | Combined all components |
| Memory Usage | 4-6 GB | Peak usage |
| GPU Utilization | 60-80% | Typical workload |

## Contributing

1. Fork the project
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [JetAuto Robot Platform](https://www.jetson.ai/jetauto)
- [ROS2](https://docs.ros.org/) - Robot Operating System
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Agent workflow
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Object detection
- [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing) - Edge AI platform


---

**Note**: This project is a research prototype. Conduct thorough testing and safety assessment before actual deployment. Power grid inspection scenarios involve high-voltage equipment, always follow on-site safety regulations.