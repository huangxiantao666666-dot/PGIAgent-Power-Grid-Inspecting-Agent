# PGIAgent 项目技术原理详解

本文档详细介绍 PGIAgent 电网巡检智能体系统的技术原理，包括 Agent 工作流设计和六个工具的实现细节。

## 目录

1. [系统概述](#系统概述)
2. [Agent 工作流](#agent-工作流)
3. [工具系统架构](#工具系统架构)
4. [六个工具详解](#六个工具详解)
5. [状态管理](#状态管理)
6. [配置管理](#配置管理)

---

## 系统概述

PGIAgent 是一个基于 ROS2 和 LangGraph 的电网巡检智能体系统，采用 **Plan-Act-Reflect**（计划-执行-反思）工作流模式。

```
┌─────────────────────────────────────────────────────────────────┐
│                        PGIAgent 系统架构                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│   │   用户任务    │───▶│ Plan-Act-    │───▶│   执行结果    │    │
│   │  （自然语言） │     │   Reflect    │     │   (任务报告)  │    │
│   └──────────────┘     │   Workflow   │     └──────────────┘    │
│                        └──────────────┘                         │
│                               │                                 │
│                               ▼                                 │
│                        ┌──────────────┐                         │
│                        │  ToolManager │                         │
│                        │  (工具管理器)  │                        │
│                        └──────────────┘                         │
│                               │                                 │
│         ┌─────────┬───────────┼───────────┬─────────┐           │
│         ▼         ▼           ▼           ▼         ▼           │
│   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │
│   │  move  │ │  yolo  │ │  VLM   │ │ track  │ │ check  │        │
│   │        │ │ _detect│ │ _detect│ │        │ │obstacle│        │
│   └────────┘ └────────┘ └────────┘ └────────┘ └────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 核心技术栈

| 层次 | 技术 |
|------|------|
| 工作流引擎 | LangGraph |
| 大模型 | DeepSeek / Qwen / 本地模型 |
| 通信框架 | ROS2 (rclpy) |
| 工具封装 | LangChain @tool |
| 视觉模型 | YOLOv11 + Qwen-VL |
| 感知硬件 | 深度相机 + 激光雷达 |

---

## Agent 工作流

### 2.1 Plan-Act-Reflect 模式

PGIAgent 采用 **Plan-Act-Reflect** 模式，这是对 ReAct (Reasoning + Acting) 模式的扩展：

```
┌──────────────────────────────────────────────────────────────────┐
│                    Plan-Act-Reflect 工作流                        │
└──────────────────────────────────────────────────────────────────┘

  ┌───────┐    ┌───────┐    ┌───────┐    ┌─────────┐    ┌───────┐
  │ Think │──▶│ Plan  │──▶│  Act   │──▶│ Examine  │──▶│  End  │
  └───────┘    └───────┘    └───────┘    └────┬────┘    └───────┘
                                              │
                                              ▼
                                        ┌───────────┐
                                        │  Reflect  │
                                        └─────┬─────┘
                                              │
                          ┌───────────────────┴───────────────────┐
                          ▼                                       ▼
                    ┌───────────┐                           ┌───────────┐
                    │   重新规划 │                          │  继续执行   │
                    └─────┬─────┘                           └─────┬─────┘
                          │                                       │
                          └───────────────┬───────────────────────┘
                                          ▼
                                    ┌───────────┐
                                    │   Act     │
                                    └───────────┘
```

### 2.2 节点详解

#### Think 节点 - 任务分析

**职责**：理解用户任务，分析任务需求

**输入**：
- 用户任务描述

**处理流程**：
1. 调用 LLM 分析任务
2. 确定任务类型（巡检、追踪、检查等）
3. 识别所需工具
4. 考虑安全因素

**使用提示词**：
```python
def get_think_prompt(task: str) -> str:
    return f"""你是一个电网巡检机器人。请仔细思考如何完成以下任务：

任务：{task}

请分析：
1. 这个任务需要哪些步骤？
2. 需要使用哪些工具？
3. 有什么安全注意事项？
4. 预期的完成标准是什么？

请用中文详细思考并回答："""
```

**输出**：
- 任务分析结果（添加到 messages）

---

#### Plan 节点 - 制定计划

**职责**：基于任务分析，制定具体的执行计划

**输入**：
- 用户任务
- Think 节点的分析结果

**处理流程**：
1. 解析 Think 节点输出
2. 生成步骤列表
3. 每个步骤关联工具
4. 考虑安全检查点

**使用提示词**：
```python
def get_plan_prompt(task: str, think_context: str = "") -> str:
    return f"""基于以下任务和思考结果，请制定详细的执行计划：

任务：{task}

之前的思考：
{think_context}

请制定具体的执行计划，格式如下：
1. 第一步要做什么
2. 第二步要做什么
3. 第三步要做什么
...

每个步骤应该：
- 明确指定要使用的工具（如需要）
- 描述具体的操作内容
- 考虑安全因素

请用中文回复，直接列出步骤编号和内容："""
```

**输出**：
- 执行计划步骤列表 (`List[str]`)
- 当前步骤索引 (`current_step_index`)

---

#### Act 节点 - 执行步骤

**职责**：使用 ReAct 模式执行当前计划步骤

**输入**：
- 当前步骤
- 已完成步骤列表
- 执行历史

**ReAct 循环**：
```
┌─────────────────────────────────────────────────────────────┐
│                    ReAct 执行循环                            │
└─────────────────────────────────────────────────────────────┘

   ┌─────────────┐
   │  生成思考   │  Thought: 需要调用YOLO检测前方物体
   └──────┬──────┘
          ▼
   ┌─────────────┐
   │  选择工具   │  Action: yolo_detect
   └──────┬──────┘
          ▼
   ┌─────────────┐
   │  调用工具   │  Action Input: {"threshold": 0.8}
   └──────┬──────┘
          ▼
   ┌─────────────┐
   │  获取结果   │  Observation: 检测到3个物体
   └──────┬──────┘
          ▼
   ┌─────────────┐
   │  判断是否   │  是否需要继续？
   │  继续      │  是 → 继续循环 / 否 → 退出
   └─────────────┘
```

**最大循环次数**：3 轮（防止无限循环）

**使用提示词**：
```python
def get_act_prompt(
    task: str,
    current_step: str,
    step_index: int,
    total_steps: int,
    past_steps: str = "",
    execution_history: str = ""
) -> str:
    return f"""请使用ReAct (Reasoning + Acting)方式执行当前步骤：

当前任务：{task}

当前步骤 ({step_index + 1}/{total_steps})：{current_step}

已完成的步骤：
{past_steps}

执行历史：
{execution_history}

请按以下格式思考和行动：
Thought: 思考需要做什么
Action: 要使用的工具名称
Action Input: 工具参数（JSON格式）
Observation: 执行结果（由系统填充）

注意：
- 每一步只能执行一个动作
- 如果当前步骤完成，请输出 "Action: 完成"
- 移动前先检查障碍物
- 保持安全距离

现在开始执行："""
```

**输出**：
- 步骤执行状态（成功/失败）
- 更新已完成步骤列表
- 推进到下一个步骤

---

#### Reflect 节点 - 反思

**职责**：反思执行过程，决定是否需要调整计划

**输入**：
- 当前步骤及状态
- 已完成步骤
- 工具调用历史

**决策**：
- **修正当前步**：小问题，继续执行
- **重新规划**：需要大幅调整，返回 Plan 节点

**使用提示词**：
```python
def get_reflect_prompt(
    task: str,
    current_step: str,
    step_status: str,
    past_steps: str = ""
) -> str:
    return f"""请反思当前的执行过程：

任务：{task}

当前步骤：{current_step}
执行状态：{step_status}

已完成步骤：
{past_steps}

请回答以下问题：
1. 当前步骤执行是否顺利？结果如何？
2. 工具调用是否都成功？
3. 是否需要调整执行计划？

请选择下一步行动：
- 如果只需要修正当前步的小问题，继续执行下一步
- 如果需要大幅调整计划，选择重新规划

请用以下格式回复：
反思结果：[你的反思]
行动选择：[修正当前步 / 重新规划]"""
```

---

#### Examine 节点 - 检查完成

**职责**：评估任务是否完成

**输入**：
- 原始任务
- 执行计划
- 已完成步骤

**决策**：
- **任务完成** → End 节点
- **任务未完成** → Reflect 节点 或 End（达到最大迭代）

**使用提示词**：
```python
def get_examine_prompt(
    task: str,
    plan: List[str],
    current_index: int,
    past_steps: str = ""
) -> str:
    return f"""请检查当前任务是否完成：

原始任务：{task}

执行计划：{plan}
已执行步骤：{current_index}/{len(plan)}

已完成步骤详情：
{past_steps}

请检查：
1. 原始任务的所有要求是否都满足了？
2. 执行计划中的所有步骤是否都完成了？
3. 是否有遗漏的子任务？

请用以下格式回复：
检查结果：[任务完成 / 任务未完成]
原因：[如果未完成，说明原因]
建议：[如果未完成，建议如何处理]"""
```

---

#### End 节点 - 生成报告

**职责**：生成最终任务报告

**输入**：
- 任务概述
- 执行情况
- 检查结果

**输出**：
- 最终任务报告

---

### 2.3 工作流条件边

```python
# Act 之后的走向
def _should_continue_after_act(self, state) -> str:
    if current_index >= len(plan):
        return "examine"      # 所有步骤完成
    if step_status == "failure":
        return "reflect"     # 执行失败
    if iteration_count >= max_iterations:
        return "examine"      # 达到最大迭代
    return "act"             # 继续执行

# Reflect 之后的走向
def _should_replan_after_reflect(self, state) -> str:
    if reflect_type == "replan":
        return "act"         # 重新规划
    return "act"             # 继续执行

# Examine 之后的走向
def _should_continue_after_examine(self, state) -> str:
    if task_completed:
        return "end"         # 任务完成
    if use_reflection and iteration_count < max_iterations:
        return "reflect"     # 继续反思
    return "end"             # 强制结束
```

---

## 工具系统架构

### 3.1 ToolManager 架构

```
┌──────────────────────────────────────────────────────────────────┐
│                       ToolManager 单例                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                    配置管理                                │  │
│   │  - AgentConfig (Agent行为配置)                            │  │
│   │  - ROS服务地址配置                                        │  │
│   │  - 默认参数                                               │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                 ROS2 服务客户端                            │  │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │  │
│   │  │  move   │ │  yolo   │ │  vlm    │ │ track   │  ...   │  │
│   │  │ Service │ │ Service │ │ Service │ │ Service │        │  │
│   │  └─────────┘ └─────────┘ └─────────┘ └─────────┘        │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                    工具方法                                │  │
│   │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐          │  │
│   │  │  move  │ │yolo_de │ │ vlm_de │ │ track  │  ...     │  │
│   │  │        │ │  tect  │ │  tect  │ │        │          │  │
│   │  └────────┘ └────────┘ └────────┘ └────────┘          │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                 模拟方法 (非ROS模式)                       │  │
│   │  用于测试和无硬件环境                                      │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 工具函数封装

LangGraph 需要将 Python 函数封装为可调用的工具：

```python
from langchain_core.tools import tool

@tool
def move_wrapper(velocity: Optional[float] = None, angle: float = 0.0, 
                seconds: Optional[float] = None) -> str:
    """移动工具"""
    result = tm.move(velocity, angle, seconds)
    return json.dumps(result, ensure_ascii=False)

@tool
def yolo_detect_wrapper(threshold: Optional[float] = None) -> str:
    """YOLO物体检测工具"""
    result = tm.yolo_detect(threshold)
    return json.dumps(result, ensure_ascii=False)

# ... 其他工具类似
```

### 3.3 工具绑定到 LLM

```python
# 创建工具函数字典
tool_functions = create_tool_functions(tool_manager)

# 绑定到 LLM
llm_with_tools = llm.bind_tools(
    list(tool_functions.values()),
    tool_choice="auto"  # 自动选择工具
)
```

---

## 六个工具详解

### 4.1 Move - 移动控制

**接口定义**：
```python
def move(velocity: Optional[float] = None, 
         angle: float = 0.0, 
         seconds: Optional[float] = None) -> Dict[str, Any]:
```

**参数说明**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| velocity | float | 0.2 m/s | 移动速度，正数前进，负数后退 |
| angle | float | 0.0° | 转向角度，0直行，正数左转，负数右转 |
| seconds | float | 2.0 s | 移动持续时间 |

**返回结果**：
```json
{
    "success": true,
    "message": "移动完成",
    "velocity": 0.2,
    "angle": 0.0,
    "seconds": 2.0
}
```

**ROS2 服务**：
- 服务名：`/pgi_agent/move`
- 服务类型：`MoveCommand.srv`

**实现逻辑**：
```python
def move(self, velocity=None, angle=0.0, seconds=None):
    # 1. 使用默认值
    velocity = velocity or self.config.default_move_velocity
    seconds = seconds or self.config.default_move_seconds
    
    # 2. 安全限制
    velocity = max(min(velocity, self.config.max_velocity), 
                  -self.config.max_velocity)
    
    # 3. 构建请求
    request = MoveCommand.Request()
    request.velocity = float(velocity)
    request.angle = float(angle)
    request.seconds = float(seconds)
    
    # 4. 调用服务
    response = self._call_service(self.config.move_service, request)
    
    # 5. 返回结果
    return {
        "success": response.success,
        "message": response.message,
        "velocity": velocity,
        "angle": angle,
        "seconds": seconds
    }
```

**模拟模式**：
```python
def _simulate_move(self, velocity, angle, seconds):
    time.sleep(min(seconds, 0.5))  # 模拟延迟
    return {
        "success": True,
        "message": f"模拟移动完成: 速度={velocity}m/s, 角度={angle}°, 时间={seconds}s",
        "velocity": velocity,
        "angle": angle,
        "seconds": seconds
    }
```

---

### 4.2 YOLO Detect - 物体检测

**接口定义**：
```python
def yolo_detect(threshold: Optional[float] = None) -> Dict[str, Any]:
```

**参数说明**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| threshold | float | 0.8 | 置信度阈值 (0.0-1.0) |

**返回结果**：
```json
{
    "success": true,
    "message": "检测完成",
    "objects": [
        {
            "name": "person",
            "confidence": 0.85,
            "distance": 2.5,
            "position": "画面中央"
        },
        {
            "name": "electric_box",
            "confidence": 0.92,
            "distance": 3.0,
            "position": "右上方"
        }
    ],
    "threshold": 0.8,
    "count": 2
}
```

**ROS2 服务**：
- 服务名：`/pgi_agent/yolo_detect`
- 服务类型：`YOLODetect.srv`

**支持的检测类别**：
- `person` - 人员
- `electric_box` - 电力控制箱
- `transformer` - 变压器
- `cable` - 电缆
- `warning_sign` - 警告标志

**实现逻辑**：
```python
def yolo_detect(self, threshold=None):
    threshold = threshold or self.config.yolo_threshold
    
    # 调用ROS服务
    request = YOLODetect.Request()
    request.threshold = float(threshold)
    response = self._call_service(self.config.yolo_service, request)
    
    # 解析检测结果
    objects = []
    if response.success:
        for i in range(len(response.objects)):
            obj = DetectedObject(
                name=response.objects[i],
                confidence=response.confidences[i],
                distance=response.distances[i],
                position=response.positions[i]
            )
            objects.append(obj.__dict__)
    
    return {
        "success": response.success,
        "message": response.message,
        "objects": objects,
        "threshold": threshold,
        "count": len(objects)
    }
```

---

### 4.3 VLM Detect - 视觉大模型场景理解

**接口定义**：
```python
def vlm_detect() -> Dict[str, Any]:
```

**参数说明**：无参数

**返回结果**：
```json
{
    "success": true,
    "message": "场景理解完成",
    "description": "这是一个变电站场景。画面中央有一个电力控制箱，箱体表面有警告标志。右侧有一个变压器设备，看起来运行正常。",
    "timestamp": 1699999999.123
}
```

**ROS2 服务**：
- 服务名：`/pgi_agent/vlm_detect`
- 服务类型：`VLMDetect.srv`

**实现逻辑**：
```python
def vlm_detect(self):
    # 调用ROS服务
    request = VLMDetect.Request()
    response = self._call_service(self.config.vlm_service, request)
    
    return {
        "success": response.success,
        "message": response.message,
        "description": response.description if response.success else "",
        "timestamp": time.time()
    }
```

**支持的模型**：
- Qwen-VL-Max（阿里千问）
- 其他兼容 OpenAI API 的 VLM

**典型应用场景**：
- 详细分析设备状态
- 判断设备是否异常
- 理解复杂场景

---

### 4.4 Track - 目标追踪

**接口定义**：
```python
def track(target: Optional[str] = None) -> Dict[str, Any]:
```

**参数说明**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| target | str | "person" | 追踪目标类别 |

**支持的追踪目标**：
- `person` - 人员
- `electric_box` - 电力控制箱
- `transformer` - 变压器
- `vehicle` - 车辆

**返回结果**：
```json
{
    "success": true,
    "message": "追踪已开始",
    "target": "person",
    "timestamp": 1699999999.123,
    "tracking_status": "active",
    "target_distance": 2.5
}
```

**ROS2 服务**：
- 服务名：`/pgi_agent/track`
- 服务类型：`Track.srv`

**实现逻辑**：
```python
def track(self, target=None):
    target = target or "person"
    
    # 调用ROS服务
    request = Track.Request()
    request.target = target
    response = self._call_service(self.config.track_service, request)
    
    return {
        "success": response.success,
        "message": response.message,
        "target": target,
        "timestamp": time.time(),
        "tracking_status": "active",
        "target_distance": 2.5
    }
```

**追踪策略**：
1. 使用 YOLO 检测目标位置
2. 计算目标与机器人相对位置
3. PID 控制跟随目标
4. 保持安全距离（默认 1.2 米）

---

### 4.5 Check Obstacle - 障碍物检测

**接口定义**：
```python
def check_obstacle() -> Dict[str, Any]:
```

**参数说明**：无参数

**返回结果**：
```json
{
    "success": true,
    "message": "前方有障碍物，建议左转15度",
    "obstacle_info": {
        "safe_direction": 15.0,
        "min_distance": 1.2,
        "safe_sectors": [true, true, true, false, false, true, true, true],
        "sector_distances": [2.0, 1.8, 1.5, 0.3, 0.4, 1.6, 1.9, 2.1]
    },
    "safe_direction": 15.0,
    "min_distance": 1.2
}
```

**ROS2 服务**：
- 服务名：`/pgi_agent/check_obstacle`
- 服务类型：`CheckObstacle.srv`

**扇区划分**：
```
          前方 (0°)
            ↑
    ┌───────┼───────┐
    │  315° │  0°   │
    │   +    │   +   │
左  │ 225°  │  45°  │ 右
侧  │   +    │   +   │ 侧
    │  180°  │  270° │
    └───────┼───────┘
          后方 (180°)
```

**障碍物等级**：
| 距离 | 等级 | 动作 |
|------|------|------|
| < 0.2m | 危险 | 立即停止 |
| < 0.3m | 警告 | 减速 |
| < 0.5m | 注意 | 谨慎通过 |
| > 0.5m | 安全 | 正常通过 |

**实现逻辑**：
```python
def check_obstacle(self):
    # 调用ROS服务
    request = CheckObstacle.Request()
    response = self._call_service(self.config.obstacle_service, request)
    
    # 解析障碍物信息
    obstacle_info = None
    if response.success:
        obstacle_info = ObstacleInfo(
            safe_direction=response.safe_direction,
            min_distance=response.min_distance,
            safe_sectors=list(response.safe_sectors),
            sector_distances=[1.0] * 8
        )
    
    return {
        "success": response.success,
        "message": response.message,
        "obstacle_info": obstacle_info.__dict__ if obstacle_info else None,
        "safe_direction": response.safe_direction,
        "min_distance": response.min_distance
    }
```

---

### 4.6 OCR - 文字识别

**接口定义**：
```python
def ocr() -> Dict[str, Any]:
```

**参数说明**：无参数

**返回结果**：
```json
{
    "success": true,
    "message": "识别完成",
    "texts": ["高压危险", "禁止入内", "变电站A区"],
    "results": [
        {
            "text": "高压危险",
            "confidence": 0.95,
            "position": "左侧上方"
        },
        {
            "text": "禁止入内",
            "confidence": 0.88,
            "position": "右侧中间"
        },
        {
            "text": "变电站A区",
            "confidence": 0.92,
            "position": "下方中间"
        }
    ],
    "count": 3
}
```

**ROS2 服务**：
- 服务名：`/pgi_agent/ocr`
- 服务类型：`OCR.srv`

**支持的语言**：
- 中文简体 (`ch_sim`)
- 英文 (`en`)

**典型识别内容**：
- 设备标签
- 警告标志
- 铭牌信息
- 编号代码

**实现逻辑**：
```python
def ocr(self):
    # 调用ROS服务
    request = OCR.Request()
    response = self._call_service(self.config.ocr_service, request)
    
    # 解析OCR结果
    ocr_results = []
    if response.success and len(response.texts) > 0:
        for i in range(len(response.texts)):
            result = OCRResult(
                text=response.texts[i],
                confidence=response.confidences[i]
            )
            ocr_results.append(result.__dict__)
    
    return {
        "success": response.success,
        "message": response.message,
        "texts": [r["text"] for r in ocr_results],
        "results": ocr_results,
        "count": len(ocr_results)
    }
```

---

## 状态管理

### 5.1 PlanActReflectState

```python
class PlanActReflectState(TypedDict):
    # 任务信息
    task: str                      # 用户任务
    iteration_count: int           # 当前迭代次数
    
    # 执行计划
    plan: List[str]               # 计划步骤列表
    current_step_index: int        # 当前步骤索引
    current_step_status: str       # 当前步骤状态
    
    # 执行历史
    messages: List[AnyMessage]     # 对话消息
    step_messages: List[AnyMessage]# 当前步骤的ReAct消息
    past_steps: List[Dict]          # 已完成步骤
    
    # 反思和检查
    reflection: str                # 反思内容
    reflect_type: str              # 反思类型
    examine_result: str            # 检查结果
    task_completed: bool           # 任务是否完成
    
    # 最终结果
    final_answer: str              # 最终答案
```

### 5.2 AgentConfig

```python
class AgentConfig:
    # 模型配置
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    
    # 执行配置
    max_iterations: int = 20
    use_reflection: bool = True
    
    # 安全配置
    max_velocity: float = 0.5
    min_obstacle_distance: float = 0.3
    
    # ROS配置
    ros_enabled: bool = True
    
    # 工具默认参数
    default_move_velocity: float = 0.2
    default_move_seconds: float = 2.0
    yolo_threshold: float = 0.8
    tracking_distance: float = 1.2
```

---

## 配置管理

### 6.1 配置文件结构

**agent_config.yaml** - Agent 行为配置：
```yaml
agent:
  name: "PowerGridInspectionAgent"
  llm_provider: "deepseek"
  llm_model: "deepseek-chat"
  max_iterations: 20
  use_reflection: true
  ros_enabled: true
```

**ros_params.yaml** - ROS 工具节点配置：
```yaml
perception_node:
  ros__parameters:
    model_path: "yolo11n.engine"
    rgb_topic: "/depth_cam/rgb/image_raw"
    use_gpu: true

move_node:
  ros__parameters:
    default_velocity: 0.2
    max_velocity: 0.5
    cmd_vel_topic: "/controller/cmd_vel"
```

---

## 使用示例

### 7.1 基本使用

```python
from PGIAgent.agent import create_plan_act_reflect_agent

# 创建 Agent
agent = create_plan_act_reflect_agent()

# 执行任务
result = agent.run("检查变电站A区的设备状态", max_iterations=10)

print(f"成功: {result['success']}")
print(f"结果: {result['final_answer']}")
```

### 7.2 流式输出

```python
# 同步流式输出
for event in agent.stream("追踪一名维护人员"):
    print(f"[{event['node']}] {event['content']}")

# 异步流式输出
async for event in agent.stream_async("检查设备状态"):
    print(f"[{event['node']}] {event['content']}")
```

### 7.3 ROS2 集成

```python
import rclpy
from PGIAgent.agent import create_plan_act_reflect_agent

class AgentNode(Node):
    def __init__(self):
        super().__init__('pgi_agent_node')
        self.agent = create_plan_act_reflect_agent(ros_node=self)
        
        # 创建服务
        self.create_service(
            Trigger, 
            '/pgi_agent/execute_task',
            self.execute_task_callback
        )
    
    def execute_task_callback(self, request, response):
        result = self.agent.run(request.task)
        response.success = result['success']
        response.message = result['final_answer']
        return response
```

---

## 附录

### A. 服务定义

**MoveCommand.srv**：
```yaml
float64 velocity
float64 angle
float64 seconds
---
bool success
string message
```

**YOLODetect.srv**：
```yaml
float64 threshold
---
bool success
string message
string[] objects
float64[] confidences
float64[] distances
string[] positions
```

**VLMDetect.srv**：
```yaml
---
bool success
string message
string description
string[] objects_detected
string scene_type
```

**Track.srv**：
```yaml
string target
---
bool success
string message
```

**CheckObstacle.srv**：
```yaml
---
bool success
string message
float64 safe_direction
float64 min_distance
bool[] sector_ranges
bool[] has_obstacle
```

**OCR.srv**：
```yaml
---
bool success
string message
string[] texts
float64[] confidences
string[] positions
```

### B. 错误处理

所有工具方法都包含完善的错误处理：
- 服务超时：返回 `{"success": False, "message": "服务调用超时"}`
- 服务不可用：返回 `{"success": False, "message": "服务不可用"}`
- 异常捕获：返回 `{"success": False, "message": "具体错误信息"}`

### C. 模拟模式

在没有 ROS 硬件或需要离线测试时，可以启用模拟模式：

```python
# 配置中设置 ros_enabled = False
config = AgentConfig(ros_enabled=False)
agent = create_plan_act_reflect_agent(config)

# 此时所有工具调用会返回模拟数据
result = agent.run("检查设备")
```

---

*本文档版本: 1.0*
*最后更新: 2026*
