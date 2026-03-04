"""
智能体状态管理
定义LangGraph智能体的状态结构
"""

from typing import TypedDict, List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
# 导入time模块用于时间戳
import time


class AgentState(TypedDict):
    """智能体状态定义"""
    # 任务相关
    task: str  # 当前任务描述
    task_history: List[str]  # 任务历史
    current_step: int  # 当前步骤
    
    # 环境感知
    current_scene: Optional[str]  # 当前场景描述
    detected_objects: List[Dict[str, Any]]  # 检测到的物体
    obstacle_info: Optional[Dict[str, Any]]  # 障碍物信息
    ocr_results: List[Dict[str, Any]]  # OCR结果
    
    # 机器人状态
    robot_position: Optional[Dict[str, float]]  # 机器人位置
    robot_orientation: Optional[float]  # 机器人朝向
    battery_level: Optional[float]  # 电池电量
    
    # 工具调用
    tool_calls: List[Dict[str, Any]]  # 工具调用历史
    last_tool_result: Optional[str]  # 上次工具调用结果
    
    # 决策相关
    plan: List[str]  # 执行计划
    reflection: Optional[str]  # 反思结果
    iteration_count: int  # 迭代次数
    
    # 对话历史
    messages: List[Dict[str, str]]  # 对话消息历史


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"  # 等待执行
    PLANNING = "planning"  # 规划中
    EXECUTING = "executing"  # 执行中
    PAUSED = "paused"  # 暂停
    COMPLETED = "completed"  # 完成
    FAILED = "failed"  # 失败


class ToolType(Enum):
    """工具类型枚举"""
    MOVE = "move"
    YOLO_DETECT = "yolo_detect"
    VLM_DETECT = "vlm_detect"
    TRACK = "track"
    CHECK_OBSTACLE = "check_obstacle"
    OCR = "ocr"


@dataclass
class DetectedObject:
    """检测到的物体"""
    name: str  # 物体名称
    confidence: float  # 置信度 (0.0-1.0)
    distance: float  # 距离 (米)
    position: str  # 位置描述 (如"左下方")
    bbox: Optional[List[float]] = None  # 边界框 [x1, y1, x2, y2]
    depth: Optional[float] = None  # 深度值


@dataclass
class ObstacleInfo:
    """障碍物信息"""
    safe_direction: float  # 安全方向 (度)
    min_distance: float  # 最小障碍物距离 (米)
    safe_sectors: List[bool]  # 安全扇区 (8个扇区)
    sector_distances: List[float]  # 每个扇区的距离


@dataclass
class OCRResult:
    """OCR结果"""
    text: str  # 识别文本
    confidence: float  # 置信度
    bbox: Optional[List[float]] = None  # 文本位置


@dataclass
class RobotState:
    """机器人状态"""
    position: Dict[str, float]  # 位置 {x, y, z}
    orientation: float  # 朝向 (弧度)
    velocity: Dict[str, float]  # 速度 {linear, angular}
    battery: float  # 电池电量 (0.0-1.0)
    timestamp: float  # 时间戳


@dataclass
class ToolCall:
    """工具调用记录"""
    tool_type: ToolType
    parameters: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    timestamp: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class AgentConfig:
    """智能体配置"""
    # 模型配置
    llm_provider: str = "deepseek"  # deepseek, qwen, local
    llm_model: str = "deepseek-chat"
    vlm_provider: str = "qwen"  # qwen, local
    vlm_model: str = "qwen-vl-max"
    
    # 行为配置
    max_iterations: int = 20
    use_reflection: bool = True
    reflection_depth: int = 2
    max_retries: int = 3
    
    # 安全配置
    max_velocity: float = 0.5
    max_angular_velocity: float = 1.0
    min_obstacle_distance: float = 0.3
    emergency_stop_distance: float = 0.2
    
    # 工具配置
    default_move_velocity: float = 0.2
    default_move_seconds: float = 2.0
    yolo_threshold: float = 0.8
    tracking_distance: float = 1.2
    
    # ROS配置
    ros_enabled: bool = True
    move_service: str = "/pgi_agent/move"
    yolo_service: str = "/pgi_agent/yolo_detect"
    vlm_service: str = "/pgi_agent/vlm_detect"
    track_service: str = "/pgi_agent/track"
    obstacle_service: str = "/pgi_agent/check_obstacle"
    ocr_service: str = "/pgi_agent/ocr"


def create_initial_state(task: str) -> AgentState:
    """创建初始状态"""
    return AgentState(
        task=task,
        task_history=[],
        current_step=0,
        current_scene=None,
        detected_objects=[],
        obstacle_info=None,
        ocr_results=[],
        robot_position=None,
        robot_orientation=None,
        battery_level=None,
        tool_calls=[],
        last_tool_result=None,
        plan=[],
        reflection=None,
        iteration_count=0,
        messages=[
            {"role": "system", "content": "你是一个电网巡检机器小车的智能体，负责执行电网巡检任务。"},
            {"role": "user", "content": task}
        ]
    )


def update_state_with_tool_result(
    state: AgentState,
    tool_type: str,
    parameters: Dict[str, Any],
    result: Dict[str, Any],
    success: bool = True,
    error_message: Optional[str] = None
) -> AgentState:
    """使用工具结果更新状态"""
    tool_call = {
        "tool_type": tool_type,
        "parameters": parameters,
        "result": result,
        "timestamp": time.time() if 'time' in locals() else 0.0,
        "success": success,
        "error_message": error_message
    }
    
    state["tool_calls"].append(tool_call)
    state["last_tool_result"] = str(result) if result else error_message
    
    # 根据工具类型更新特定状态
    if tool_type == ToolType.YOLO_DETECT.value and success:
        state["detected_objects"] = result.get("objects", [])
    elif tool_type == ToolType.VLM_DETECT.value and success:
        state["current_scene"] = result.get("description", "")
    elif tool_type == ToolType.CHECK_OBSTACLE.value and success:
        state["obstacle_info"] = result
    elif tool_type == ToolType.OCR.value and success:
        state["ocr_results"] = result.get("texts", [])
    
    return state


def get_state_summary(state: AgentState) -> str:
    """获取状态摘要"""
    summary = f"任务: {state['task']}\n"
    summary += f"当前步骤: {state['current_step']}/{len(state['plan']) if state['plan'] else 0}\n"
    summary += f"迭代次数: {state['iteration_count']}\n"
    
    if state['current_scene']:
        summary += f"当前场景: {state['current_scene'][:100]}...\n"
    
    if state['detected_objects']:
        summary += f"检测到物体: {len(state['detected_objects'])}个\n"
    
    if state['obstacle_info']:
        summary += f"障碍物信息: 安全方向 {state['obstacle_info'].get('safe_direction', 0)}度\n"
    
    if state['tool_calls']:
        summary += f"工具调用: {len(state['tool_calls'])}次\n"
    
    return summary


