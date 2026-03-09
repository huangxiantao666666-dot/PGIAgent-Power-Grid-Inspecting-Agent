"""
智能体状态管理
定义LangGraph智能体的状态结构
"""

from typing import TypedDict, List, Optional, Dict, Any, Annotated, Literal, Tuple
from dataclasses import dataclass, field
from enum import Enum
# 导入time模块用于时间戳
import time
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, AnyMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """智能体状态定义"""
    # 任务相关
    task: str  # 当前任务描述
    current_step_num: int  # 当前步骤
    past_steps: List[Tuple[str, str, str]] # 已完成的步骤 [(步骤, 结果, 状态)]
    final_answer: Optional[str]
            
    # 决策相关
    plan: List[str]  # 执行计划
    iteration_count: int  # 迭代次数
    last_step_status: Literal["success", "failure", "pending"]  # 上一步执行结果
    
    # 对话历史
    messages: Annotated[List[AnyMessage], add_messages]  # 对话消息历史
    
    
def create_initial_state(task: str) -> AgentState:
    """创建初始状态"""
    return AgentState(
        task=task,
        current_step_num=0,
        past_steps=[],
        final_answer=None,
        plan=[],
        last_step_status="pending",
        iteration_count=0,
    )


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
    max_iterations: int = 10
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



