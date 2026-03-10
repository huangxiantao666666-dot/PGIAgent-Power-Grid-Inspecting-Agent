"""
智能体状态管理
定义LangGraph智能体的状态结构

注意：此文件只定义Agent工作流相关的状态，不包含工具配置
工具配置应该在 tools.py 或独立的配置文件中管理
"""

from typing import TypedDict, List, Optional, Dict, Any, Annotated, Literal
from dataclasses import dataclass, field
from enum import Enum
import time
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


# ========== Plan-Act-Reflect Agent 状态 ==========

class PlanActReflectState(TypedDict):
    """Plan-Act-Reflect Agent 状态定义"""
    # 任务相关
    task: str  # 用户任务描述
    
    # 执行计划
    plan: List[str]  # 执行计划步骤列表
    current_step_index: int  # 当前执行的步骤索引
    
    # 历史记录
    past_steps: List[Dict[str, Any]]  # 已完成步骤记录
    
    # ReAct执行相关
    step_messages: Annotated[List[AnyMessage], add_messages]  # ReAct节点的对话历史
    current_step_status: Literal["success", "failure", "pending", "completed"]
    
    # 反思相关
    reflection: Optional[str]  # 反思结果
    reflect_type: Optional[Literal["modify_current", "replan"]]
    
    # 任务完成相关
    task_completed: bool  # 任务是否完成
    final_answer: Optional[str]  # 最终答案
    examine_result: Optional[str]  # 检查结果
    
    # 对话历史 (用于think, plan, reflect, examine节点)
    messages: Annotated[List[AnyMessage], add_messages]
    
    # 迭代控制
    iteration_count: int  # 总迭代次数
    max_iterations: int  # 最大迭代次数


def create_initial_state(task: str, max_iterations: int = 20) -> PlanActReflectState:
    """创建Plan-Act-Reflect Agent初始状态"""
    return PlanActReflectState(
        task=task,
        plan=[],
        current_step_index=0,
        past_steps=[],
        step_messages=[],
        current_step_status="pending",
        reflection=None,
        reflect_type=None,
        task_completed=False,
        final_answer=None,
        examine_result=None,
        messages=[],
        iteration_count=0,
        max_iterations=max_iterations
    )


# ========== 旧版 AgentState (保留兼容) ==========

class AgentState(TypedDict):
    """旧版智能体状态定义（兼容）"""
    task: str
    current_step_num: int
    past_steps: List  # [(步骤, 结果, 状态)]
    final_answer: Optional[str]
    plan: List[str]
    iteration_count: int
    last_step_status: Literal["success", "failure", "pending"]
    messages: Annotated[List[AnyMessage], add_messages]


def create_agent_initial_state(task: str) -> AgentState:
    """创建旧版Agent初始状态（兼容）"""
    return AgentState(
        task=task,
        current_step_num=0,
        past_steps=[],
        final_answer=None,
        plan=[],
        last_step_status="pending",
        iteration_count=0,
        messages=[],
    )


# ========== 数据类定义 ==========

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


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
    name: str
    confidence: float
    distance: float
    position: str
    bbox: Optional[List[float]] = None
    depth: Optional[float] = None


@dataclass
class ObstacleInfo:
    """障碍物信息"""
    safe_direction: float
    min_distance: float
    safe_sectors: List[bool]
    sector_distances: List[float]


@dataclass
class OCRResult:
    """OCR结果"""
    text: str
    confidence: float
    bbox: Optional[List[float]] = None


# ========== Agent 配置类 ==========

@dataclass
class AgentConfig:
    """
    智能体配置
    
    注意：只包含Agent行为相关的配置
    工具配置应该通过ToolManager或独立配置文件管理
    """
    # 模型配置
    llm_provider: str = "deepseek"  # deepseek, qwen, local
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    
    # 行为配置
    max_iterations: int = 20
    use_reflection: bool = True
    reflection_interval: int = 3  # 每N步反思一次
    
    # 安全配置
    max_velocity: float = 0.5
    min_obstacle_distance: float = 0.3
    
    # 执行模式
    ros_enabled: bool = True  # 是否启用ROS模式
    stream_output: bool = True  # 是否流式输出
    
    def __post_init__(self):
        """验证配置"""
        if self.llm_provider not in ["deepseek", "qwen", "local", "mock"]:
            raise ValueError(f"不支持的LLM provider: {self.llm_provider}")
        
        if self.max_iterations <= 0:
            raise ValueError("max_iterations必须大于0")


def load_agent_config(config_path: Optional[str] = None) -> AgentConfig:
    """从配置文件加载Agent配置"""
    import os
    import yaml
    from dotenv import load_dotenv
    
    load_dotenv()
    
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), 
            '../../config/agent_config.yaml'
        )
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        return AgentConfig()
    
    agent_config = config_data.get('agent', {}) if config_data else {}
    
    config = AgentConfig()
    config.llm_provider = agent_config.get('llm_provider', config.llm_provider)
    config.llm_model = agent_config.get('llm_model', config.llm_model)
    config.llm_temperature = agent_config.get('llm_temperature', config.llm_temperature)
    config.llm_max_tokens = agent_config.get('llm_max_tokens', config.llm_max_tokens)
    config.max_iterations = agent_config.get('max_iterations', config.max_iterations)
    config.use_reflection = agent_config.get('use_reflection', config.use_reflection)
    
    safety = agent_config.get('safety', {})
    config.max_velocity = safety.get('max_velocity', config.max_velocity)
    config.min_obstacle_distance = safety.get('min_obstacle_distance', config.min_obstacle_distance)
    
    return config


# ========== 辅助函数 ==========

def get_state_summary(state: PlanActReflectState) -> str:
    """获取状态的文本摘要"""
    task = state.get("task", "无任务")
    current_step = state.get("current_step_index", 0)
    plan = state.get("plan", [])
    iteration = state.get("iteration_count", 0)
    last_status = state.get("current_step_status", "pending")
    past_steps = state.get("past_steps", [])
    
    summary = f"""
任务: {task}
当前步骤: {current_step}/{len(plan) if plan else 0}
迭代次数: {iteration}
当前状态: {last_status}
已完成步骤数: {len(past_steps)}
"""
    return summary.strip()
