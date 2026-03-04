"""
PGIAgent - 电网巡检机器小车智能体
基于ROS2和LangGraph的智能体系统
"""

from .state import (
    AgentState, TaskStatus, ToolType, DetectedObject, 
    ObstacleInfo, OCRResult, RobotState, ToolCall, AgentConfig,
    create_initial_state, update_state_with_tool_result, get_state_summary
)

from .tools import ToolManager, create_tool_functions, load_tool_config
from .prompts import (
    get_system_prompt, get_planning_prompt, get_reflection_prompt,
    get_tool_selection_prompt, get_execution_prompt, get_error_recovery_prompt,
    get_task_completion_prompt, get_safety_check_prompt,
    get_power_inspection_prompt, get_person_tracking_prompt, get_equipment_check_prompt
)

from .agent_graph import AgentGraph, create_agent, run_agent_task

__version__ = "0.1.0"
__author__ = "PGIAgent Team"
__description__ = "Power Grid Inspection Agent for JetAuto robot with ROS2 and LangGraph integration"

__all__ = [
    # 状态管理
    "AgentState", "TaskStatus", "ToolType", "DetectedObject",
    "ObstacleInfo", "OCRResult", "RobotState", "ToolCall", "AgentConfig",
    "create_initial_state", "update_state_with_tool_result", "get_state_summary",
    
    # 工具管理
    "ToolManager", "create_tool_functions", "load_tool_config",
    
    # 提示词模板
    "get_system_prompt", "get_planning_prompt", "get_reflection_prompt",
    "get_tool_selection_prompt", "get_execution_prompt", "get_error_recovery_prompt",
    "get_task_completion_prompt", "get_safety_check_prompt",
    "get_power_inspection_prompt", "get_person_tracking_prompt", "get_equipment_check_prompt",
    
    # 智能体工作流
    "AgentGraph", "create_agent", "run_agent_task",
]