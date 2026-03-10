"""
PGIAgent - 电网巡检机器小车智能体
基于ROS2和LangGraph的智能体系统
"""

from .state import (
    PlanActReflectState, AgentState,
    TaskStatus, ToolType, DetectedObject, 
    ObstacleInfo, OCRResult,
    AgentConfig,
    create_initial_state, create_agent_initial_state,
    load_agent_config, get_state_summary
)

from .tools import (
    ToolManager, create_tool_functions, 
    create_async_tool_functions, get_tool_manager
)

from .prompts import (
    get_system_prompt, get_planning_prompt, 
    get_reflection_prompt, get_reflect_prompt,
    get_tool_selection_prompt, get_execution_prompt, get_error_recovery_prompt,
    get_task_completion_prompt, get_safety_check_prompt,
    get_power_inspection_prompt, get_person_tracking_prompt, get_equipment_check_prompt,
    get_think_prompt, get_plan_prompt, get_act_prompt, 
    get_examine_prompt, get_summary_prompt
)

# Plan-Act-Reflect Agent (新版)
from .agent import (
    PlanActReflectAgent, 
    create_plan_act_reflect_agent, 
    run_task
)

# 旧版Agent (兼容)
from .agent_graph import AgentGraph, create_agent, run_agent_task

__version__ = "0.2.0"
__author__ = "PGIAgent Team"
__description__ = "Power Grid Inspection Agent for JetAuto robot with ROS2 and LangGraph"

__all__ = [
    # 状态管理
    "PlanActReflectState", "AgentState",
    "TaskStatus", "ToolType", "DetectedObject",
    "ObstacleInfo", "OCRResult",
    "AgentConfig",
    "create_initial_state", "create_agent_initial_state",
    "load_agent_config", "get_state_summary",
    
    # 工具管理
    "ToolManager", "create_tool_functions", 
    "create_async_tool_functions", "get_tool_manager",
    
    # 提示词模板
    "get_system_prompt", "get_planning_prompt", "get_reflection_prompt",
    "get_tool_selection_prompt", "get_execution_prompt", "get_error_recovery_prompt",
    "get_task_completion_prompt", "get_safety_check_prompt",
    "get_power_inspection_prompt", "get_person_tracking_prompt", "get_equipment_check_prompt",
    
    # Plan-Act-Reflect Agent
    "PlanActReflectAgent", "create_plan_act_reflect_agent", "run_task",
    
    # 旧版 (兼容)
    "AgentGraph", "create_agent", "run_agent_task",
]
