"""
提示词模板
定义智能体使用的各种提示词模板
"""

from typing import Dict, Any
from .state import AgentState, get_state_summary


def get_system_prompt() -> str:
    """获取系统提示词"""
    return """你是一个电网巡检机器小车的智能体，负责执行电网巡检任务。

你的能力包括：
1. 移动控制：控制机器人前进、后退、转向
2. 视觉感知：使用YOLO检测物体，使用视觉大模型分析场景
3. 目标追踪：追踪特定物体并保持安全距离
4. 障碍物检测：使用激光雷达检测安全方向
5. 文字识别：读取设备标签和警告标志

你可以使用的工具：
1. move(velocity, angle, seconds) - 控制机器人移动
   - velocity: 移动速度 (m/s)，正数为前进，负数为后退
   - angle: 转向角度 (度)，0为直行，正数为左转，负数为右转
   - seconds: 移动时间 (秒)

2. yolo_detect(threshold=0.8) - YOLO物体检测
   - threshold: 置信度阈值 (0.0-1.0)
   - 返回：检测到的物体列表、距离和位置描述

3. VLM_detect() - 视觉大模型场景理解
   - 返回：详细的场景描述

4. track(which="person") - 目标追踪
   - which: 追踪目标 (如"person", "electric_box")
   - 返回：追踪结果

5. check_obstacle() - 障碍物检测
   - 返回：安全方向、最小障碍物距离、安全扇区

6. ocr() - 文字识别
   - 返回：识别出的文本列表和置信度

安全规则：
1. 移动前必须检查障碍物
2. 保持与障碍物的安全距离（至少0.3米）
3. 追踪人员时保持1.2米安全距离
4. 遇到紧急情况立即停止

请根据用户的任务要求，制定合理的执行计划，并安全地完成任务。"""


def get_planning_prompt(task: str) -> str:
    """获取规划提示词"""
    return f"""请为以下电网巡检任务制定详细的执行计划：

任务：{task}

请按照以下步骤制定计划：
1. 分析任务需求，确定需要使用的工具
2. 制定具体的执行步骤
3. 考虑安全因素，在移动前加入障碍物检查
4. 确保步骤逻辑清晰、可执行

请用中文回复，列出具体的执行步骤。每个步骤应该明确指定要使用的工具或操作。

示例格式：
1. 使用YOLO检测当前场景中的物体
2. 根据检测结果，使用VLM详细分析场景
3. 检查障碍物，确定安全移动方向
4. 移动到目标位置
5. 使用OCR读取设备标签
6. 完成巡检任务

现在请为上述任务制定计划："""


def get_reflection_prompt(state: AgentState) -> str:
    """获取反思提示词"""
    state_summary = get_state_summary(state)
    
    return f"""请反思当前的执行过程：

当前状态摘要：
{state_summary}

任务：{state['task']}
当前步骤：{state['current_step'] + 1}/{len(state['plan']) if state['plan'] else 0}
已执行步骤：{state['plan'][:state['current_step']] if state['current_step'] > 0 else '无'}
工具调用历史：{len(state['tool_calls'])}次

请回答以下问题：
1. 当前执行是否顺利？遇到了什么问题？
2. 工具调用是否都成功？失败的原因是什么？
3. 执行计划是否需要调整？如何调整？
4. 下一步应该做什么？

请提供具体的反思和建议："""


def get_tool_selection_prompt(step: str, available_tools: list) -> str:
    """获取工具选择提示词"""
    tools_list = "\n".join([f"- {tool}" for tool in available_tools])
    
    return f"""请为以下步骤选择合适的工具：

步骤：{step}

可用工具：
{tools_list}

请根据步骤描述，选择最合适的工具，并说明理由。
如果需要调用工具，请指定工具名称和参数。
如果不需要工具，请说明原因。

请用以下格式回复：
工具：[工具名称或"无"]
理由：[选择理由]
参数：[参数列表，如无参数则写"无"]"""


def get_execution_prompt(state: AgentState, current_step: str) -> str:
    """获取执行提示词"""
    previous_steps = state['plan'][:state['current_step']] if state['current_step'] > 0 else []
    previous_steps_str = "\n".join([f"{i+1}. {step}" for i, step in enumerate(previous_steps)])
    
    return f"""请执行当前步骤：

任务：{state['task']}
已完成的步骤：
{previous_steps_str if previous_steps_str else "无"}

当前步骤：{current_step}

当前状态：
- 检测到的物体：{len(state['detected_objects'])}个
- 场景描述：{state['current_scene'][:100] + '...' if state['current_scene'] else '无'}
- 障碍物信息：{'有' if state['obstacle_info'] else '无'}
- OCR结果：{len(state['ocr_results'])}条

请决定如何执行当前步骤：
1. 如果需要调用工具，请指定工具名称和参数
2. 如果不需要工具，请说明如何完成该步骤
3. 考虑安全因素，确保操作安全

请用以下格式回复：
行动：[工具调用或具体操作]
理由：[行动理由]
参数：[参数列表，如无参数则写"无"]"""


def get_error_recovery_prompt(state: AgentState, error_message: str) -> str:
    """获取错误恢复提示词"""
    return f"""工具调用失败，需要恢复：

错误信息：{error_message}

当前状态：
{get_state_summary(state)}

请分析失败原因，并提出恢复方案：
1. 错误原因分析
2. 是否重试？如果重试，需要调整什么参数？
3. 是否有替代方案？
4. 是否需要调整执行计划？

请提供具体的恢复建议："""


def get_task_completion_prompt(state: AgentState) -> str:
    """获取任务完成提示词"""
    return f"""任务即将完成：

任务：{state['task']}
完成状态：{state['current_step']}/{len(state['plan'])} 步骤

执行总结：
- 总迭代次数：{state['iteration_count']}
- 工具调用次数：{len(state['tool_calls'])}
- 成功工具调用：{sum(1 for tc in state['tool_calls'] if tc.get('success', False))}
- 检测到的物体：{len(state['detected_objects'])}
- OCR识别结果：{len(state['ocr_results'])}

请评估任务完成情况：
1. 任务是否成功完成？
2. 遇到了哪些挑战？如何解决的？
3. 有哪些可以改进的地方？
4. 对下次执行类似任务的建议？

请提供任务完成评估："""


def get_safety_check_prompt(action: str, state: AgentState) -> str:
    """获取安全检查提示词"""
    return f"""请进行安全检查：

计划执行的操作：{action}

当前安全状态：
- 最小障碍物距离：{state['obstacle_info'].get('min_distance', '未知') if state['obstacle_info'] else '未知'}米
- 安全方向：{state['obstacle_info'].get('safe_direction', '未知') if state['obstacle_info'] else '未知'}度
- 电池电量：{state['battery_level'] if state['battery_level'] else '未知'}

请检查以下安全事项：
1. 移动操作前是否检查了障碍物？
2. 是否保持了安全距离？
3. 电池电量是否充足？
4. 是否有紧急停止的需要？

请给出安全检查结果和建议："""


# 特定任务的提示词模板
def get_power_inspection_prompt(area: str) -> str:
    """获取电力巡检特定提示词"""
    return f"""执行变电站{area}巡检任务：

重点检查项目：
1. 电力设备状态（变压器、开关柜、控制箱）
2. 安全标志和警告标签
3. 电缆和连接状态
4. 环境安全隐患

巡检流程：
1. 进入{area}区域
2. 全面扫描环境，识别所有电力设备
3. 逐个检查设备状态
4. 读取设备标签和警告标志
5. 记录异常情况
6. 生成巡检报告

安全注意事项：
1. 保持与高压设备的安全距离
2. 注意地面电缆，避免碾压
3. 识别警告标志，遵守安全规定
4. 遇到异常立即停止并报告"""


def get_person_tracking_prompt() -> str:
    """获取人员追踪特定提示词"""
    return """执行人员追踪任务：

追踪目标：维护人员
安全距离：1.2米
追踪策略：
1. 使用YOLO检测人员位置
2. 启动追踪模式
3. 保持安全距离跟随
4. 避障移动
5. 人员停止时保持观察

安全规则：
1. 始终保持1.2米安全距离
2. 人员进入危险区域时发出警告
3. 追踪过程中持续检查障碍物
4. 人员消失时停止追踪并报告"""


def get_equipment_check_prompt(equipment_type: str) -> str:
    """获取设备检查特定提示词"""
    return f"""执行{equipment_type}设备检查：

检查项目：
1. 设备外观是否完好
2. 指示灯状态
3. 标签和铭牌信息
4. 连接部位状态
5. 周围环境安全

检查步骤：
1. 移动到设备前方安全位置
2. 使用VLM详细观察设备状态
3. 使用OCR读取设备标签
4. 检查设备周围环境
5. 记录检查结果

安全要求：
1. 保持与设备的适当距离
2. 不接触任何设备
3. 发现异常立即记录并报告
4. 遵守设备区域的安全规定"""