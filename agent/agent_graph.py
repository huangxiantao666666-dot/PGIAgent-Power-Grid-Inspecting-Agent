"""
智能体工作流定义
使用LangGraph构建电网巡检智能体的工作流
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
import json
import time
import os

from .state import AgentState, create_initial_state, update_state_with_tool_result, get_state_summary
from .tools import ToolManager, create_tool_functions, load_tool_config
from .prompts import get_system_prompt, get_planning_prompt, get_reflection_prompt


class AgentGraph:
    """智能体工作流管理器"""
    
    def __init__(self, config_path: Optional[str] = None, ros_node=None):
        # 加载配置
        if config_path and os.path.exists(config_path):
            self.config = load_tool_config(config_path)
        else:
            from .state import AgentConfig
            self.config = AgentConfig()
        
        # 初始化工具管理器
        self.tool_manager = ToolManager(self.config, ros_node)
        self.tool_functions = create_tool_functions(self.tool_manager)
        
        # 初始化大模型
        self.llm = self._init_llm()
        
        # 构建工作流
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
    
    def _init_llm(self):
        """初始化大模型"""
        if self.config.llm_provider == "deepseek":
            # 使用DeepSeek API
            import os
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
            base_url = self.config.llm_model_config.get("base_url", "https://api.deepseek.com")
            
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.config.llm_model,
                api_key=api_key,
                base_url=base_url,
                temperature=0.7,
                max_tokens=2000
            )
        elif self.config.llm_provider == "qwen":
            # 使用通义千问API
            import os
            api_key = os.getenv("QWEN_API_KEY", "")
            
            from langchain_community.chat_models.tongyi import ChatTongyi
            return ChatTongyi(
                model=self.config.llm_model,
                api_key=api_key,
                temperature=0.7,
                max_tokens=2000
            )
        elif self.config.llm_provider == "local":
            # 使用本地Ollama
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                model=self.config.llm_model,
                temperature=0.8,
                num_ctx=4096
            )
        else:
            # 默认使用模拟LLM
            return self._create_mock_llm()
    
    def _create_mock_llm(self):
        """创建模拟LLM用于测试"""
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import BaseMessage, AIMessage
        from langchain_core.outputs import ChatResult, ChatGeneration
        
        class MockLLM(BaseChatModel):
            def _generate(self, messages, stop=None, **kwargs):
                # 模拟LLM响应
                last_message = messages[-1].content if messages else ""
                
                # 根据消息内容生成响应
                if "规划" in last_message or "plan" in last_message.lower():
                    response = """基于当前任务，我制定以下执行计划：
1. 首先使用YOLO检测当前场景中的物体
2. 根据检测结果，使用VLM详细分析场景
3. 检查障碍物，确定安全移动方向
4. 移动到目标位置
5. 使用OCR读取设备标签
6. 完成巡检任务"""
                elif "反思" in last_message or "reflect" in last_message.lower():
                    response = "反思：当前执行顺利，所有工具调用都成功完成。下一步可以继续执行剩余任务。"
                else:
                    response = "我理解了任务要求。我将使用以下工具：move, yolo_detect, VLM_detect, track, check_obstacle, ocr。开始执行任务。"
                
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])
            
            @property
            def _llm_type(self):
                return "mock"
        
        return MockLLM()
    
    def _build_workflow(self):
        """构建LangGraph工作流"""
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("execute", self._execute_node)
        workflow.add_node("reflect", self._reflect_node)
        workflow.add_node("tools", self._tools_node)
        
        # 设置入口点
        workflow.set_entry_point("plan")
        
        # 添加边
        workflow.add_edge("plan", "execute")
        workflow.add_conditional_edges(
            "execute",
            self._should_continue,
            {
                "continue": "execute",
                "reflect": "reflect",
                "complete": END
            }
        )
        workflow.add_edge("reflect", "execute")
        workflow.add_edge("tools", "execute")
        
        return workflow
    
    def _plan_node(self, state: AgentState) -> Dict[str, Any]:
        """规划节点：制定执行计划"""
        # 获取系统提示词
        system_prompt = get_system_prompt()
        planning_prompt = get_planning_prompt(state["task"])
        
        # 构建消息
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=planning_prompt)
        ]
        
        # 调用LLM生成计划
        response = self.llm.invoke(messages)
        
        # 解析计划
        plan_text = response.content
        plan_steps = self._parse_plan(plan_text)
        
        # 更新状态
        state["plan"] = plan_steps
        state["messages"].append({"role": "assistant", "content": f"执行计划：\n{plan_text}"})
        
        return state
    
    def _execute_node(self, state: AgentState) -> Dict[str, Any]:
        """执行节点：执行当前步骤"""
        # 检查是否完成所有步骤
        if state["current_step"] >= len(state["plan"]):
            state["messages"].append({"role": "assistant", "content": "所有步骤已完成！"})
            return state
        
        # 获取当前步骤
        current_step = state["plan"][state["current_step"]]
        state["messages"].append({"role": "assistant", "content": f"执行步骤 {state['current_step']+1}: {current_step}"})
        
        # 分析步骤并决定使用哪个工具
        tool_to_use = self._determine_tool(current_step)
        
        if tool_to_use:
            # 需要调用工具
            state["messages"].append({"role": "assistant", "content": f"将使用工具: {tool_to_use}"})
            # 工具调用将在tools节点处理
            return {"next": "tools", "tool": tool_to_use, **state}
        else:
            # 不需要工具，直接完成步骤
            state["current_step"] += 1
            state["iteration_count"] += 1
            state["messages"].append({"role": "assistant", "content": f"步骤 {state['current_step']} 完成"})
            return state
    
    def _tools_node(self, state: AgentState) -> Dict[str, Any]:
        """工具节点：调用工具并处理结果"""
        tool_name = state.get("tool", "")
        current_step = state["plan"][state["current_step"]] if state["current_step"] < len(state["plan"]) else ""
        
        # 根据工具名称调用相应的工具
        tool_result = {}
        success = False
        
        try:
            if tool_name == "move":
                # 解析移动参数
                params = self._parse_move_params(current_step)
                tool_result = self.tool_manager.move(**params)
                success = tool_result.get("success", False)
                
            elif tool_name == "yolo_detect":
                # 解析检测参数
                params = self._parse_yolo_params(current_step)
                tool_result = self.tool_manager.yolo_detect(**params)
                success = tool_result.get("success", False)
                
            elif tool_name == "VLM_detect":
                tool_result = self.tool_manager.vlm_detect()
                success = tool_result.get("success", False)
                
            elif tool_name == "track":
                # 解析追踪参数
                params = self._parse_track_params(current_step)
                tool_result = self.tool_manager.track(**params)
                success = tool_result.get("success", False)
                
            elif tool_name == "check_obstacle":
                tool_result = self.tool_manager.check_obstacle()
                success = tool_result.get("success", False)
                
            elif tool_name == "ocr":
                tool_result = self.tool_manager.ocr()
                success = tool_result.get("success", False)
                
            else:
                tool_result = {"success": False, "message": f"未知工具: {tool_name}"}
                
        except Exception as e:
            tool_result = {"success": False, "message": f"工具调用异常: {str(e)}"}
        
        # 更新状态
        state = update_state_with_tool_result(
            state, tool_name, {}, tool_result, success
        )
        
        # 如果工具调用成功，进入下一步
        if success:
            state["current_step"] += 1
            state["iteration_count"] += 1
            state["messages"].append({
                "role": "assistant", 
                "content": f"工具 {tool_name} 调用成功: {tool_result.get('message', '')}"
            })
        else:
            state["messages"].append({
                "role": "assistant", 
                "content": f"工具 {tool_name} 调用失败: {tool_result.get('message', '')}"
            })
        
        return state
    
    def _reflect_node(self, state: AgentState) -> Dict[str, Any]:
        """反思节点：反思执行过程并调整计划"""
        if not self.config.use_reflection:
            return state
        
        # 获取反思提示词
        reflection_prompt = get_reflection_prompt(state)
        
        # 调用LLM进行反思
        messages = [
            SystemMessage(content="你是一个电网巡检智能体，请反思当前的执行过程。"),
            HumanMessage(content=reflection_prompt)
        ]
        
        response = self.llm.invoke(messages)
        reflection = response.content
        
        # 更新状态
        state["reflection"] = reflection
        state["messages"].append({"role": "assistant", "content": f"反思结果：\n{reflection}"})
        
        # 根据反思结果调整计划
        adjusted_plan = self._adjust_plan_based_on_reflection(state["plan"], reflection)
        if adjusted_plan:
            state["plan"] = adjusted_plan
            state["messages"].append({"role": "assistant", "content": "根据反思调整了执行计划"})
        
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """判断是否继续执行"""
        # 检查是否完成所有步骤
        if state["current_step"] >= len(state["plan"]):
            return "complete"
        
        # 检查是否达到最大迭代次数
        if state["iteration_count"] >= self.config.max_iterations:
            return "complete"
        
        # 检查是否需要反思
        if self.config.use_reflection and state["iteration_count"] % 3 == 0:
            return "reflect"
        
        return "continue"
    
    def _parse_plan(self, plan_text: str) -> List[str]:
        """解析LLM生成的计划文本为步骤列表"""
        # 简单的解析逻辑
        steps = []
        lines = plan_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and any(marker in line for marker in ['.', '、', '，', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']):
                # 移除编号和标记
                for marker in ['.', '、', '，', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '*']:
                    if line.startswith(marker):
                        line = line[1:].strip()
                
                if line:
                    steps.append(line)
        
        # 如果没有解析出步骤，使用默认步骤
        if not steps:
            steps = [
                "使用YOLO检测当前场景中的物体",
                "使用VLM详细分析场景",
                "检查障碍物，确定安全移动方向",
                "移动到目标位置",
                "使用OCR读取设备标签",
                "完成巡检任务"
            ]
        
        return steps
    
    def _determine_tool(self, step: str) -> Optional[str]:
        """根据步骤描述确定需要使用的工具"""
        step_lower = step.lower()
        
        if any(word in step_lower for word in ["移动", "前进", "后退", "转向", "move", "go", "forward", "backward"]):
            return "move"
        elif any(word in step_lower for word in ["yolo", "检测", "识别", "物体", "detect", "object"]):
            return "yolo_detect"
        elif any(word in step_lower for word in ["vlm", "视觉大模型", "场景分析", "描述", "scene", "analyze"]):
            return "VLM_detect"
        elif any(word in step_lower for word in ["追踪", "跟踪", "follow", "track"]):
            return "track"
        elif any(word in step_lower for word in ["障碍物", "避障", "安全方向", "obstacle", "avoid"]):
            return "check_obstacle"
        elif any(word in step_lower for word in ["ocr", "文字", "文本", "标签", "text", "read"]):
            return "ocr"
        
        return None
    
    def _parse_move_params(self, step: str) -> Dict[str, Any]:
        """解析移动参数"""
        # 简单的参数解析
        params = {"velocity": None, "angle": 0.0, "seconds": None}
        
        step_lower = step.lower()
        if "快速" in step_lower or "fast" in step_lower:
            params["velocity"] = 0.4
        elif "慢速" in step_lower or "slow" in step_lower:
            params["velocity"] = 0.1
        else:
            params["velocity"] = 0.2
        
        if "左转" in step_lower or "left" in step_lower:
            params["angle"] = 30.0
        elif "右转" in step_lower or "right" in step_lower:
            params["angle"] = -30.0
        
        if "短距离" in step_lower or "short" in step_lower:
            params["seconds"] = 1.0
        elif "长距离" in step_lower or "long" in step_lower:
            params["seconds"] = 5.0
        else:
            params["seconds"] = 2.0
        
        return params
    
    def _parse_yolo_params(self, step: str) -> Dict[str, Any]:
        """解析YOLO检测参数"""
        params = {"threshold": None}
        
        step_lower = step.lower()
        if "高置信度" in step_lower or "high confidence" in step_lower:
            params["threshold"] = 0.9
        elif "低置信度" in step_lower or "low confidence" in step_lower:
            params["threshold"] = 0.5
        
        return params
    
    def _parse_track_params(self, step: str) -> Dict[str, Any]:
        """解析追踪参数"""
        params = {"target": None}
        
        step_lower = step.lower()
        if "人" in step_lower or "person" in step_lower:
            params["target"] = "person"
        elif "电箱" in step_lower or "electric" in step_lower:
            params["target"] = "electric_box"
        elif "变压器" in step_lower or "transformer" in step_lower:
            params["target"] = "transformer"
        
        return params
    
    def _adjust_plan_based_on_reflection(self, plan: List[str], reflection: str) -> Optional[List[str]]:
        """根据反思结果调整计划"""
        # 简单的调整逻辑
        reflection_lower = reflection.lower()
        
        if "障碍物" in reflection_lower or "obstacle" in reflection_lower:
            # 在移动步骤前插入障碍物检查
            new_plan = []
            for step in plan:
                new_plan.append(step)
                if any(word in step.lower() for word in ["移动", "move", "前进", "forward"]):
                    new_plan.append("检查障碍物，确定安全移动方向")
            return new_plan
        
        return None
    
    def run(self, task: str, max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """运行智能体工作流"""
        # 创建初始状态
        initial_state = create_initial_state(task)
        
        # 设置最大迭代次数
        if max_iterations:
            self.config.max_iterations = max_iterations
        
        # 运行工作流
        final_state = initial_state
        try:
            # 由于LangGraph的工作流执行方式，这里简化处理
            # 在实际使用中，应该使用self.app.invoke()
            
            # 简化版本：手动执行工作流节点
            current_state = initial_state.copy()
            
            # 执行规划节点
            current_state = self._plan_node(current_state)
            
            # 循环执行直到完成
            while current_state["current_step"] < len(current_state["plan"]) and \
                  current_state["iteration_count"] < self.config.max_iterations:
                
                # 执行当前步骤
                result = self._execute_node(current_state)
                
                if "next" in result and result["next"] == "tools":
                    # 需要调用工具
                    tool_state = self._tools_node(result)
                    current_state.update(tool_state)
                else:
                    current_state.update(result)
                
                # 检查是否需要反思
                if self.config.use_reflection and current_state["iteration_count"] % 3 == 0:
                    reflection_state = self._reflect_node(current_state)
                    current_state.update(reflection_state)
                
                # 更新迭代计数
                current_state["iteration_count"] += 1
            
            final_state = current_state
            
        except Exception as e:
            print(f"智能体工作流执行异常: {e}")
            final_state = initial_state
            final_state["messages"].append({
                "role": "assistant", 
                "content": f"工作流执行异常: {str(e)}"
            })
        
        # 返回最终状态
        return {
            "success": final_state["current_step"] >= len(final_state["plan"]),
            "state": final_state,
            "summary": get_state_summary(final_state),
            "task": task,
            "iterations": final_state["iteration_count"]
        }
    
    def get_available_tools(self) -> List[str]:
        """获取可用工具列表"""
        return list(self.tool_functions.keys())
    
    def get_tool_function(self, tool_name: str):
        """获取工具函数"""
        return self.tool_functions.get(tool_name)
    
    def get_state_summary(self, state: AgentState) -> str:
        """获取状态摘要"""
        return get_state_summary(state)
    
    def reset(self):
        """重置智能体状态"""
        self.config = load_tool_config(None) if hasattr(self, 'config_path') else self.config
        self.tool_manager = ToolManager(self.config, self.tool_manager.node if hasattr(self.tool_manager, 'node') else None)
        self.tool_functions = create_tool_functions(self.tool_manager)
        self.llm = self._init_llm()
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()


def create_agent(config_path: Optional[str] = None, ros_node=None) -> AgentGraph:
    """创建智能体实例"""
    return AgentGraph(config_path, ros_node)


def run_agent_task(task: str, config_path: Optional[str] = None, max_iterations: int = 20) -> Dict[str, Any]:
    """运行智能体任务（简化接口）"""
    agent = create_agent(config_path)
    return agent.run(task, max_iterations)


# 测试函数
def test_agent():
    """测试智能体"""
    print("=== 测试电网巡检智能体 ===")
    
    # 创建智能体
    agent = create_agent()
    
    # 测试任务
    test_tasks = [
        "检查变电站A区的设备状态",
        "追踪维护人员并保持安全距离",
        "读取电力控制箱上的警告标志",
        "巡检整个变电站区域"
    ]
    
    for i, task in enumerate(test_tasks):
        print(f"\n--- 测试任务 {i+1}: {task} ---")
        
        # 运行智能体
        result = agent.run(task, max_iterations=10)
        
        # 打印结果
        print(f"任务: {result['task']}")
        print(f"成功: {result['success']}")
        print(f"迭代次数: {result['iterations']}")
        print(f"状态摘要:\n{result['summary']}")
        
        # 打印工具调用历史
        if result['state']['tool_calls']:
            print(f"工具调用记录:")
            for tool_call in result['state']['tool_calls']:
                print(f"  - {tool_call['tool_type']}: {tool_call.get('success', False)}")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_agent()