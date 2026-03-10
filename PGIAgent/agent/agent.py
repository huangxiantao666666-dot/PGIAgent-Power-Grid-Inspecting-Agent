"""
Plan-Act-Reflect Agent 工作流定义
使用LangGraph构建电网巡检智能体的Plan-Act-Reflect工作流

特性：
- 异步执行：支持 async/await
- 流式输出：支持 graph.stream(stream_mode="message")
- 兼容模式：保留同步 invoke 接口

工作流结构：
think -> plan -> act -> (reflect | act) -> examine -> (reflect | end)

其中 Act 节点使用 ReactAgent 子图来实现 ReAct 循环
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated, Literal, Union, Iterator, AsyncIterator, Callable
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, AnyMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
import json
import time
import os
import re
import asyncio
from dotenv import load_dotenv

from .state import (
    PlanActReflectState, AgentConfig, 
    create_initial_state, load_agent_config, get_state_summary
)
from .tools import ToolManager, create_tool_functions, create_async_tool_functions
from .prompts import (
    get_system_prompt, 
    get_think_prompt, get_plan_prompt,
    get_reflect_prompt, get_examine_prompt, get_summary_prompt
)
from .react_agent import ReactAgent, create_react_agent

load_dotenv()


# ========== 辅助函数 ==========

def parse_plan_from_text(plan_text: str) -> List[str]:
    """从LLM输出的计划文本中解析步骤列表"""
    steps = []
    plan_text = plan_text.strip()
    
    # 移除markdown代码块
    if plan_text.startswith('```'):
        lines = plan_text.split('\n')
        plan_text = '\n'.join(lines[1:] if lines else [])
        if plan_text.endswith('```'):
            plan_text = plan_text[:-3]
    
    lines = plan_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        step_text = line
        # 移除编号前缀
        patterns = [
            r'^\d+[\.\)]\s*',
            r'^[①②③④⑤⑥⑦⑧⑨⑩]\s*',
            r'^[-*]\s*',
            r'^\[\s*\d+\s*\]\s*',
        ]
        
        for pattern in patterns:
            step_text = re.sub(pattern, '', step_text)
        
        step_text = step_text.strip()
        
        if step_text and len(step_text) > 2:
            steps.append(step_text)
    
    if not steps:
        steps = ["分析任务需求", "制定执行计划", "开始执行", "检查结果", "完成任务"]
    
    return steps


def format_past_steps(past_steps: List[Dict]) -> str:
    """格式化已完成步骤"""
    if not past_steps:
        return "无"
    
    formatted = []
    for i, step_info in enumerate(past_steps):
        formatted.append(f"{i+1}. {step_info.get('step', '')} - {step_info.get('status', '')}")
    
    return "\n".join(formatted)


# ========== Agent类定义 ==========

class PlanActReflectAgent:
    """Plan-Act-Reflect Agent工作流管理器"""
    
    def __init__(self, config: Optional[AgentConfig] = None, ros_node=None):
        # 加载配置
        self.config = config or load_agent_config()
        
        # 初始化工具管理器
        self.tool_manager = ToolManager(self.config, ros_node)
        
        # 创建 ReactAgent 子图（用于 Act 节点）
        self.react_agent = create_react_agent(self.config, self.tool_manager)
        
        # 初始化LLM（用于 Think, Plan, Reflect, Examine, End 节点）
        self.llm = self._init_llm()
        
        # 初始化异步LLM
        self.async_llm = self._init_async_llm()
        
        # 构建工作流（同步和异步）
        self.workflow = self._build_workflow()
        self.async_workflow = self._build_async_workflow()
        
        # 编译工作流
        self.app = self.workflow.compile()
        self.async_app = self.async_workflow.compile()
        
        print(f"✅ Plan-Act-Reflect Agent 初始化完成 (provider: {self.config.llm_provider})")
    
    def _init_llm(self):
        """初始化同步LLM"""
        if self.config.llm_provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
            return ChatOpenAI(
                model=self.config.llm_model,
                api_key=api_key,
                base_url="https://api.deepseek.com",
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens
            )
        elif self.config.llm_provider == "qwen":
            api_key = os.getenv("QWEN_API_KEY", "")
            try:
                from langchain_community.chat_models.tongyi import ChatTongyi
                return ChatTongyi(
                    model=self.config.llm_model or "qwen-plus",
                    api_key=api_key,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens
                )
            except ImportError:
                return ChatOpenAI(
                    model="qwen-turbo",
                    api_key=api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens
                )
        elif self.config.llm_provider == "local":
            return ChatOllama(
                model=self.config.llm_model or "qwen2.5:7b",
                temperature=0.8,
                num_ctx=4096
            )
        else:
            return self._create_mock_llm()
    
    def _init_async_llm(self):
        """初始化异步LLM"""
        return self._create_llm()
    
    def _create_llm(self):
        """创建LLM实例（同步/异步共用）"""
        if self.config.llm_provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
            return ChatOpenAI(
                model=self.config.llm_model,
                api_key=api_key,
                base_url="https://api.deepseek.com",
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens
            )
        elif self.config.llm_provider == "qwen":
            api_key = os.getenv("QWEN_API_KEY", "")
            try:
                from langchain_community.chat_models.tongyi import ChatTongyi
                return ChatTongyi(
                    model=self.config.llm_model or "qwen-plus",
                    api_key=api_key,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens
                )
            except ImportError:
                return ChatOpenAI(
                    model="qwen-turbo",
                    api_key=api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens
                )
        elif self.config.llm_provider == "local":
            return ChatOllama(
                model=self.config.llm_model or "qwen2.5:7b",
                temperature=0.8,
                num_ctx=4096
            )
        else:
            return self._create_mock_llm()
    
    def _create_mock_llm(self):
        """创建模拟LLM用于测试"""
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatResult, ChatGeneration
        
        class MockLLM(BaseChatModel):
            def _generate(self, messages, stop=None, **kwargs):
                last_message = messages[-1].content if messages else ""
                
                if "think" in last_message.lower():
                    response = "思考：这是一个电网巡检任务。我需要：1. 先了解当前环境；2. 制定巡检计划；3. 执行巡检任务；4. 检查完成情况。"
                elif "plan" in last_message.lower() and "制定" in last_message.lower():
                    response = "1. 使用YOLO检测当前场景中的物体\n2. 使用VLM详细分析场景\n3. 检查障碍物，确定安全移动方向\n4. 移动到目标设备位置\n5. 使用OCR读取设备标签\n6. 完成任务并总结"
                elif "reflect" in last_message.lower():
                    response = "反思：上一步执行成功。继续执行下一个步骤。"
                elif "examine" in last_message.lower():
                    response = "检查结果：任务未完全完成，建议继续执行剩余步骤。"
                else:
                    response = "我理解了任务，将开始执行。"
                
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])
            
            async def _agenerate(self, messages, stop=None, **kwargs):
                return self._generate(messages, stop, **kwargs)
            
            @property
            def _llm_type(self):
                return "mock"
        
        return MockLLM()
    
    def _build_workflow(self) -> StateGraph:
        """构建同步工作流"""
        return self._build_workflow_base(is_async=False)
    
    def _build_async_workflow(self) -> StateGraph:
        """构建异步工作流"""
        return self._build_workflow_base(is_async=True)
    
    def _build_workflow_base(self, is_async: bool = False) -> StateGraph:
        """构建工作流（同步/异步通用）"""
        workflow = StateGraph(PlanActReflectState)
        
        if is_async:
            # 异步节点
            workflow.add_node("think", self._think_node_async)
            workflow.add_node("plan", self._plan_node_async)
            workflow.add_node("act", self._act_node_async)
            workflow.add_node("reflect", self._reflect_node_async)
            workflow.add_node("examine", self._examine_node_async)
            workflow.add_node("end", self._end_node_async)
        else:
            # 同步节点
            workflow.add_node("think", self._think_node)
            workflow.add_node("plan", self._plan_node)
            workflow.add_node("act", self._act_node)
            workflow.add_node("reflect", self._reflect_node)
            workflow.add_node("examine", self._examine_node)
            workflow.add_node("end", self._end_node)
        
        workflow.set_entry_point("think")
        workflow.add_edge("think", "plan")
        workflow.add_edge("plan", "act")
        
        # Conditional edges
        workflow.add_conditional_edges("act", self._should_continue_after_act, {
            "act": "act",
            "reflect": "reflect",
            "examine": "examine"
        })
        
        workflow.add_conditional_edges("reflect", self._should_replan_after_reflect, {
            "act": "act",
            "examine": "examine"
        })
        
        workflow.add_conditional_edges("examine", self._should_continue_after_examine, {
            "reflect": "reflect",
            "end": END
        })
        
        workflow.add_edge("end", END)
        
        return workflow
    
    # ========== 同步节点实现 ==========
    
    def _think_node(self, state: PlanActReflectState) -> Dict[str, Any]:
        """Think节点：思考如何做用户任务"""
        task = state["task"]
        think_prompt = get_think_prompt(task)
        
        messages = [
            SystemMessage(content=get_system_prompt()),
            HumanMessage(content=think_prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        new_messages = list(state.get("messages", []))
        new_messages.append(HumanMessage(content=think_prompt))
        new_messages.append(AIMessage(content=response.content))
        
        return {
            "messages": new_messages,
            "iteration_count": state.get("iteration_count", 0)
        }
    
    def _plan_node(self, state: PlanActReflectState) -> Dict[str, Any]:
        """Plan节点：制定执行计划"""
        task = state["task"]
        
        # 获取思考结果
        think_context = ""
        messages = state.get("messages", [])
        if len(messages) > 1:
            think_context = messages[-1].content
        
        plan_prompt = get_plan_prompt(task, think_context)
        
        messages = [
            SystemMessage(content=get_system_prompt()),
            HumanMessage(content=plan_prompt)
        ]
        
        response = self.llm.invoke(messages)
        plan_text = response.content
        plan_steps = parse_plan_from_text(plan_text)
        
        new_messages = list(state.get("messages", []))
        new_messages.append(HumanMessage(content=plan_prompt))
        new_messages.append(AIMessage(content=plan_text))
        
        return {
            "plan": plan_steps,
            "current_step_index": 0,
            "messages": new_messages,
            "step_messages": [],
            "past_steps": []
        }
    
    def _act_node(self, state: PlanActReflectState) -> Dict[str, Any]:
        """Act节点：调用 ReactAgent 子图执行当前步骤"""
        plan = state.get("plan", [])
        current_index = state.get("current_step_index", 0)
        
        if not plan or current_index >= len(plan):
            return {
                "current_step_status": "completed",
                "task_completed": True
            }
        
        current_step = plan[current_index]
        
        # 调用 ReactAgent 子图来执行这个步骤
        react_result = self.react_agent.run(
            task=state["task"],
            current_step=current_step,
            step_index=current_index,
            total_steps=len(plan),
            max_rounds=3,
            past_steps=format_past_steps(state.get('past_steps', []))
        )
        
        # 提取 ReAct 执行的消息
        react_messages = react_result.get("messages", [])
        
        # 更新已完成步骤
        new_past_steps = state.get("past_steps", []) + [{
            "step": current_step,
            "result": "步骤执行完成",
            "status": "success" if react_result.get("success") else "failed"
        }]
        
        return {
            "step_messages": react_messages,
            "current_step_status": "success",
            "past_steps": new_past_steps,
            "current_step_index": current_index + 1,
            "iteration_count": state.get("iteration_count", 0) + react_result.get("iteration_count", 0)
        }
    
    def _reflect_node(self, state: PlanActReflectState) -> Dict[str, Any]:
        """Reflect节点：反思执行过程"""
        current_step = state.get("plan", [])[state.get("current_step_index", 0) - 1] if state.get("current_step_index", 0) > 0 else "无"
        step_status = state.get("current_step_status", "pending")
        
        reflect_prompt = get_reflect_prompt(
            task=state['task'],
            current_step=current_step,
            step_status=step_status,
            past_steps=format_past_steps(state.get('past_steps', []))
        )
        
        messages = [
            SystemMessage(content=get_system_prompt()),
            HumanMessage(content=reflect_prompt)
        ]
        
        for msg in state.get("messages", [])[-4:]:
            messages.append(msg)
        
        response = self.llm.invoke(messages)
        reflection = response.content
        reflect_type = "modify_current"
        
        if "重新规划" in reflection or "重新制定" in reflection:
            reflect_type = "replan"
        
        new_messages = list(state.get("messages", []))
        new_messages.append(HumanMessage(content=reflect_prompt))
        new_messages.append(AIMessage(content=reflection))
        
        return {
            "reflection": reflection,
            "reflect_type": reflect_type,
            "messages": new_messages,
            "step_messages": [],
            "current_step_status": "pending"
        }
    
    def _examine_node(self, state: PlanActReflectState) -> Dict[str, Any]:
        """Examine节点：检查任务是否完成"""
        task = state["task"]
        plan = state.get("plan", [])
        current_index = state.get("current_step_index", 0)
        past_steps = state.get("past_steps", [])
        
        examine_prompt = get_examine_prompt(
            task=task,
            plan=plan,
            current_index=current_index,
            past_steps=format_past_steps(past_steps)
        )
        
        messages = [
            SystemMessage(content=get_system_prompt()),
            HumanMessage(content=examine_prompt)
        ]
        
        for msg in state.get("messages", [])[-4:]:
            messages.append(msg)
        
        response = self.llm.invoke(messages)
        examine_result = response.content
        task_completed = "完成" in examine_result and "未完成" not in examine_result
        
        new_messages = list(state.get("messages", []))
        new_messages.append(HumanMessage(content=examine_prompt))
        new_messages.append(AIMessage(content=examine_result))
        
        return {
            "examine_result": examine_result,
            "task_completed": task_completed,
            "messages": new_messages
        }
    
    def _end_node(self, state: PlanActReflectState) -> Dict[str, Any]:
        """End节点：生成最终答案"""
        task = state["task"]
        past_steps = state.get("past_steps", [])
        examine_result = state.get("examine_result", "")
        
        summary_prompt = get_summary_prompt(
            task=task,
            past_steps=format_past_steps(past_steps),
            examine_result=examine_result
        )
        
        messages = [
            SystemMessage(content=get_system_prompt()),
            HumanMessage(content=summary_prompt)
        ]
        
        for msg in state.get("messages", []):
            messages.append(msg)
        
        response = self.llm.invoke(messages)
        final_answer = response.content
        
        return {
            "final_answer": final_answer,
            "messages": state.get("messages", []) + [AIMessage(content=final_answer)]
        }
    
    # ========== 异步节点实现 ==========
    
    async def _think_node_async(self, state: PlanActReflectState) -> Dict[str, Any]:
        """Think节点（异步）：思考如何做用户任务"""
        task = state["task"]
        think_prompt = get_think_prompt(task)
        
        messages = [
            SystemMessage(content=get_system_prompt()),
            HumanMessage(content=think_prompt)
        ]
        
        response = await self.async_llm.ainvoke(messages)
        
        new_messages = list(state.get("messages", []))
        new_messages.append(HumanMessage(content=think_prompt))
        new_messages.append(AIMessage(content=response.content))
        
        return {
            "messages": new_messages,
            "iteration_count": state.get("iteration_count", 0)
        }
    
    async def _plan_node_async(self, state: PlanActReflectState) -> Dict[str, Any]:
        """Plan节点（异步）：制定执行计划"""
        task = state["task"]
        
        # 获取思考结果
        think_context = ""
        messages = state.get("messages", [])
        if len(messages) > 1:
            think_context = messages[-1].content
        
        plan_prompt = get_plan_prompt(task, think_context)
        
        messages = [
            SystemMessage(content=get_system_prompt()),
            HumanMessage(content=plan_prompt)
        ]
        
        response = await self.async_llm.ainvoke(messages)
        plan_text = response.content
        plan_steps = parse_plan_from_text(plan_text)
        
        new_messages = list(state.get("messages", []))
        new_messages.append(HumanMessage(content=plan_prompt))
        new_messages.append(AIMessage(content=plan_text))
        
        return {
            "plan": plan_steps,
            "current_step_index": 0,
            "messages": new_messages,
            "step_messages": [],
            "past_steps": []
        }
    
    async def _act_node_async(self, state: PlanActReflectState) -> Dict[str, Any]:
        """Act节点（异步）：调用 ReactAgent 子图执行当前步骤"""
        plan = state.get("plan", [])
        current_index = state.get("current_step_index", 0)
        
        if not plan or current_index >= len(plan):
            return {
                "current_step_status": "completed",
                "task_completed": True
            }
        
        current_step = plan[current_index]
        
        # 调用 ReactAgent 子图（异步版本）
        react_result = await self.react_agent.run_async(
            task=state["task"],
            current_step=current_step,
            step_index=current_index,
            total_steps=len(plan),
            max_rounds=3,
            past_steps=format_past_steps(state.get('past_steps', []))
        )
        
        # 提取 ReAct 执行的消息
        react_messages = react_result.get("messages", [])
        
        # 更新已完成步骤
        new_past_steps = state.get("past_steps", []) + [{
            "step": current_step,
            "result": "步骤执行完成",
            "status": "success" if react_result.get("success") else "failed"
        }]
        
        return {
            "step_messages": react_messages,
            "current_step_status": "success",
            "past_steps": new_past_steps,
            "current_step_index": current_index + 1,
            "iteration_count": state.get("iteration_count", 0) + react_result.get("iteration_count", 0)
        }
    
    async def _reflect_node_async(self, state: PlanActReflectState) -> Dict[str, Any]:
        """Reflect节点（异步）：反思执行过程"""
        current_step = state.get("plan", [])[state.get("current_step_index", 0) - 1] if state.get("current_step_index", 0) > 0 else "无"
        step_status = state.get("current_step_status", "pending")
        
        reflect_prompt = get_reflect_prompt(
            task=state['task'],
            current_step=current_step,
            step_status=step_status,
            past_steps=format_past_steps(state.get('past_steps', []))
        )
        
        messages = [
            SystemMessage(content=get_system_prompt()),
            HumanMessage(content=reflect_prompt)
        ]
        
        for msg in state.get("messages", [])[-4:]:
            messages.append(msg)
        
        response = await self.async_llm.ainvoke(messages)
        reflection = response.content
        reflect_type = "modify_current"
        
        if "重新规划" in reflection or "重新制定" in reflection:
            reflect_type = "replan"
        
        new_messages = list(state.get("messages", []))
        new_messages.append(HumanMessage(content=reflect_prompt))
        new_messages.append(AIMessage(content=reflection))
        
        return {
            "reflection": reflection,
            "reflect_type": reflect_type,
            "messages": new_messages,
            "step_messages": [],
            "current_step_status": "pending"
        }
    
    async def _examine_node_async(self, state: PlanActReflectState) -> Dict[str, Any]:
        """Examine节点（异步）：检查任务是否完成"""
        task = state["task"]
        plan = state.get("plan", [])
        current_index = state.get("current_step_index", 0)
        past_steps = state.get("past_steps", [])
        
        examine_prompt = get_examine_prompt(
            task=task,
            plan=plan,
            current_index=current_index,
            past_steps=format_past_steps(past_steps)
        )
        
        messages = [
            SystemMessage(content=get_system_prompt()),
            HumanMessage(content=examine_prompt)
        ]
        
        for msg in state.get("messages", [])[-4:]:
            messages.append(msg)
        
        response = await self.async_llm.ainvoke(messages)
        examine_result = response.content
        task_completed = "完成" in examine_result and "未完成" not in examine_result
        
        new_messages = list(state.get("messages", []))
        new_messages.append(HumanMessage(content=examine_prompt))
        new_messages.append(AIMessage(content=examine_result))
        
        return {
            "examine_result": examine_result,
            "task_completed": task_completed,
            "messages": new_messages
        }
    
    async def _end_node_async(self, state: PlanActReflectState) -> Dict[str, Any]:
        """End节点（异步）：生成最终答案"""
        task = state["task"]
        past_steps = state.get("past_steps", [])
        examine_result = state.get("examine_result", "")
        
        summary_prompt = get_summary_prompt(
            task=task,
            past_steps=format_past_steps(past_steps),
            examine_result=examine_result
        )
        
        messages = [
            SystemMessage(content=get_system_prompt()),
            HumanMessage(content=summary_prompt)
        ]
        
        for msg in state.get("messages", []):
            messages.append(msg)
        
        response = await self.async_llm.ainvoke(messages)
        final_answer = response.content
        
        return {
            "final_answer": final_answer,
            "messages": state.get("messages", []) + [AIMessage(content=final_answer)]
        }
    
    # ========== 条件边函数 ==========
    
    def _should_continue_after_act(self, state: PlanActReflectState) -> str:
        """判断Act节点之后的走向"""
        current_index = state.get("current_step_index", 0)
        plan = state.get("plan", [])
        step_status = state.get("current_step_status", "pending")
        
        if current_index >= len(plan):
            return "examine"
        
        if step_status == "failure":
            return "reflect"
        
        if state.get("iteration_count", 0) >= self.config.max_iterations:
            return "examine"
        
        return "act"
    
    def _should_replan_after_reflect(self, state: PlanActReflectState) -> str:
        """判断Reflect节点之后的走向"""
        reflect_type = state.get("reflect_type", "modify_current")
        
        if reflect_type == "replan":
            return "act"
        
        return "act"
    
    def _should_continue_after_examine(self, state: PlanActReflectState) -> str:
        """判断Examine节点之后的走向"""
        if state.get("task_completed", False):
            return "end"
        
        if self.config.use_reflection and state.get("iteration_count", 0) < self.config.max_iterations:
            return "reflect"
        
        return "end"
    
    # ========== 同步执行接口 ==========
    
    def run(self, task: str, max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """运行Agent工作流（同步接口）"""
        if max_iterations:
            self.config.max_iterations = max_iterations
        
        initial_state = create_initial_state(task, self.config.max_iterations)
        
        try:
            final_state = self.app.invoke(initial_state)
            
            return {
                "success": final_state.get("task_completed", False),
                "task": task,
                "final_answer": final_state.get("final_answer", "任务完成"),
                "past_steps": final_state.get("past_steps", []),
                "iteration_count": final_state.get("iteration_count", 0),
                "messages": final_state.get("messages", [])
            }
            
        except Exception as e:
            print(f"Agent执行异常: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "task": task,
                "final_answer": f"任务执行失败: {str(e)}",
                "past_steps": [],
                "iteration_count": 0,
                "messages": []
            }
    
    # ========== 流式输出接口（同步） ==========
    
    def stream(self, task: str, max_iterations: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """
        运行Agent工作流并流式输出（同步版本）
        
        使用 stream_mode="message" 模式输出每条消息
        """
        if max_iterations:
            self.config.max_iterations = max_iterations
        
        initial_state = create_initial_state(task, self.config.max_iterations)
        
        try:
            # 使用 stream 模式输出
            for event in self.app.stream(initial_state, stream_mode="message"):
                # event 是 tuple (node_name, messages)
                if isinstance(event, tuple):
                    node_name, messages = event
                    # 获取最后一条消息
                    if messages and len(messages) > 0:
                        last_msg = messages[-1]
                        yield {
                            "node": node_name,
                            "type": last_msg.type if hasattr(last_msg, 'type') else "unknown",
                            "content": last_msg.content if hasattr(last_msg, 'content') else str(last_msg),
                        }
                else:
                    # 兼容其他格式
                    yield {"node": "unknown", "type": "event", "content": str(event)}
                    
        except Exception as e:
            yield {"node": "error", "type": "error", "content": f"执行异常: {str(e)}"}
    
    # ========== 异步执行接口 ==========
    
    async def run_async(self, task: str, max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """运行Agent工作流（异步接口）"""
        if max_iterations:
            self.config.max_iterations = max_iterations
        
        initial_state = create_initial_state(task, self.config.max_iterations)
        
        try:
            # 异步调用
            final_state = await self.async_app.ainvoke(initial_state)
            
            return {
                "success": final_state.get("task_completed", False),
                "task": task,
                "final_answer": final_state.get("final_answer", "任务完成"),
                "past_steps": final_state.get("past_steps", []),
                "iteration_count": final_state.get("iteration_count", 0),
                "messages": final_state.get("messages", [])
            }
            
        except Exception as e:
            print(f"Agent执行异常: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "task": task,
                "final_answer": f"任务执行失败: {str(e)}",
                "past_steps": [],
                "iteration_count": 0,
                "messages": []
            }
    
    async def stream_async(self, task: str, max_iterations: Optional[int] = None) -> AsyncIterator[Dict[str, Any]]:
        """
        运行Agent工作流并流式输出（异步版本）
        
        使用 stream_mode="message" 模式输出每条消息
        """
        if max_iterations:
            self.config.max_iterations = max_iterations
        
        initial_state = create_initial_state(task, self.config.max_iterations)
        
        try:
            # 异步流式调用
            async for event in self.async_app.astream(initial_state, stream_mode="message"):
                if isinstance(event, tuple):
                    node_name, messages = event
                    if messages and len(messages) > 0:
                        last_msg = messages[-1]
                        yield {
                            "node": node_name,
                            "type": last_msg.type if hasattr(last_msg, 'type') else "unknown",
                            "content": last_msg.content if hasattr(last_msg, 'content') else str(last_msg),
                        }
                else:
                    yield {"node": "unknown", "type": "event", "content": str(event)}
                    
        except Exception as e:
            yield {"node": "error", "type": "error", "content": f"执行异常: {str(e)}"}


# ========== 便捷函数 ==========

def create_plan_act_reflect_agent(
    config: Optional[AgentConfig] = None, 
    ros_node=None
) -> PlanActReflectAgent:
    """创建Plan-Act-Reflect Agent实例"""
    return PlanActReflectAgent(config, ros_node)


def run_task(task: str, config: Optional[AgentConfig] = None, max_iterations: int = 20) -> Dict[str, Any]:
    """运行任务的便捷接口（同步）"""
    agent = create_plan_act_reflect_agent(config)
    return agent.run(task, max_iterations)


async def run_task_async(task: str, config: Optional[AgentConfig] = None, max_iterations: int = 20) -> Dict[str, Any]:
    """运行任务的便捷接口（异步）"""
    agent = create_plan_act_reflect_agent(config)
    return await agent.run_async(task, max_iterations)


# ========== 测试代码 ==========

def test_agent():
    """测试Plan-Act-Reflect Agent（同步版本）"""
    print("=== 测试 Plan-Act-Reflect Agent (同步) ===\n")
    
    agent = create_plan_act_reflect_agent()
    
    # 测试1: 同步 run
    print("\n--- 测试 run (同步) ---")
    result = agent.run("检查变电站A区的设备状态", max_iterations=5)
    print(f"✅ 成功: {result['success']}")
    print(f"迭代: {result['iteration_count']}")
    print(f"结果: {result['final_answer'][:200]}...")
    
    # 测试2: stream 输出
    print("\n--- 测试 stream (流式输出) ---")
    for event in agent.stream("追踪一名维护人员", max_iterations=3):
        print(f"[{event['node']}] {event['type']}: {event['content'][:100]}...")


async def test_async():
    """测试异步版本"""
    print("\n=== 测试异步版本 ===\n")
    
    agent = create_plan_act_reflect_agent()
    
    # 测试异步 run
    print("\n--- 测试 run_async ---")
    result = await agent.run_async("检查变电站A区的设备状态", max_iterations=5)
    print(f"✅ 成功: {result['success']}")
    print(f"结果: {result['final_answer'][:200]}...")
    
    # 测试异步 stream
    print("\n--- 测试 stream_async ---")
    async for event in agent.stream_async("追踪一名维护人员", max_iterations=3):
        print(f"[{event['node']}] {event['type']}: {event['content'][:100]}...")


if __name__ == "__main__":
    test_agent()
