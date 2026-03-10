"""
ReAct Agent 子图实现
使用 LangGraph 实现 ReAct (Reasoning + Acting) 工作流

ReAct 核心循环：
1. Thought - LLM 生成思考
2. Action - 选择工具并执行
3. Observation - 获取工具执行结果
4. 重复直到任务完成

使用 LangGraph 的 ToolNode 自动处理工具调用
"""

from typing import Dict, Any, List, Optional, TypedDict, Iterator, AsyncIterator
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AnyMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
import os
from dotenv import load_dotenv

from .state import AgentConfig
from .tools import create_tool_functions, create_async_tool_functions
from .prompts import get_act_prompt

load_dotenv()


# ========== ReAct 状态定义 ==========

class ReactAgentState(TypedDict):
    """ReAct Agent 状态"""
    # 任务信息
    task: str                          # 原始任务
    current_step: str                  # 当前要执行的步骤
    step_index: int                    # 步骤索引
    total_steps: int                   # 总步骤数
    
    # 执行历史
    messages: List[AnyMessage]          # 对话消息（包含HumanMessage/AIMessage/ToolMessage）
    past_steps: List[Dict]              # 已完成的步骤
    
    # 执行控制
    max_rounds: int                    # 最大循环轮数
    current_round: int                 # 当前轮数
    iteration_count: int                # 迭代次数
    
    # 结果
    step_status: str                   # 步骤执行状态


# ========== 辅助函数 ==========

def should_continue(state: ReactAgentState) -> str:
    """判断是否继续执行
    
    使用 LangGraph 的条件边函数
    """
    messages = state.get("messages", [])
    
    # 检查最后一条消息是否有工具调用
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"  # 继续执行工具调用
    
    # 检查是否明确完成
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, AIMessage):
            content = last_message.content
            if "Action: 完成" in content or "Action:完成" in content:
                return "end"
    
    # 检查是否达到最大轮数
    if state.get("current_round", 0) >= state.get("max_rounds", 3):
        return "end"
    
    return "end"


# ========== ReAct Agent 类 ==========

class ReactAgent:
    """
    ReAct Agent 子图
    
    使用 LangGraph 的 ToolNode 自动处理工具调用
    """
    
    def __init__(self, config: Optional[AgentConfig] = None, tool_manager=None, use_async=True):
        """初始化 ReAct Agent"""
        self.config = config or AgentConfig()
        self.tool_manager = tool_manager
        
        # 创建工具函数，工具节点，llm以及工作流图
        if use_async:
            self.async_tool_functions = create_async_tool_functions(tool_manager)
            self.async_tool_node = ToolNode(list(self.async_tool_functions.values()))
            self.async_llm = self._init_async_llm()
            self.async_workflow = self._build_async_workflow()
            self.async_app = self.async_workflow.compile()
        else:
            self.tool_functions = create_tool_functions(tool_manager)
            self.tool_node = ToolNode(list(self.tool_functions.values()))
            self.llm = self._init_llm()
            self.workflow = self._build_workflow()
            self.app = self.workflow.compile()
            
    def _init_llm(self) -> ChatOpenAI:
        """初始化 LLM（绑定工具）

        :return: LLM 对象
        :rtype: ChatOpenAI
        """
        if self.config.llm_provider == "deepseek":
            api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
            return ChatOpenAI(
                model: str = self.config.llm_model,
                api_key: str = api_key,
                base_url: str = "https://api.deepseek.com",
                temperature: float = self.config.llm_temperature,
                max_tokens: int = self.config.llm_max_tokens
            ).bind_tools(list(self.tool_functions.values()))
        elif self.config.llm_provider == "qwen":
            api_key: str = os.getenv("QWEN_API_KEY", "")
            try:
                from langchain_community.chat_models.tongyi import ChatTongyi
                return ChatTongyi(
                    model: str = self.config.llm_model or "qwen-plus",
                    api_key: str = api_key,
                    temperature: float = self.config.llm_temperature,
                    max_tokens: int = self.config.llm_max_tokens
                ).bind_tools(list(self.tool_functions.values()))
            except ImportError:
                return ChatOpenAI(
                    model: str = "qwen-turbo",
                    api_key: str = api_key,
                    base_url: str = "https://dashscopealiyuncs.com/compatible-mode/v1",
                    temperature: float = self.config.llm_temperature,
                    max_tokens: int = self.config.llm_max_tokens
                ).bind_tools(list(self.tool_functions.values()))
        elif self.config.llm_provider == "local":
            return ChatOllama(
                model: str = self.config.llm_model or "qwen2.5:7b",
                temperature: float = 0.8,
                num_ctx: int = 4096
            ).bind_tools(list(self.tool_functions.values()))
        else:
            return self._create_mock_llm()
    
    def _init_async_llm(self) -> ChatOpenAI:
        """初始化异步 LLM"""
        if self.config.llm_provider == "deepseek":
            api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
            return ChatOpenAI(
                model: str = self.config.llm_model,
                api_key: str = api_key,
                base_url: str = "https://api.deepseek.com",
                temperature: float = self.config.llm_temperature,
                max_tokens: int = self.config.llm_max_tokens
            ).bind_tools(list(self.async_tool_functions.values()))
        elif self.config.llm_provider == "qwen":
            api_key = os.getenv("QWEN_API_KEY", "")
            try:
                from langchain_community.chat_models.tongyi import ChatTongyi
                return ChatTongyi(
                    model=self.config.llm_model or "qwen-plus",
                    api_key=api_key,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens
                ).bind_tools(list(self.async_tool_functions.values()))
            except ImportError:
                return ChatOpenAI(
                    model="qwen-turbo",
                    api_key=api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens
                ).bind_tools(list(self.async_tool_functions.values()))
        elif self.config.llm_provider == "local":
            return ChatOllama(
                model=self.config.llm_model or "qwen2.5:7b",
                temperature=0.8,
                num_ctx=4096
            ).bind_tools(list(self.async_tool_functions.values()))
        else:
            return self._create_mock_llm()
    
    def _create_mock_llm(self):
        """创建模拟 LLM"""
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatResult, ChatGeneration
        
        class MockLLM(BaseChatModel):
            def _generate(self, messages, stop=None, **kwargs):
                # 模拟返回工具调用
                response = "Action: yolo_detect\nAction Input: {\"threshold\": 0.8}"
                return ChatResult(generations=[ChatGeneration(message=AIMessage(
                    content=response,
                    tool_calls=[{
                        "name": "yolo_detect",
                        "args": {"threshold": 0.8},
                        "id": "mock_call_1"
                    }]
                ))])
            
            async def _agenerate(self, messages, stop=None, **kwargs):
                return self._generate(messages, stop, **kwargs)
            
            @property
            def _llm_type(self):
                return "mock"
        
        return MockLLM().bind_tools(list(self.tool_functions.values()))
    
    def _build_workflow(self) -> StateGraph:
        """构建同步工作流
        
        流程:
        agent → (检查是否有工具调用) → [tool_node] → agent
              → (无工具调用) → end
        """
        workflow = StateGraph(ReactAgentState)
        
        # 添加节点
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self.tool_node)
        
        # 设置入口
        workflow.set_entry_point("agent")
        
        # 条件边：根据是否有工具调用决定走向
        workflow.add_conditional_edges(
            "agent",
            self._should_call_tools,
            {
                "tools": "tools",
                "end": END
            }
        )
        
        # 工具执行完后返回 agent 节点
        workflow.add_edge("tools", "agent")
        
        return workflow
    
    def _build_async_workflow(self) -> StateGraph:
        """构建异步工作流"""
        workflow = StateGraph(ReactAgentState)
        
        workflow.add_node("agent", self._agent_node_async)
        workflow.add_node("tools", self.async_tool_node)
        
        workflow.set_entry_point("agent")
        
        workflow.add_conditional_edges(
            "agent",
            self._should_call_tools,
            {
                "tools": "tools",
                "end": END
            }
        )
        
        workflow.add_edge("tools", "agent")
        
        return workflow
    
    def _should_call_tools(self, state: ReactAgentState) -> str:
        """判断是否需要调用工具
        
        检查 LLM 响应是否有 tool_calls
        - 如果最后一条是 HumanMessage（初始提示刚添加），需要调用 LLM
        - 如果最后一条是 AIMessage 且有 tool_calls，调用工具
        - 否则结束
        """
        messages = state.get("messages", [])
        
        if not messages:
            return "end"
        
        last_message = messages[-1]
        
        # 如果是 HumanMessage刚添加（第一次调用），需要调用 LLM
        if isinstance(last_message, HumanMessage):
            return "tools"
        
        # 如果是 AIMessage 且有 tool_calls，调用工具
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        
        return "end"
    
    def _agent_node(self, state: ReactAgentState) -> Dict[str, Any]:
        """Agent 节点：调用 LLM 生成响应"""
        messages = list(state.get("messages", []))
        current_round = state.get("current_round", 0)
        
        # 如果是第一次，添加初始提示
        if current_round == 0:
            initial_prompt = get_act_prompt(
                task=state["task"],
                current_step=state["current_step"],
                step_index=state["step_index"],
                total_steps=state["total_steps"],
                past_steps="",
                execution_history=""
            )
            messages.append(HumanMessage(content=initial_prompt))
        
        # 调用 LLM
        response = self.llm.invoke(messages)
        
        # 增加轮数
        new_round = current_round + 1
        
        return {
            "messages": messages + [response],
            "current_round": new_round,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "step_status": "in_progress"
        }
    
    async def _agent_node_async(self, state: ReactAgentState) -> Dict[str, Any]:
        """Agent 节点（异步）"""
        messages = list(state.get("messages", []))
        current_round = state.get("current_round", 0)
        
        if current_round == 0:
            initial_prompt = get_act_prompt(
                task=state["task"],
                current_step=state["current_step"],
                step_index=state["step_index"],
                total_steps=state["total_steps"],
                past_steps="",
                execution_history=""
            )
            messages.append(HumanMessage(content=initial_prompt))
        
        response = await self.async_llm.ainvoke(messages)
        
        new_round = current_round + 1
        
        return {
            "messages": messages + [response],
            "current_round": new_round,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "step_status": "in_progress"
        }
    
    # ========== 执行接口 ==========
    
    def run(self, 
            task: str, 
            current_step: str, 
            step_index: int = 0, 
            total_steps: int = 1,
            max_rounds: int = 3,
            past_steps: str = "") -> Dict[str, Any]:
        """运行 ReAct Agent（同步版本）"""
        initial_state: ReactAgentState = {
            "task": task,
            "current_step": current_step,
            "step_index": step_index,
            "total_steps": total_steps,
            "messages": [],
            "past_steps": [],
            "max_rounds": max_rounds,
            "current_round": 0,
            "iteration_count": 0,
            "step_status": "pending"
        }
        
        try:
            final_state = self.app.invoke(initial_state)
            
            # 检查是否成功完成
            messages = final_state.get("messages", [])
            success = False
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, AIMessage):
                    content = last_msg.content
                    if "完成" in content and "Action: 完成" not in str(getattr(last_msg, 'tool_calls', [])):
                        success = True
            
            return {
                "success": success,
                "messages": messages,
                "iteration_count": final_state.get("iteration_count", 0),
                "current_round": final_state.get("current_round", 0),
                "step_status": final_state.get("step_status", "unknown")
            }
            
        except Exception as e:
            print(f"ReAct Agent 执行异常: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "messages": [],
                "error": str(e),
                "step_status": "error"
            }
    
    async def run_async(self, 
                       task: str, 
                       current_step: str, 
                       step_index: int = 0, 
                       total_steps: int = 1,
                       max_rounds: int = 3,
                       past_steps: str = "") -> Dict[str, Any]:
        """运行 ReAct Agent（异步版本）"""
        initial_state: ReactAgentState = {
            "task": task,
            "current_step": current_step,
            "step_index": step_index,
            "total_steps": total_steps,
            "messages": [],
            "past_steps": [],
            "max_rounds": max_rounds,
            "current_round": 0,
            "iteration_count": 0,
            "step_status": "pending"
        }
        
        try:
            final_state = await self.async_app.ainvoke(initial_state)
            
            messages = final_state.get("messages", [])
            success = False
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, AIMessage):
                    content = last_msg.content
                    if "完成" in content and "Action: 完成" not in str(getattr(last_msg, 'tool_calls', [])):
                        success = True
            
            return {
                "success": success,
                "messages": messages,
                "iteration_count": final_state.get("iteration_count", 0),
                "current_round": final_state.get("current_round", 0),
                "step_status": final_state.get("step_status", "unknown")
            }
            
        except Exception as e:
            print(f"ReAct Agent 执行异常: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "messages": [],
                "error": str(e),
                "step_status": "error"
            }
    
    def stream(self, 
               task: str, 
               current_step: str, 
               step_index: int = 0, 
               total_steps: int = 1,
               max_rounds: int = 3) -> Iterator[Dict[str, Any]]:
        """流式输出（同步版本）"""
        initial_state: ReactAgentState = {
            "task": task,
            "current_step": current_step,
            "step_index": step_index,
            "total_steps": total_steps,
            "messages": [],
            "past_steps": [],
            "max_rounds": max_rounds,
            "current_round": 0,
            "iteration_count": 0,
            "step_status": "pending"
        }
        
        try:
            for event in self.app.stream(initial_state, stream_mode="message"):
                if isinstance(event, tuple):
                    node_name, messages = event
                    if messages and len(messages) > 0:
                        last_msg = messages[-1]
                        yield {
                            "node": node_name,
                            "type": last_msg.type if hasattr(last_msg, 'type') else "unknown",
                            "content": last_msg.content if hasattr(last_msg, 'content') else str(last_msg),
                        }
        except Exception as e:
            yield {"node": "error", "type": "error", "content": f"执行异常: {str(e)}"}
    
    async def stream_async(self, 
                          task: str, 
                          current_step: str, 
                          step_index: int = 0, 
                          total_steps: int = 1,
                          max_rounds: int = 3) -> AsyncIterator[Dict[str, Any]]:
        """流式输出（异步版本）"""
        initial_state: ReactAgentState = {
            "task": task,
            "current_step": current_step,
            "step_index": step_index,
            "total_steps": total_steps,
            "messages": [],
            "past_steps": [],
            "max_rounds": max_rounds,
            "current_round": 0,
            "iteration_count": 0,
            "step_status": "pending"
        }
        
        try:
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
        except Exception as e:
            yield {"node": "error", "type": "error", "content": f"执行异常: {str(e)}"}


# ========== 便捷函数 ==========

def create_react_agent(config: Optional[AgentConfig] = None, tool_manager=None, use_async=True) -> ReactAgent:
    """创建 ReAct Agent 实例"""
    return ReactAgent(config, tool_manager, use_async)


# ========== 测试代码 ==========

def test_react_agent():
    """测试 ReAct Agent"""
    print("=== 测试 ReAct Agent ===\n")
    
    agent = create_react_agent()
    
    # 测试执行
    print("--- 测试 run ---")
    result = agent.run(
        task="检查变电站设备状态",
        current_step="使用YOLO检测当前场景中的物体",
        step_index=0,
        total_steps=3,
        max_rounds=3
    )
    
    print(f"成功: {result['success']}")
    print(f"迭代次数: {result['iteration_count']}")
    print(f"轮数: {result['current_round']}")
    print(f"状态: {result['step_status']}")
    
    # 打印消息
    print("\n--- 消息历史 ---")
    for msg in result.get("messages", []):
        if hasattr(msg, 'content'):
            msg_type = msg.type if hasattr(msg, 'type') else "unknown"
            print(f"[{msg_type}] {msg.content[:200]}...")


if __name__ == "__main__":
    test_react_agent()