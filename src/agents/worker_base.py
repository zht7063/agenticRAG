"""
Worker Base - Worker 基类

定义所有 Worker 的通用接口和基础行为：

1. 工具管理
   - 工具注册和初始化
   - 工具调用封装
   - 工具执行结果处理

2. 任务执行
   - 标准化的任务输入格式
   - 任务执行主流程
   - 异常处理和重试机制

3. 结果格式化
   - 统一的结果数据结构
   - 来源信息追踪
   - 执行状态和元数据

所有专业化 Worker 都继承此基类：
- RetrievalWorker: 文献检索
- SQLWorker: 数据库查询
- WebSearchWorker: 网络搜索
- GenerationWorker: 内容生成
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseWorker(ABC):
    """Worker 基类，定义通用接口"""
    
    def __init__(self, name: str, tools: list = None):
        self.name = name
        self.tools = tools or []
        self.llm = None
    
    @abstractmethod
    def execute(self, task: dict) -> dict:
        """
        执行任务并返回结构化结果
        
        Args:
            task: 任务描述，包含 type, query, context 等字段
            
        Returns:
            执行结果字典，包含 status, data, sources 等字段
        """
        pass
    
    def format_result(self, raw_result: Any, sources: list = None) -> dict:
        """
        格式化执行结果为标准结构
        
        Args:
            raw_result: 原始执行结果
            sources: 来源信息列表
            
        Returns:
            标准化的结果字典
        """
        return {
            "worker": self.name,
            "status": "success",
            "data": raw_result,
            "sources": sources or []
        }
    
    def format_error(self, error: Exception) -> dict:
        """格式化错误结果"""
        return {
            "worker": self.name,
            "status": "error",
            "error": str(error),
            "data": None,
            "sources": []
        }
    
    def register_tool(self, tool) -> None:
        """注册工具到 Worker"""
        pass
    
    def get_tools(self) -> list:
        """获取 Worker 的工具列表"""
        return self.tools

