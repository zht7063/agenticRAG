"""
agents 模块 - Agent 实现

本模块包含 ScholarRAG 系统的所有 Agent 实现：
- MasterAgent: 主控代理，负责用户交互和 Worker 协调
- BaseWorker: Worker 基类，定义通用接口和行为
- RetrievalWorker: 检索 Worker，负责文献检索和向量搜索
- SQLWorker: SQL Worker，负责数据库查询
- WebSearchWorker: 网络搜索 Worker，负责 MCP 网络搜索
- GenerationWorker: 生成 Worker，负责答案生成和文献综述
"""

from .master import MasterAgent
from .worker_base import BaseWorker

__all__ = [
    "MasterAgent",
    "BaseWorker",
]

