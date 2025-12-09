"""
agents 模块 - Agent 实现

本模块包含 ScholarRAG 系统的所有 Agent 实现：
- MasterAgent: 主控代理，负责用户交互、Worker 协调和答案生成
- BaseAgent: Agent 基类，定义通用接口和行为
- RetrievalAgent: 检索代理，负责文献检索和向量搜索
- ResourceAgent: 资源代理，负责处理 PDF 和 URL 资源
- sql_worker: SQL Agent，负责数据库查询
"""

# 导入基类和具体实现
from .agent_base import BaseAgent
from .master import MasterAgent
from .retrieval_agent import RetrievalAgent
from .resource_agent import ResourceAgent
from .sql_agent import sql_worker

__all__ = [
    "BaseAgent",
    "MasterAgent",
    "RetrievalAgent",
    "ResourceAgent",
    "sql_worker",
]

