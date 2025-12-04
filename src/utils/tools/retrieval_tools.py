"""
Retrieval Tools - 检索工具

提供 Agent 可调用的检索工具函数：

1. semantic_search_tool
   - 基于向量的语义检索
   - 支持 PDF 和 HTML 来源
   - 返回相关文档和来源信息

2. keyword_search_tool
   - 基于关键词的精确检索
   - 支持布尔查询
   - 适用于精确匹配场景

3. hybrid_search_tool
   - 语义 + 关键词的混合检索
   - 结果融合和重排序
   - 适用于复杂查询场景

这些工具被包装为 LangChain Tool，可被 Agent 直接调用。
"""

from typing import List, Optional
from langchain_core.tools import tool


@tool
def semantic_search_tool(query: str, top_k: int = 5, source_type: str = "all") -> List[dict]:
    """
    语义向量检索工具
    
    根据查询文本在向量库中进行语义相似度检索，
    返回最相关的文档片段及其来源信息。
    
    Args:
        query: 查询文本
        top_k: 返回结果数量，默认 5
        source_type: 来源类型，可选 "pdf", "html", "all"
        
    Returns:
        检索结果列表，每项包含 content, source, score 字段
    """
    pass


@tool
def keyword_search_tool(keywords: str, top_k: int = 5) -> List[dict]:
    """
    关键词检索工具
    
    根据关键词在文档库中进行精确匹配检索。
    
    Args:
        keywords: 搜索关键词，多个关键词用空格分隔
        top_k: 返回结果数量，默认 5
        
    Returns:
        检索结果列表
    """
    pass


@tool
def hybrid_search_tool(query: str, top_k: int = 5) -> List[dict]:
    """
    混合检索工具
    
    结合语义检索和关键词检索，对结果进行融合和重排序，
    适用于复杂查询场景。
    
    Args:
        query: 查询文本
        top_k: 返回结果数量，默认 5
        
    Returns:
        融合后的检索结果列表
    """
    pass


@tool
def search_by_paper_title(title: str) -> List[dict]:
    """
    按论文标题检索
    
    在已存储的论文中搜索匹配的标题。
    
    Args:
        title: 论文标题（支持模糊匹配）
        
    Returns:
        匹配的论文信息列表
    """
    pass


@tool
def search_by_author(author: str) -> List[dict]:
    """
    按作者检索论文
    
    Args:
        author: 作者姓名
        
    Returns:
        该作者的论文列表
    """
    pass

