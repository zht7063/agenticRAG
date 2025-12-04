"""
Search Tools - 搜索工具

提供 Agent 可调用的网络搜索工具：

1. web_search_tool
   - 通用网络搜索
   - 通过 MCP 协议调用
   - 返回搜索结果摘要

2. academic_search_tool
   - 学术论文搜索
   - 获取论文基本信息
   - 支持引用数排序

3. fetch_url_tool
   - 获取指定 URL 内容
   - 网页内容解析
   - 学术页面处理

这些工具被包装为 LangChain Tool，可被 Agent 直接调用。
"""

from typing import List, Optional
from langchain_core.tools import tool


@tool
def web_search_tool(query: str, max_results: int = 10) -> List[dict]:
    """
    通用网络搜索工具
    
    使用 MCP 协议进行网络搜索，获取相关网页信息。
    
    Args:
        query: 搜索查询
        max_results: 最大返回结果数
        
    Returns:
        搜索结果列表，每项包含 title, url, snippet 字段
    """
    pass


@tool
def academic_search_tool(query: str, max_results: int = 10) -> List[dict]:
    """
    学术论文搜索工具
    
    搜索学术论文，返回论文基本信息。
    
    Args:
        query: 搜索查询（论文标题、作者、关键词等）
        max_results: 最大返回结果数
        
    Returns:
        论文信息列表，包含 title, authors, abstract, url, citations 字段
    """
    pass


@tool
def fetch_url_tool(url: str) -> dict:
    """
    获取 URL 内容工具
    
    抓取指定 URL 的网页内容并解析。
    
    Args:
        url: 目标 URL
        
    Returns:
        包含 title, content, metadata 的字典
    """
    pass


@tool
def search_by_doi_tool(doi: str) -> dict:
    """
    按 DOI 搜索论文
    
    Args:
        doi: 论文 DOI
        
    Returns:
        论文详细信息
    """
    pass


@tool
def get_paper_citations_tool(paper_url: str) -> List[dict]:
    """
    获取论文引用信息
    
    Args:
        paper_url: 论文页面 URL
        
    Returns:
        引用论文列表
    """
    pass

