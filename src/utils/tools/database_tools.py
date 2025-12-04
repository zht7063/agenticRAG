"""
Database Tools - 数据库工具

提供 Agent 可调用的数据库操作工具：

1. 论文查询工具
   - query_papers_tool: 查询论文元数据
   - get_paper_detail_tool: 获取论文详情

2. 笔记管理工具
   - add_note_tool: 添加研究笔记
   - query_notes_tool: 查询笔记

3. 实验记录工具
   - query_experiments_tool: 查询实验记录
   - update_experiment_tool: 更新实验状态

4. 合集管理工具
   - query_collections_tool: 查询文献合集
   - add_to_collection_tool: 添加论文到合集

这些工具被包装为 LangChain Tool，可被 Agent 直接调用。
"""

from typing import List, Optional
from langchain_core.tools import tool


@tool
def query_papers_tool(
    keywords: str = None,
    author: str = None,
    year: str = None,
    limit: int = 10
) -> List[dict]:
    """
    查询论文元数据
    
    根据条件查询数据库中的论文信息。
    
    Args:
        keywords: 关键词筛选
        author: 作者筛选
        year: 发表年份筛选
        limit: 返回数量限制
        
    Returns:
        论文信息列表
    """
    pass


@tool
def get_paper_detail_tool(paper_id: int) -> dict:
    """
    获取论文详细信息
    
    Args:
        paper_id: 论文 ID
        
    Returns:
        论文详细信息
    """
    pass


@tool
def add_note_tool(
    paper_id: int,
    content: str,
    note_type: str = "comment",
    page_number: int = None
) -> dict:
    """
    添加研究笔记
    
    为指定论文添加研究笔记。
    
    Args:
        paper_id: 论文 ID
        content: 笔记内容
        note_type: 笔记类型（highlight, comment, question）
        page_number: 页码（可选）
        
    Returns:
        创建的笔记信息
    """
    pass


@tool
def query_notes_tool(paper_id: int = None, note_type: str = None) -> List[dict]:
    """
    查询研究笔记
    
    Args:
        paper_id: 按论文筛选
        note_type: 按类型筛选
        
    Returns:
        笔记列表
    """
    pass


@tool
def query_experiments_tool(status: str = None) -> List[dict]:
    """
    查询实验记录
    
    Args:
        status: 实验状态筛选（planned, running, completed）
        
    Returns:
        实验记录列表
    """
    pass


@tool
def update_experiment_tool(experiment_id: int, status: str = None, results: str = None) -> dict:
    """
    更新实验记录
    
    Args:
        experiment_id: 实验 ID
        status: 新状态
        results: 实验结果（JSON 格式）
        
    Returns:
        更新后的实验信息
    """
    pass


@tool
def query_collections_tool() -> List[dict]:
    """
    查询所有文献合集
    
    Returns:
        合集列表
    """
    pass


@tool
def add_to_collection_tool(collection_id: int, paper_id: int) -> dict:
    """
    添加论文到合集
    
    Args:
        collection_id: 合集 ID
        paper_id: 论文 ID
        
    Returns:
        操作结果
    """
    pass


@tool
def query_inspirations_tool(status: str = None, priority: str = None) -> List[dict]:
    """
    查询研究灵感
    
    Args:
        status: 状态筛选（new, exploring, archived）
        priority: 优先级筛选（high, medium, low）
        
    Returns:
        灵感列表
    """
    pass

