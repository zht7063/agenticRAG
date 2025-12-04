"""
Repository - 数据访问层

提供各数据表的 CRUD 操作接口：

1. PaperRepository - 论文元数据管理
   - 论文的增删改查
   - 按关键词、作者、时间检索
   - 批量导入和导出

2. CollectionRepository - 文献合集管理
   - 合集的创建和维护
   - 论文与合集的关联
   - 合集内容检索

3. NoteRepository - 研究笔记管理
   - 笔记的增删改查
   - 按论文检索笔记
   - 笔记类型筛选

4. ExperimentRepository - 实验记录管理
   - 实验的创建和更新
   - 实验状态追踪
   - 关联论文管理

5. InspirationRepository - 研究灵感管理
   - 灵感记录
   - 优先级和状态管理
   - 来源论文关联
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Paper:
    """论文数据模型"""
    id: Optional[int] = None
    title: str = ""
    authors: str = ""
    abstract: str = ""
    keywords: str = ""
    publish_date: str = ""
    venue: str = ""
    doi: str = ""
    url: str = ""
    pdf_path: str = ""
    vector_ids: str = ""
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class Collection:
    """文献合集数据模型"""
    id: Optional[int] = None
    name: str = ""
    description: str = ""
    tags: str = ""
    created_at: datetime = None


@dataclass
class Note:
    """研究笔记数据模型"""
    id: Optional[int] = None
    paper_id: Optional[int] = None
    content: str = ""
    note_type: str = ""  # highlight, comment, question
    page_number: Optional[int] = None
    created_at: datetime = None


@dataclass
class Experiment:
    """实验记录数据模型"""
    id: Optional[int] = None
    name: str = ""
    description: str = ""
    parameters: str = ""  # JSON
    results: str = ""     # JSON
    related_papers: str = ""
    status: str = ""      # planned, running, completed
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class Inspiration:
    """研究灵感数据模型"""
    id: Optional[int] = None
    title: str = ""
    content: str = ""
    source_papers: str = ""
    priority: str = ""    # high, medium, low
    status: str = ""      # new, exploring, archived
    created_at: datetime = None


class PaperRepository:
    """论文元数据仓储"""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def create(self, paper: Paper) -> int:
        """创建论文记录"""
        pass
    
    def get_by_id(self, paper_id: int) -> Optional[Paper]:
        """根据 ID 获取论文"""
        pass
    
    def get_all(self, limit: int = 100, offset: int = 0) -> List[Paper]:
        """获取所有论文"""
        pass
    
    def search_by_title(self, title: str) -> List[Paper]:
        """按标题搜索"""
        pass
    
    def search_by_keywords(self, keywords: str) -> List[Paper]:
        """按关键词搜索"""
        pass
    
    def search_by_author(self, author: str) -> List[Paper]:
        """按作者搜索"""
        pass
    
    def update(self, paper: Paper) -> bool:
        """更新论文记录"""
        pass
    
    def delete(self, paper_id: int) -> bool:
        """删除论文记录"""
        pass


class CollectionRepository:
    """文献合集仓储"""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def create(self, collection: Collection) -> int:
        """创建合集"""
        pass
    
    def get_by_id(self, collection_id: int) -> Optional[Collection]:
        """根据 ID 获取合集"""
        pass
    
    def get_all(self) -> List[Collection]:
        """获取所有合集"""
        pass
    
    def add_paper(self, collection_id: int, paper_id: int) -> bool:
        """添加论文到合集"""
        pass
    
    def remove_paper(self, collection_id: int, paper_id: int) -> bool:
        """从合集移除论文"""
        pass
    
    def get_papers(self, collection_id: int) -> List[Paper]:
        """获取合集中的论文"""
        pass
    
    def update(self, collection: Collection) -> bool:
        """更新合集"""
        pass
    
    def delete(self, collection_id: int) -> bool:
        """删除合集"""
        pass


class NoteRepository:
    """研究笔记仓储"""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def create(self, note: Note) -> int:
        """创建笔记"""
        pass
    
    def get_by_id(self, note_id: int) -> Optional[Note]:
        """根据 ID 获取笔记"""
        pass
    
    def get_by_paper(self, paper_id: int) -> List[Note]:
        """获取论文的所有笔记"""
        pass
    
    def get_by_type(self, note_type: str) -> List[Note]:
        """按笔记类型获取"""
        pass
    
    def update(self, note: Note) -> bool:
        """更新笔记"""
        pass
    
    def delete(self, note_id: int) -> bool:
        """删除笔记"""
        pass


class ExperimentRepository:
    """实验记录仓储"""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def create(self, experiment: Experiment) -> int:
        """创建实验记录"""
        pass
    
    def get_by_id(self, experiment_id: int) -> Optional[Experiment]:
        """根据 ID 获取实验"""
        pass
    
    def get_all(self) -> List[Experiment]:
        """获取所有实验"""
        pass
    
    def get_by_status(self, status: str) -> List[Experiment]:
        """按状态获取实验"""
        pass
    
    def update(self, experiment: Experiment) -> bool:
        """更新实验记录"""
        pass
    
    def update_status(self, experiment_id: int, status: str) -> bool:
        """更新实验状态"""
        pass
    
    def delete(self, experiment_id: int) -> bool:
        """删除实验记录"""
        pass


class InspirationRepository:
    """研究灵感仓储"""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def create(self, inspiration: Inspiration) -> int:
        """创建灵感记录"""
        pass
    
    def get_by_id(self, inspiration_id: int) -> Optional[Inspiration]:
        """根据 ID 获取灵感"""
        pass
    
    def get_all(self) -> List[Inspiration]:
        """获取所有灵感"""
        pass
    
    def get_by_priority(self, priority: str) -> List[Inspiration]:
        """按优先级获取"""
        pass
    
    def get_by_status(self, status: str) -> List[Inspiration]:
        """按状态获取"""
        pass
    
    def update(self, inspiration: Inspiration) -> bool:
        """更新灵感记录"""
        pass
    
    def delete(self, inspiration_id: int) -> bool:
        """删除灵感记录"""
        pass

