"""
database 模块 - 数据库服务

提供 SQLite 数据库的连接管理和数据访问：

1. connection - 连接管理
   - 数据库连接池
   - 事务管理
   - 连接复用

2. schema - Schema 定义
   - 数据库表结构定义
   - 表创建和迁移
   - 索引管理

3. repository - 数据访问层
   - 论文元数据 CRUD
   - 文献合集管理
   - 研究笔记和实验记录操作
"""

from .connection import DatabaseConnection
from .schema import SchemaManager
from .repository import (
    PaperRepository,
    CollectionRepository,
    NoteRepository,
    ExperimentRepository,
    InspirationRepository,
)

__all__ = [
    "DatabaseConnection",
    "SchemaManager",
    "PaperRepository",
    "CollectionRepository",
    "NoteRepository",
    "ExperimentRepository",
    "InspirationRepository",
]

