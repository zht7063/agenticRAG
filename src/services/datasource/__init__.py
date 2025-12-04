"""
datasource 模块 - 数据源服务

提供多种数据源的向量化存储和检索能力：

1. PDFVectorStore
   - PDF 文档解析和分块
   - 向量化存储和检索
   - 元数据管理

2. HTMLVectorStore
   - 网页内容抓取和解析
   - 正文提取和清洗
   - 向量化存储

3. SQLVectorStore
   - 数据库 Schema 语义化
   - 表结构和字段描述向量化
   - 支持 NL2SQL 的语义匹配
"""

from .PDFVectorStore import PDFVectorStore
from .HTMLVectorStore import HTMLVectorStore
from .SQLVectorStore import SQLVectorStore

__all__ = [
    "PDFVectorStore",
    "HTMLVectorStore",
    "SQLVectorStore",
]

