"""
config 模块 - 配置管理

集中管理 ScholarRAG 系统的所有配置项：

1. 环境变量配置
   - API 密钥
   - 服务端点

2. 路径配置
   - 项目根目录
   - 数据目录
   - 日志目录

3. 模型配置
   - LLM 参数
   - Embedding 参数

4. 数据库配置
   - SQLite 路径
   - 向量存储路径
"""

from .settings import (
    project_root,
    PROJECT_ROOT,
    DATA_DIR,
    VECTORSTORE_DIR,
    DOCUMENTS_DIR,
    SQLITE_DB_PATH,
    OPENAI_BASE_URL,
    OPENAI_API_KEY,
    DASHSCOPE_API_KEY,
    DEFAULT_LLM_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    ensure_directories,
)

__all__ = [
    "project_root",
    "PROJECT_ROOT",
    "DATA_DIR",
    "VECTORSTORE_DIR",
    "DOCUMENTS_DIR",
    "SQLITE_DB_PATH",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "DASHSCOPE_API_KEY",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "ensure_directories",
]

