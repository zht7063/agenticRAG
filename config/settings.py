"""
Settings - 应用配置管理

集中管理 ScholarRAG 系统的所有配置：

1. 环境变量
   - OPENAI_BASE_URL: OpenAI API 基础 URL
   - OPENAI_API_KEY: OpenAI API 密钥
   - DASHSCOPE_API_KEY: 阿里云 DashScope API 密钥

2. LLM 配置
   - 默认模型选择
   - 模型参数配置
   - Embedding 模型配置

3. 数据库配置
   - SQLite 数据库路径
   - 向量数据库路径
   - 文档存储路径

4. MCP 服务配置
   - MCP 服务端点
   - 连接参数

5. 日志配置
   - 日志级别
   - 日志文件路径
"""

import os
from pathlib import Path
import subprocess
from dotenv import load_dotenv


# 加载环境变量
load_dotenv()


# ============================================================
# 项目路径配置
# ============================================================

def project_root() -> Path:
    """
    获取项目根目录
    
    通过 git 命令获取仓库根目录路径。
    """
    git_root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        text=True
    ).strip()
    return Path(git_root)


# 项目根目录
PROJECT_ROOT = project_root()

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
DOCUMENTS_DIR = DATA_DIR / "documents"

# 数据库文件
SQLITE_DB_PATH = DATA_DIR / "scholar.db"


# ============================================================
# API 密钥配置
# ============================================================

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")


# ============================================================
# LLM 模型配置
# ============================================================

# 默认 LLM 配置
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_LLM_TEMPERATURE = 0.7
DEFAULT_LLM_MAX_TOKENS = 4096

# Embedding 模型配置
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"


# ============================================================
# 向量数据库配置
# ============================================================

# Chroma 配置
CHROMA_COLLECTION_NAME = "scholar_documents"
CHROMA_PERSIST_DIRECTORY = str(VECTORSTORE_DIR)

# 检索配置
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.7


# ============================================================
# 文档处理配置
# ============================================================

# 分块配置
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


# ============================================================
# MCP 服务配置
# ============================================================

MCP_SERVERS: dict[str, dict] = {
    # 示例配置
    # "bing-search": {
    #     "transport": "streamable_http",
    #     "url": "https://mcp.example.com/search"
    # }
}


# ============================================================
# 日志配置
# ============================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = DATA_DIR / "logs" / "scholar_rag.log"


# ============================================================
# 确保必要目录存在
# ============================================================

def ensure_directories():
    """确保必要的目录存在"""
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
