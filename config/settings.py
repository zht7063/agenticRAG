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

使用方式：
1. 在应用启动时调用 initialize() 函数，确保配置正确且目录已创建：
   from config.settings import initialize
   initialize()

2. initialize() 函数会：
   - 自动创建所有必要的数据存储目录（data、vectorstore、documents、logs）
   - 验证所有配置参数的有效性（API 密钥、URL、路径等）
   - 如果配置无效，会抛出清晰的 ValueError 异常
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
# 配置验证辅助函数
# ============================================================

def _validate_api_key(api_key: str, key_name: str) -> None:
    """
    验证 API 密钥
    
    Args:
        api_key: API 密钥值
        key_name: 密钥名称（用于错误提示）
        
    Raises:
        ValueError: 如果 API 密钥为空或无效
    """
    if not api_key:
        raise ValueError(f"{key_name} 未设置，请在 .env 文件中配置")
    if not isinstance(api_key, str) or len(api_key.strip()) == 0:
        raise ValueError(f"{key_name} 格式无效，应为非空字符串")


def _validate_url(url: str, url_name: str) -> None:
    """
    验证 URL 格式
    
    Args:
        url: URL 字符串
        url_name: URL 名称（用于错误提示）
        
    Raises:
        ValueError: 如果 URL 格式无效
    """
    if url:
        url = url.strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError(f"{url_name} 格式无效，应以 http:// 或 https:// 开头")


def _validate_path_writable(path: Path, path_name: str) -> None:
    """
    验证路径是否可写
    
    Args:
        path: 路径对象
        path_name: 路径名称（用于错误提示）
        
    Raises:
        ValueError: 如果路径不可写
    """
    # 如果是文件路径（有扩展名），检查父目录；否则检查目录本身
    if path.suffix:
        parent_dir = path.parent
    else:
        parent_dir = path
    
    parent_dir = parent_dir.resolve()
    
    if not parent_dir.exists():
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise ValueError(f"{path_name} 的父目录无法创建: {e}")
    
    if not os.access(parent_dir, os.W_OK):
        raise ValueError(f"{path_name} 的父目录不可写: {parent_dir}")


# ============================================================
# 确保必要目录存在
# ============================================================

def ensure_directories():
    """确保必要的目录存在"""
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 配置初始化函数
# ============================================================

def initialize() -> None:
    """
    初始化配置：创建数据存储目录并验证所有配置参数
    
    此函数会：
    1. 创建所有必要的数据存储目录（data 目录及其子目录）
    2. 验证所有配置参数的有效性（API 密钥、URL、路径等）
    
    使用方式：
    在应用启动时调用此函数，确保配置正确且目录已创建：
        from config.settings import initialize
        initialize()
    
    Raises:
        ValueError: 如果配置验证失败
    """
    # 创建所有必要的数据存储目录
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # 验证 API 密钥配置
    _validate_api_key(OPENAI_API_KEY, "OPENAI_API_KEY")
    
    # 验证 URL 配置（如果提供）
    if OPENAI_BASE_URL:
        _validate_url(OPENAI_BASE_URL, "OPENAI_BASE_URL")
    
    # 验证路径配置
    _validate_path_writable(SQLITE_DB_PATH, "SQLITE_DB_PATH")
    _validate_path_writable(Path(CHROMA_PERSIST_DIRECTORY), "CHROMA_PERSIST_DIRECTORY")
    _validate_path_writable(DOCUMENTS_DIR, "DOCUMENTS_DIR")
    _validate_path_writable(LOG_FILE, "LOG_FILE")
