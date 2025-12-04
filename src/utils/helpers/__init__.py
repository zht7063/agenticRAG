"""
helpers 模块 - 辅助函数

提供通用的辅助函数：

1. text_processing - 文本处理
   - 文本清洗和规范化
   - 特殊字符处理
   - 编码转换

2. chunking - 文档分块
   - 智能分块策略
   - 语义边界识别
   - 分块参数配置

3. logger - 日志工具
   - 统一日志格式
   - 日志级别控制
   - 文件和控制台输出
"""

from .text_processing import (
    clean_text,
    normalize_whitespace,
    extract_keywords,
)
from .chunking import (
    smart_chunk,
    get_chunk_config,
)
from .logger import (
    get_logger,
    setup_logging,
)

__all__ = [
    "clean_text",
    "normalize_whitespace",
    "extract_keywords",
    "smart_chunk",
    "get_chunk_config",
    "get_logger",
    "setup_logging",
]

