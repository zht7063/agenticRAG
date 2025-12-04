"""
mcp 模块 - MCP 协议服务

通过 Model Context Protocol (MCP) 接入外部信息源：

1. MCPWebSearch - 网络搜索服务
   - 学术论文搜索
   - 通用网页搜索
   - 搜索结果处理

2. MCPFetch - 网页抓取服务
   - URL 内容获取
   - 页面解析
   - 内容提取

3. MCPTime - 时间服务
   - 当前时间获取
   - 时区处理
"""

from .MCPWebSearch import MCPWebSearch
from .MCPFetch import MCPFetch
from .MCPTime import MCPTime

__all__ = [
    "MCPWebSearch",
    "MCPFetch",
    "MCPTime",
]

